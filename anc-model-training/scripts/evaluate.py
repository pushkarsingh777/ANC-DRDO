#!/usr/bin/env python3
"""
Comprehensive Model Evaluation & Benchmarking Harness (FR-2.3, PRD section 3.2).

Evaluates a trained Causal Conv-TasNet checkpoint on the held-out test split:
- Computes Segmental SNR Improvement (target > 15 dB)
- Computes STOI (Short-Time Objective Intelligibility, target > 0.85)
- Computes PESQ (ITU-T P.862 Perceptual Evaluation, target > 2.5)
- Profiles Real-Time Factor (RTF)
- Generates a per-noise-category breakdown table
- Saves sample enhanced WAV files for subjective listening

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_checkpoint.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_checkpoint.pt --save-samples 10
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml
import soundfile as sf
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CausalConvTasNet
from src.dataset import SpeechEnhancementDataset, collate_fn_pad
from src.metrics import (
    calculate_snr_improvement,
    calculate_stoi,
    calculate_pesq,
    profile_real_time_factor,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speech Enhancement Model on Test Set")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pt", help="Path to checkpoint .pt")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--manifest", default=None, help="Override test manifest path")
    parser.add_argument("--save-samples", type=int, default=5, help="Number of sample audio triplets to save")
    parser.add_argument("--out-dir", default="eval_results", help="Directory to save evaluation results and wavs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on device: {device}")

    # Load configuration
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("config", {})
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ckpt = None

    model = CausalConvTasNet.from_config(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.checkpoint} (Epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found. Running with uninitialized model for testing.")
    model.eval()

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = os.path.join(cfg["paths"]["manifests_dir"], "mixtures_test.csv")

    test_dataset = SpeechEnhancementDataset(
        manifest_path=manifest_path,
        sample_rate=cfg["data"]["sample_rate"],
        segment_length_s=None,  # Evaluate on full utterance
        is_train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )
    print(f"Evaluating {len(test_dataset)} utterances from {manifest_path}...\n")

    os.makedirs(args.out_dir, exist_ok=True)
    wavs_dir = os.path.join(args.out_dir, "samples")
    if args.save_samples > 0:
        os.makedirs(wavs_dir, exist_ok=True)

    results_by_cat: dict[str, list[dict]] = {}
    all_snr_improvements = []
    all_stoi_scores = []
    all_pesq_scores = []

    samples_saved = 0
    sr = cfg["data"]["sample_rate"]

    with torch.no_grad():
        for item in tqdm(test_loader, desc="Evaluating"):
            mix_t = item["mixture"].to(device)  # (1, T)
            clean_t = item["clean"].to(device)
            cat = item["noise_categories"][0]
            mix_id = item["mixture_ids"][0]

            est_t = model(mix_t)

            mix_np = mix_t.squeeze().cpu().numpy()
            clean_np = clean_t.squeeze().cpu().numpy()
            est_np = est_t.squeeze().cpu().numpy()

            # Metrics
            snr_imp = calculate_snr_improvement(mix_np, est_np, clean_np)
            stoi_score = calculate_stoi(clean_np, est_np, sr)
            pesq_score = calculate_pesq(clean_np, est_np, sr)

            rec = {
                "id": mix_id,
                "category": cat,
                "snr_imp": snr_imp,
                "stoi": stoi_score,
                "pesq": pesq_score,
            }
            results_by_cat.setdefault(cat, []).append(rec)
            all_snr_improvements.append(snr_imp)
            all_stoi_scores.append(stoi_score)
            if pesq_score > 0:
                all_pesq_scores.append(pesq_score)

            # Save audio samples
            if samples_saved < args.save_samples:
                sf.write(os.path.join(wavs_dir, f"{mix_id}_mixture.wav"), mix_np, sr)
                sf.write(os.path.join(wavs_dir, f"{mix_id}_enhanced.wav"), est_np, sr)
                sf.write(os.path.join(wavs_dir, f"{mix_id}_clean.wav"), clean_np, sr)
                samples_saved += 1

    # Profile Real-Time Factor
    rtf, avg_time = profile_real_time_factor(model, audio_duration_s=2.0, sr=sr, device=device.type)

    print("\n" + "=" * 80)
    print("                      SPEECH ENHANCEMENT BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Noise Category':<18} | {'Utterances':<10} | {'Delta SNR (dB)':<15} | {'STOI (0-1)':<12} | {'PESQ (wb)':<10}")
    print("-" * 80)

    for cat, items in sorted(results_by_cat.items()):
        n = len(items)
        mean_snr = np.mean([x["snr_imp"] for x in items])
        mean_stoi = np.mean([x["stoi"] for x in items])
        pesq_vals = [x["pesq"] for x in items if x["pesq"] > 0]
        mean_pesq = np.mean(pesq_vals) if pesq_vals else 0.0
        print(f"{cat:<18} | {n:<10} | {mean_snr:<15.2f} | {mean_stoi:<12.3f} | {mean_pesq:<10.2f}")

    print("-" * 80)
    mean_all_snr = float(np.mean(all_snr_improvements))
    mean_all_stoi = float(np.mean(all_stoi_scores))
    mean_all_pesq = float(np.mean(all_pesq_scores)) if all_pesq_scores else 0.0

    print(f"{'OVERALL AVERAGE':<18} | {len(all_snr_improvements):<10} | {mean_all_snr:<15.2f} | {mean_all_stoi:<12.3f} | {mean_all_pesq:<10.2f}")
    print("=" * 80)

    print("\nReal-Time Performance Profile:")
    print(f"  - Real-Time Factor (RTF): {rtf:.4f} (Target < 0.5 — {'PASSED' if rtf < 0.5 else 'SUB-OPTIMAL'})")
    print(f"  - 2.0s Audio Processed in: {avg_time * 1000:.2f} ms ({avg_time / 2.0 * 100:.1f}% of real-time)")
    if samples_saved > 0:
        print(f"  - Sample audio files saved to: {wavs_dir}/")


if __name__ == "__main__":
    main()
