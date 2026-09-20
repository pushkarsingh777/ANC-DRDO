#!/usr/bin/env python3
"""
Real-Time Audio Noise Cancellation & Speech Enhancement Inference (FR-3).

Enhance any noisy audio file (.wav, .flac, .mp3, etc.) using a trained Causal Conv-TasNet
checkpoint (.pt) or ONNX model (.onnx).

Features:
- Single file or batch directory processing
- Automatic channel mixdown to mono & resampling to 16 kHz
- Support for both PyTorch (.pt) checkpoints and ONNX Runtime (.onnx)
- Latency and Real-Time Factor (RTF) profiling
- Peak normalization to prevent clipping

Usage:
    # Basic single-file enhancement:
    python scripts/infer.py --input path/to/noisy.wav --output path/to/clean.wav

    # Using a specific checkpoint:
    python scripts/infer.py --input noisy.wav --checkpoint checkpoints/best_checkpoint.pt

    # Using ONNX model on CPU / Edge:
    python scripts/infer.py --input noisy.wav --onnx checkpoints/causal_conv_tasnet.onnx

    # Process an entire folder:
    python scripts/infer.py --input-dir test_audio/ --output-dir enhanced_audio/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
import soundfile as sf
import numpy as np

# Ensure anc-model-training is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CausalConvTasNet


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Loads audio file, converts to mono float32, and resamples if necessary."""
    data, sr = sf.read(path, dtype="float32")
    
    # Convert stereo/multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample if sample rate != target_sr (16 kHz)
    if sr != target_sr:
        try:
            # pyrefly: ignore [missing-import]
            import torchaudio.functional as AF
            t_data = torch.from_numpy(data).unsqueeze(0)
            resampled = AF.resample(t_data, orig_freq=sr, new_freq=target_sr)
            data = resampled.squeeze(0).numpy()
            sr = target_sr
        except Exception:
            duration = len(data) / sr
            new_length = int(duration * target_sr)
            data = np.interp(
                np.linspace(0, len(data), new_length, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
            sr = target_sr
            
    return data, sr


def enhance_audio_pytorch(
    model: torch.nn.Module,
    audio: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Runs enhancement using PyTorch model."""
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        enhanced_t = model(audio_t)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_proc = time.perf_counter() - t0
    
    enhanced = enhanced_t.squeeze(0).cpu().numpy()
    return enhanced, t_proc


def enhance_audio_onnx(
    session,
    audio: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Runs enhancement using ONNX Runtime session."""
    input_name = session.get_inputs()[0].name
    input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)  # (1, T)
    
    t0 = time.perf_counter()
    ort_outs = session.run(None, {input_name: input_tensor})
    t_proc = time.perf_counter() - t0
    
    enhanced = np.squeeze(ort_outs[0], axis=0)
    return enhanced, t_proc


def save_audio(enhanced: np.ndarray, out_path: str, sr: int = 16000, normalize: bool = True):
    """Saves enhanced audio with peak-normalization if clipping occurs."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    peak = np.max(np.abs(enhanced))
    if normalize and peak > 0.99:
        enhanced = enhanced / (peak + 1e-8) * 0.95
        
    sf.write(out_path, enhanced, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Enhance Audio with Causal Conv-TasNet")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input noisy audio file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output enhanced audio file")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of noisy audio files to batch process")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save enhanced files")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_checkpoint.pt", help="Path to PyTorch checkpoint .pt")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model file (.onnx) for inference")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    args = parser.parse_args()

    # Determine input files
    if args.input:
        input_files = [args.input]
    elif args.input_dir:
        exts = (".wav", ".flac", ".mp3", ".ogg")
        input_files = [str(p) for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in exts]
        if not input_files:
            print(f"Error: No audio files found in directory {args.input_dir}")
            sys.exit(1)
    else:
        print("Error: Please provide either --input <file> or --input-dir <dir>.")
        parser.print_help()
        sys.exit(1)

    # Initialize model (ONNX or PyTorch)
    use_onnx = args.onnx is not None
    if use_onnx:
        try:
            # pyrefly: ignore [missing-import]
            import onnxruntime as ort
            print(f"Loading ONNX Model: {args.onnx}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            session = ort.InferenceSession(args.onnx, providers=providers)
            model = None
            device = None
            active_provider = session.get_providers()[0]
            print(f"ONNX Runtime Provider: {active_provider}")
        except ImportError:
            print("Error: onnxruntime is not installed. Install with `pip install onnxruntime` or `onnxruntime-gpu`.")
            sys.exit(1)
    else:
        dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(dev_str)
        print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
            sys.exit(1)
            
        print(f"Loading Checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("config", {})
        
        model = CausalConvTasNet.from_config(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        session = None

    print(f"\nProcessing {len(input_files)} audio file(s)...")
    print("-" * 75)

    for idx, in_file in enumerate(input_files, 1):
        if not os.path.exists(in_file):
            print(f"[{idx}/{len(input_files)}] File not found: {in_file}")
            continue

        # Determine output path
        if args.output and len(input_files) == 1:
            out_file = args.output
        elif args.output_dir:
            rel = os.path.relpath(in_file, args.input_dir) if args.input_dir else os.path.basename(in_file)
            stem = os.path.splitext(rel)[0]
            out_file = os.path.join(args.output_dir, f"{stem}_enhanced.wav")
        else:
            p = Path(in_file)
            out_file = str(p.with_stem(p.stem + "_enhanced"))

        # Load audio
        audio, sr = load_audio(in_file, target_sr=16000)
        audio_dur = len(audio) / sr

        # Run inference
        if use_onnx:
            enhanced, proc_time = enhance_audio_onnx(session, audio)
        else:
            enhanced, proc_time = enhance_audio_pytorch(model, audio, device)

        rtf = proc_time / max(audio_dur, 1e-6)

        # Save enhanced audio
        save_audio(enhanced, out_file, sr=sr)

        print(f"[{idx}/{len(input_files)}] Input:    {in_file} ({audio_dur:.2f}s)")
        print(f"        Output:   {out_file}")
        print(f"        Latency:  {proc_time * 1000:.1f} ms | RTF: {rtf:.4f} ({1/rtf:.1f}x real-time)")
        print("-" * 75)

    print(f"\nAll enhancement tasks completed successfully!")


if __name__ == "__main__":
    main()
