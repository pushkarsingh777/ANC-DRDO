#!/usr/bin/env python3
"""
Create small synthetic speech-like and noise-like wav files so the full
pipeline (build_manifests.py -> generate_dataset.py) can be exercised and
verified without waiting on real corpus downloads.

"Speech" = band-limited harmonic tones with amplitude envelopes, roughly
mimicking voiced/unvoiced alternation. NOT real speech — only for wiring
tests, not for training. Swap in real data via scripts/download_data.py
before actually training a model.

Usage:
    python scripts/generate_synthetic_smoke_data.py --out_root data --n_speakers 4 --utts_per_speaker 5
"""
import argparse
import os

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf


def synth_speech_like(duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(100, 220)  # pitch range
    sig = np.zeros(n, dtype=np.float32)
    for harmonic in range(1, 6):
        sig += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    # amplitude envelope to fake voiced/unvoiced syllable structure
    n_syll = max(int(duration_s * 3), 1)
    env = np.zeros(n, dtype=np.float32)
    seg = n // n_syll
    for i in range(n_syll):
        start = i * seg
        end = min(start + seg, n)
        local = np.hanning(max(end - start, 1))
        env[start:end] = local[: end - start] * rng.uniform(0.5, 1.0)
    sig = sig * env
    sig += 0.01 * rng.standard_normal(n)  # tiny breath noise
    sig /= (np.max(np.abs(sig)) + 1e-8)
    return (sig * 0.7).astype(np.float32)


def synth_noise(kind: str, duration_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    if kind == "impulsive":
        # sparse sharp clicks/bursts to stand in for gunshots/artillery
        sig = np.zeros(n, dtype=np.float32)
        n_bursts = max(int(duration_s * rng.uniform(0.5, 1.5)), 1)
        for _ in range(n_bursts):
            pos = int(rng.uniform(0, max(n - 200, 1)))
            burst_len = int(sr * rng.uniform(0.01, 0.05))
            decay = np.exp(-np.linspace(0, 12, burst_len))
            sig[pos:pos + burst_len] += rng.standard_normal(burst_len) * decay
        sig /= (np.max(np.abs(sig)) + 1e-8)
        return (sig * 0.9).astype(np.float32)
    if kind == "rotor":
        f0 = rng.uniform(15, 35)  # low-frequency blade-pass-like thump
        sig = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
        sig += 0.05 * rng.standard_normal(n)
        return (sig / (np.max(np.abs(sig)) + 1e-8) * 0.6).astype(np.float32)
    if kind == "vehicle":
        f0 = rng.uniform(60, 140)  # engine hum
        drift = 1.0 + 0.05 * np.sin(2 * np.pi * 0.2 * t)  # slow RPM drift
        sig = 0.5 * np.sin(2 * np.pi * f0 * drift * t)
        sig += 0.1 * rng.standard_normal(n)
        return (sig / (np.max(np.abs(sig)) + 1e-8) * 0.6).astype(np.float32)
    if kind == "ambient":
        # filtered white noise, wind/crowd-like
        white = rng.standard_normal(n).astype(np.float32)
        kernel = np.ones(15) / 15
        sig = np.convolve(white, kernel, mode="same")
        return (sig / (np.max(np.abs(sig)) + 1e-8) * 0.4).astype(np.float32)
    if kind == "sirens":
        f0 = 600 + 400 * np.sin(2 * np.pi * 0.5 * t)  # warbling siren
        sig = np.sin(2 * np.pi * f0 * t / sr * sr)  # instantaneous freq via phase accumulation approx
        sig = np.sin(2 * np.pi * np.cumsum(f0) / sr)
        return (sig * 0.5).astype(np.float32)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_speakers", type=int, default=4)
    ap.add_argument("--utts_per_speaker", type=int, default=5)
    ap.add_argument("--utt_duration_s", type=float, default=3.0)
    ap.add_argument("--n_noise_per_category", type=int, default=6)
    ap.add_argument("--noise_duration_s", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    speech_root = os.path.join(args.out_root, "clean_speech")
    for spk in range(args.n_speakers):
        spk_dir = os.path.join(speech_root, f"spk{spk:03d}")
        os.makedirs(spk_dir, exist_ok=True)
        for u in range(args.utts_per_speaker):
            audio = synth_speech_like(args.utt_duration_s, args.sr, rng)
            sf.write(os.path.join(spk_dir, f"utt{u:03d}.wav"), audio, args.sr, subtype="PCM_16")

    for category in ["impulsive", "rotor", "vehicle", "ambient", "sirens"]:
        cat_dir = os.path.join(args.out_root, "noise", category)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(args.n_noise_per_category):
            audio = synth_noise(category, args.noise_duration_s, args.sr, rng)
            sf.write(os.path.join(cat_dir, f"{category}_{i:03d}.wav"), audio, args.sr, subtype="PCM_16")

    print(f"Synthetic smoke data written under {args.out_root}/")
    print(f"  {args.n_speakers} speakers x {args.utts_per_speaker} utterances (clean_speech/)")
    print(f"  {args.n_noise_per_category} clips x 5 categories (noise/)")


if __name__ == "__main__":
    main()
