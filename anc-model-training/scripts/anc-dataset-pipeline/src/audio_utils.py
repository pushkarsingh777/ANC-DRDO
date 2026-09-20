"""
Core audio I/O and signal utilities used across the dataset pipeline.

Kept dependency-light: numpy + scipy + soundfile only, no librosa.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load a wav file as mono float32 in [-1, 1], resampled to target_sr."""
    audio, sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr, subtype=subtype)


def rms(audio: np.ndarray, eps: float = 1e-10) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + eps))


def fit_or_loop_to_length(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Make `audio` exactly `length` samples: loop-tile if short, random-crop if long."""
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(audio) < length:
        reps = int(np.ceil(length / len(audio)))
        audio = np.tile(audio, reps)
    if len(audio) > length:
        max_start = len(audio) - length
        src_rms = rms(audio)
        # If source has audio energy, ensure the crop is not purely silent trailing pad
        best_crop = audio[:length]
        for _ in range(5):
            start = int(rng.integers(0, max_start + 1))
            candidate = audio[start:start + length]
            if src_rms < 1e-4 or rms(candidate) > 1e-4:
                return candidate.astype(np.float32)
        return best_crop.astype(np.float32)
    return audio.astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale `noise` so that mixing with `speech` achieves the target SNR (in dB).
    Returns (adjusted_speech, scaled_noise, mixture) such that
    `mixture == adjusted_speech + scaled_noise` EXACTLY, including after any
    anti-clipping peak normalization. Callers must save `adjusted_speech` as
    the "clean" target, not the original `speech` array, or the mixture/clean/
    noise triplet will be inconsistent on disk.
    """
    speech_rms = rms(speech, eps)
    noise_rms = rms(noise, eps)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / (noise_rms + eps)
    scaled_noise = noise * scale
    mixture = speech + scaled_noise
    adjusted_speech = speech
    # peak-normalize mixture (and apply the SAME gain to every stem) if it would clip
    peak = np.max(np.abs(mixture)) + eps
    if peak > 0.99:
        gain = 0.99 / peak
        mixture = mixture * gain
        scaled_noise = scaled_noise * gain
        adjusted_speech = adjusted_speech * gain
    return adjusted_speech, scaled_noise, mixture


def measure_snr_db(speech: np.ndarray, noise: np.ndarray, eps: float = 1e-10) -> float:
    """Sanity-check helper: measure the realized SNR of a (speech, noise) pair."""
    return 20 * np.log10((rms(speech, eps) + eps) / (rms(noise, eps) + eps))
