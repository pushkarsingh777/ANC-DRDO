"""
Augmentations used when generating noisy-clean training pairs (FR-1.4).

- reverb: convolution with a synthetic exponential-decay impulse response
  (swap in real recorded IRs later, e.g. OpenAIR / MIT IR survey, without
  changing the call site — see apply_reverb's `ir` argument).
- clipping: hard-clip simulating saturated radio/mic hardware.
- speed_perturbation: resample-based tempo change (applied to speech only,
  so the "clean" target still matches the perturbed mixture).
- impulsive bursts: overlay short transient events (gunshots, explosions,
  hatch slams, etc.) at random timestamps, independent of the main noise bed.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .audio_utils import resample, fit_or_loop_to_length


def synth_exponential_ir(rt60_s: float, sr: int) -> np.ndarray:
    """Generate a synthetic room impulse response via exponentially-decaying noise."""
    n = int(rt60_s * sr)
    n = max(n, 8)
    t = np.arange(n) / sr
    decay = np.exp(-t * (6.9 / rt60_s))  # -60dB at t=rt60
    ir = np.random.default_rng().standard_normal(n).astype(np.float32) * decay
    ir /= (np.max(np.abs(ir)) + 1e-10)
    return ir.astype(np.float32)


def apply_reverb(audio: np.ndarray, rt60_s: float, sr: int, rng: np.random.Generator, ir: np.ndarray | None = None) -> np.ndarray:
    if ir is None:
        n = max(int(rt60_s * sr), 8)
        t = np.arange(n) / sr
        decay = np.exp(-t * (6.9 / rt60_s))
        ir = rng.standard_normal(n).astype(np.float32) * decay
        ir /= (np.max(np.abs(ir)) + 1e-10)
    wet = fftconvolve(audio, ir)[: len(audio)]
    # keep energy comparable to the dry signal
    dry_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    wet_rms = np.sqrt(np.mean(wet ** 2) + 1e-10)
    wet *= (dry_rms / (wet_rms + 1e-10))
    return wet.astype(np.float32)


def apply_clipping(audio: np.ndarray, clip_threshold: float) -> np.ndarray:
    """Hard-clip at `clip_threshold` * current peak, simulating saturated hardware."""
    peak = np.max(np.abs(audio)) + 1e-10
    limit = peak * clip_threshold
    return np.clip(audio, -limit, limit).astype(np.float32)


def apply_speed_perturbation(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """
    Change tempo/pitch together by resampling then restretching to original length
    is avoided here (that reintroduces length mismatch); instead we simply resample
    to `rate * sr` and treat that as the new sample rate, then resample back to `sr`.
    This changes formants/pitch slightly, which is a reasonable proxy for speaker/
    mic variability and is a standard cheap speed-perturbation trick.
    """
    if rate == 1.0:
        return audio
    intermediate_sr = int(sr * rate)
    resampled = resample(audio, sr, intermediate_sr)
    back = resample(resampled, intermediate_sr, sr)
    return back.astype(np.float32)


def overlay_impulsive_bursts(
    base: np.ndarray,
    burst_pool: list[np.ndarray],
    sr: int,
    rng: np.random.Generator,
    count_range: tuple[int, int] = (1, 3),
    gain_range: tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    """
    Overlay `count` short impulsive noise clips (gunshots/artillery/explosions)
    at random positions on top of `base` (speech+noise bed). Returns a new array;
    `base` is not modified in place.
    """
    if not burst_pool:
        return base
    out = base.copy()
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    for _ in range(count):
        burst = burst_pool[rng.integers(0, len(burst_pool))]
        if len(burst) == 0:
            continue
        if len(burst) >= len(out):
            burst = burst[: len(out)]
        max_start = max(len(out) - len(burst), 0)
        start = int(rng.integers(0, max_start + 1))
        gain = float(rng.uniform(*gain_range))
        out[start:start + len(burst)] += burst * gain
    return out
