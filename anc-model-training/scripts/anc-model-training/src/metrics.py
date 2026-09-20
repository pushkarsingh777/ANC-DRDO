"""
Objective evaluation metrics for Speech Enhancement (FR-2.3, PRD section 3.2).

Computes:
- SNR Improvement (delta SNR in dB)
- STOI (Short-Time Objective Intelligibility: 0.0 to 1.0)
- PESQ (Perceptual Evaluation of Speech Quality: -0.5 to 4.5, ITU-T P.862)
- Real-Time Factor (RTF: processing time / audio duration)
"""
from __future__ import annotations

import time
import numpy as np
# pyrefly: ignore [missing-import]
import torch

try:
    # pyrefly: ignore [import-error, missing-import]
    from pystoi import stoi as pystoi_fn
    HAS_PYSTOI = True
except ImportError:
    HAS_PYSTOI = False

try:
    # pyrefly: ignore [import-error, missing-import]
    from pesq import pesq as pypesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False


def calculate_snr(clean: np.ndarray, noise: np.ndarray, eps: float = 1e-10) -> float:
    """Computes Signal-to-Noise Ratio in dB."""
    s_pwr = np.mean(clean ** 2) + eps
    n_pwr = np.mean(noise ** 2) + eps
    return float(10.0 * np.log10(s_pwr / n_pwr))


def calculate_snr_improvement(
    mixture: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
) -> float:
    """
    Computes SNR improvement: SNR_enhanced - SNR_mixture.
    """
    min_len = min(len(mixture), len(enhanced), len(clean))
    mix = mixture[:min_len]
    enh = enhanced[:min_len]
    ref = clean[:min_len]
    
    input_noise = mix - ref
    output_noise = enh - ref
    
    snr_in = calculate_snr(ref, input_noise)
    snr_out = calculate_snr(ref, output_noise)
    return snr_out - snr_in


def calculate_stoi(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """Computes Short-Time Objective Intelligibility (STOI)."""
    if not HAS_PYSTOI:
        return 0.0
    min_len = min(len(clean), len(enhanced))
    try:
        return float(pystoi_fn(clean[:min_len], enhanced[:min_len], sr, extended=False))
    except Exception:
        return 0.0


def calculate_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """Computes Wideband PESQ score (WB-PESQ, ITU-T P.862.2)."""
    if not HAS_PESQ:
        return 0.0
    min_len = min(len(clean), len(enhanced))
    try:
        # Requires 16kHz for wideband
        mode = "wb" if sr == 16000 else "nb"
        return float(pypesq_fn(sr, clean[:min_len], enhanced[:min_len], mode))
    except Exception:
        return 0.0


def profile_real_time_factor(
    model: torch.nn.Module,
    audio_duration_s: float = 2.0,
    sr: int = 16000,
    device: str = "cuda",
    n_runs: int = 10,
) -> tuple[float, float]:
    """
    Measures the Real-Time Factor (RTF) and processing latency for a given model.
    RTF < 1.0 indicates faster-than-real-time performance (target < 0.5).
    """
    model.eval()
    dummy_input = torch.randn(1, int(audio_duration_s * sr), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
                
    durations = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            durations.append(t1 - t0)
            
    avg_proc_time = float(np.mean(durations))
    rtf = avg_proc_time / audio_duration_s
    return rtf, avg_proc_time
