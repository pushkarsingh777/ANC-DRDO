"""
Loss functions for Waveform-Domain Speech Enhancement (FR-2.2).

Includes:
- SISNRLoss: Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR)
- MultiResolutionSTFTLoss: Spectral convergence and log magnitude STFT loss
- CombinedLoss: Configurable multi-objective loss
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR) loss.
    
    Given estimated speech s_hat and reference clean speech s:
        s_target = (<s_hat, s> / ||s||^2) * s
        e_noise  = s_hat - s_target
        SI-SNR   = 10 * log10( ||s_target||^2 / (||e_noise||^2 + eps) )
        
    Loss = - mean(SI-SNR)
    """
    def __init__(self, eps: float = 1e-8, zero_mean: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est: (B, T) or (B, 1, T) Estimated speech waveform
            ref: (B, T) or (B, 1, T) Ground truth clean speech waveform
        Returns:
            loss: Scalar tensor (-SI-SNR in dB)
        """
        if est.ndim == 3:
            est = est.squeeze(1)
        if ref.ndim == 3:
            ref = ref.squeeze(1)

        # Force float32 for loss computation stability (prevents FP16 overflow/underflow)
        est = est.float()
        ref = ref.float()

        # Match lengths if slight difference
        min_len = min(est.shape[-1], ref.shape[-1])
        est = est[:, :min_len]
        ref = ref[:, :min_len]

        if self.zero_mean:
            est = est - torch.mean(est, dim=-1, keepdim=True)
            ref = ref - torch.mean(ref, dim=-1, keepdim=True)

        # Inner product <est, ref>
        dot = torch.sum(est * ref, dim=-1, keepdim=True)  # (B, 1)
        ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + self.eps  # (B, 1)
        
        # Target projection
        s_target = (dot / ref_energy) * ref  # (B, T)
        
        # Noise residual
        e_noise = est - s_target  # (B, T)
        
        target_norm = torch.sum(s_target ** 2, dim=-1) + self.eps
        noise_norm = torch.sum(e_noise ** 2, dim=-1) + self.eps
        
        ratio = torch.clamp(target_norm / noise_norm, min=self.eps, max=1e8)
        si_snr = 10.0 * torch.log10(ratio)  # (B,)
        return -torch.mean(si_snr)


class SpectralConvergenceLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.eps)


class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(torch.log(x_mag + self.eps), torch.log(y_mag + self.eps))


class STFTLoss(nn.Module):
    """Single resolution STFT loss."""
    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 128,
        win_length: int = 512,
        window: str = "hann_window",
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if est.ndim == 3:
            est = est.squeeze(1)
        if ref.ndim == 3:
            ref = ref.squeeze(1)
            
        x_stft = torch.stft(
            est,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        y_stft = torch.stft(
            ref,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        sc = self.sc_loss(x_mag, y_mag)
        mag = self.mag_loss(x_mag, y_mag)
        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss across multiple window/hop sizes."""
    def __init__(
        self,
        fft_sizes: list[int] = [512, 1024, 2048],
        hop_sizes: list[int] = [128, 256, 512],
        win_lengths: list[int] = [512, 1024, 2048],
    ):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        total_sc = 0.0
        total_mag = 0.0
        for loss_fn in self.losses:
            sc, mag = loss_fn(est, ref)
            total_sc += sc
            total_mag += mag
        return (total_sc + total_mag) / len(self.losses)


class CombinedLoss(nn.Module):
    """
    Weighted combination of SI-SNR loss and Multi-resolution STFT loss.
    """
    def __init__(self, si_snr_weight: float = 1.0, stft_weight: float = 0.0):
        super().__init__()
        self.si_snr_weight = si_snr_weight
        self.stft_weight = stft_weight
        self.si_snr_loss = SISNRLoss()
        if stft_weight > 0.0:
            self.stft_loss = MultiResolutionSTFTLoss()
        else:
            self.stft_loss = None

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = 0.0
        metrics = {}
        
        if self.si_snr_weight > 0.0:
            l_si = self.si_snr_loss(est, ref)
            loss += self.si_snr_weight * l_si
            metrics["si_snr_loss"] = float(l_si.item())
            metrics["si_snr_db"] = float(-l_si.item())
            
        if self.stft_loss is not None and self.stft_weight > 0.0:
            l_stft = self.stft_loss(est, ref)
            loss += self.stft_weight * l_stft
            metrics["stft_loss"] = float(l_stft.item())
            
        metrics["total_loss"] = float(loss.item())
        return loss, metrics
