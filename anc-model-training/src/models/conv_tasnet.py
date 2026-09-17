"""
Causal Conv-TasNet Architecture for Real-Time Speech Enhancement (FR-2, FR-3).

Paper Reference:
  Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking
  for Speech Separation", IEEE/ACM TASLP 2019.

Customized for Defence Adaptive Noise Cancellation:
- Strictly causal (left-padded dilated convolutions only)
- Edge-optimized lightweight parameter budget (< 5M params)
- Direct waveform input -> enhanced speech waveform output
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

from .causal_layers import CausalConvBlock, ChannelLayerNorm, CumulativeLayerNorm


class CausalTCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) separator consisting of R repeats
    of X dilated causal convolutional blocks.
    """
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 512,
        kernel_size: int = 3,
        n_blocks_per_repeat: int = 8,
        n_repeats: int = 3,
        norm_type: str = "ChLN",
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        total_blocks = n_repeats * n_blocks_per_repeat
        idx = 0
        for _ in range(n_repeats):
            for b in range(n_blocks_per_repeat):
                dilation = 2 ** b
                is_last = (idx == total_blocks - 1)
                self.blocks.append(
                    CausalConvBlock(
                        in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        norm_type=norm_type,
                        is_last=is_last,
                    )
                )
                idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T_frames)
        skip_connection_sum = torch.zeros_like(x)
        current = x
        for block in self.blocks:
            res, skip = block(current)
            if res is not None:
                current = res
            skip_connection_sum = skip_connection_sum + skip
        return skip_connection_sum


class CausalConvTasNet(nn.Module):
    """
    Complete Causal Conv-TasNet Speech Enhancement Model.
    
    Args:
        n_filters: N, number of filters in autoencoder (feature dimension)
        encoder_kernel_size: L, kernel size in encoder/decoder (samples, e.g. 32 = 2ms at 16kHz)
        encoder_stride: hop size in encoder/decoder (e.g. 16 = 1ms)
        tcn_channels: B, bottleneck channels in TCN
        tcn_hidden_channels: H, internal depthwise channels
        kernel_size: P, kernel size in TCN depthwise convs
        n_blocks_per_repeat: X, number of convolutional blocks per repeat (dilations 1...2^(X-1))
        n_repeats: R, number of repeats
        norm_type: "ChLN" (Channel-wise LayerNorm) or "cLN" (Cumulative LayerNorm)
        mask_activation: "relu" or "sigmoid"
    """
    def __init__(
        self,
        n_filters: int = 256,
        encoder_kernel_size: int = 32,
        encoder_stride: int = 16,
        tcn_channels: int = 128,
        tcn_hidden_channels: int = 512,
        kernel_size: int = 3,
        n_blocks_per_repeat: int = 8,
        n_repeats: int = 3,
        norm_type: str = "ChLN",
        mask_activation: str = "relu",
    ):
        super().__init__()
        self.n_filters = n_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_stride = encoder_stride
        
        # 1. 1D Waveform Encoder
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            bias=False,
        )
        
        # 2. Bottleneck Layer
        NormLayer = ChannelLayerNorm if norm_type in ("ChLN", "gLN") else CumulativeLayerNorm
        self.bottleneck_norm = NormLayer(n_filters)
        self.bottleneck_conv = nn.Conv1d(n_filters, tcn_channels, 1, bias=False)
        
        # 3. TCN Separator
        self.tcn = CausalTCN(
            in_channels=tcn_channels,
            hidden_channels=tcn_hidden_channels,
            kernel_size=kernel_size,
            n_blocks_per_repeat=n_blocks_per_repeat,
            n_repeats=n_repeats,
            norm_type=norm_type,
        )
        
        # 4. Mask Generation
        self.mask_conv = nn.Conv1d(tcn_channels, n_filters, 1, bias=True)
        if mask_activation == "relu":
            self.mask_act = nn.ReLU()
        elif mask_activation == "sigmoid":
            self.mask_act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown mask_activation: {mask_activation}")
            
        # 5. 1D Waveform Decoder
        self.decoder = nn.ConvTranspose1d(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            bias=False,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (Batch, 1, Time) or (Batch, Time) raw audio waveform tensor.
        Returns:
            enhanced_speech: (Batch, Time) enhanced speech waveform.
        """
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)  # (B, 1, T)
            
        orig_length = mixture.shape[-1]
        
        # 1. Strictly causal left-padding: prevents lookahead across frame boundary
        causal_pad = self.encoder_kernel_size - self.encoder_stride
        if causal_pad > 0:
            mixture = F.pad(mixture, (causal_pad, 0))
            
        # 2. Right-padding so total length aligns with encoder stride
        pad_samples = (self.encoder_stride - (mixture.shape[-1] - self.encoder_kernel_size) % self.encoder_stride) % self.encoder_stride
        if pad_samples > 0:
            mixture = F.pad(mixture, (0, pad_samples))
            
        # 3. Encode waveform to feature representation
        # w: (B, n_filters, T_frames)
        w = F.relu(self.encoder(mixture))
        
        # 4. Bottleneck
        feats = self.bottleneck_conv(self.bottleneck_norm(w))
        
        # 5. Separator TCN
        separated = self.tcn(feats)
        
        # 6. Estimate speech mask
        mask = self.mask_act(self.mask_conv(separated))
        
        # 7. Apply mask
        masked_w = w * mask
        
        # 8. Decode back to waveform
        enhanced = self.decoder(masked_w)
        
        # Match original length
        enhanced = enhanced[:, 0, :orig_length]
        return enhanced

    @classmethod
    def from_config(cls, cfg: dict) -> CausalConvTasNet:
        model_cfg = cfg.get("model", {})
        return cls(
            n_filters=model_cfg.get("n_filters", 256),
            encoder_kernel_size=model_cfg.get("encoder_kernel_size", 32),
            encoder_stride=model_cfg.get("encoder_stride", 16),
            tcn_channels=model_cfg.get("tcn_channels", 128),
            tcn_hidden_channels=model_cfg.get("tcn_hidden_channels", 512),
            kernel_size=model_cfg.get("kernel_size", 3),
            n_blocks_per_repeat=model_cfg.get("n_blocks_per_repeat", 8),
            n_repeats=model_cfg.get("n_repeats", 3),
            norm_type=model_cfg.get("norm_type", "ChLN"),
        )
