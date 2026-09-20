"""
Causal neural network layers for real-time streaming audio processing (FR-2, FR-3).

All layers here strictly enforce causality:
- CausalConv1d: Left-padded 1D convolution with zero future lookahead.
- ChannelLayerNorm: Per-frame normalization across channels (no temporal leakage).
- CausalConvBlock: Dilated depthwise-separable convolutional block with residual
  and skip paths.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """
    Applies Layer Normalization across the channel dimension at each time step.
    
    Shape:
        Input: (B, C, T)
        Output: (B, C, T)
    
    This is strictly causal because statistics are computed independently
    for each time step t across channels C, without looking across time T.
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean = torch.mean(x_f32, dim=1, keepdim=True)
        var = torch.var(x_f32, dim=1, unbiased=False, keepdim=True)
        x_norm = (x_f32 - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.to(orig_dtype)
        return x_norm * self.gamma + self.beta


class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization (cLN) from the original Conv-TasNet paper.
    Computes running mean and variance cumulatively from t=0 to t=k.
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        orig_dtype = x.dtype
        x_f32 = x.float()
        B, C, T = x_f32.shape
        cum_sum = torch.cumsum(x_f32, dim=2)
        cum_power_sum = torch.cumsum(x_f32 ** 2, dim=2)
        
        step = torch.arange(1, T + 1, device=x.device, dtype=torch.float32).view(1, 1, T)
        cum_mean = cum_sum / step
        cum_var = (cum_power_sum / step) - (cum_mean ** 2)
        cum_var = torch.clamp(cum_var, min=0.0)
        
        x_norm = (x_f32 - cum_mean) / torch.sqrt(cum_var + self.eps)
        x_norm = x_norm.to(orig_dtype)
        return x_norm * self.gamma + self.beta


class CausalConv1d(nn.Module):
    """
    1D Convolution with causal left-padding.
    Guarantees output at time t depends only on inputs at <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """
    Dilated depthwise separable 1D convolutional block.
    
    Structure:
      1x1 Conv (B -> H)
      PReLU
      ChannelLayerNorm
      Causal Depthwise Conv (kernel_size, dilation, groups=H)
      PReLU
      ChannelLayerNorm
      1x1 Conv (H -> B)
      Residual output + Skip output
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        norm_type: str = "ChLN",
        is_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.is_last = is_last
        
        NormLayer = ChannelLayerNorm if norm_type in ("ChLN", "gLN") else CumulativeLayerNorm
        
        # 1x1 Conv (expand)
        self.conv1x1_in = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.norm1 = NormLayer(hidden_channels)
        
        # Causal Depthwise Conv
        self.depthwise_conv = CausalConv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            bias=False,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = NormLayer(hidden_channels)
        
        # 1x1 Conv (project back to in_channels for residual and skip)
        # The final block in TCN does not need a residual projection
        self.conv1x1_out = nn.Conv1d(hidden_channels, in_channels, 1, bias=False) if not is_last else None
        self.conv1x1_skip = nn.Conv1d(hidden_channels, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        # x: (B, in_channels, T)
        out = self.norm1(self.prelu1(self.conv1x1_in(x)))
        out = self.norm2(self.prelu2(self.depthwise_conv(out)))
        
        res = self.conv1x1_out(out) + x if self.conv1x1_out is not None else None
        skip = self.conv1x1_skip(out)
        return res, skip
