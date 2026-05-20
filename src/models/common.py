"""
Shared neural-network building blocks used across all generative models.

These modules are copied verbatim from the original Block-2 implementations
so that pre-trained checkpoints saved with the original code load cleanly
into the refactored architecture (identical parameter names and shapes).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal time-step embedding (Vaswani-style)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock1D(nn.Module):
    """1-D residual block conditioned on time and external condition.

    Conditioning is injected by adding linearly-projected time/condition
    embeddings to the activation after the first conv. The skip path is
    a 1×1 conv when input/output channel counts differ, else identity.
    """

    def __init__(self, ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.c_proj = nn.Linear(c_dim, out_ch)
        self.skip = nn.Conv1d(ch, out_ch, 1) if ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, t, c):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t)[:, :, None] + self.c_proj(c)[:, :, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ResBlock2D(nn.Module):
    """2-D residual block conditioned on time and external condition."""

    def __init__(self, ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.c_proj = nn.Linear(c_dim, out_ch)
        self.skip = nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, t, c):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t)[:, :, None, None] + self.c_proj(c)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UpBlock2D(nn.Module):
    """Resize-then-convolve upsampling block used by the Proposed UNet2D.

    Bilinear ×2 upsample → two 3×3 convs (with GroupNorm + SiLU) → residual.
    This avoids the checkerboard artefacts that ``ConvTranspose2d`` is
    prone to, while preserving expressive power. Used in place of
    ``ConvTranspose2d`` in the up-sampling path of the Proposed method.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        # 1×1 conv skip when channel counts differ.
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        x_up = self.up(x)
        h = F.silu(self.norm1(self.conv1(x_up)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x_up))
