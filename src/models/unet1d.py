"""
1-D conditional UNet for time-series diffusion models.

Verbatim port of the architecture in Block-2 1D-DDPM. The forward pass
is structurally identical (same conv kernels, same skip connections,
same up/down sample operations), so checkpoints trained with the
original code load directly via ``model.load_state_dict(...)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import ResBlock1D, SinusoidalEmbedding


class UNet1D(nn.Module):
    def __init__(
        self,
        img_ch: int = 1,
        base: int = 128,
        cond_dim: int = 13,
        t_dim: int = 128,
    ):
        super().__init__()
        self.t_mlp = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.enc1_in = nn.Conv1d(img_ch, base, 3, padding=1)
        self.enc1_b1 = ResBlock1D(base, base, t_dim, t_dim)
        self.enc1_b2 = ResBlock1D(base, base, t_dim, t_dim)
        self.down1 = nn.Conv1d(base, base, 4, 2, 1)

        self.enc2_b1 = ResBlock1D(base, base * 2, t_dim, t_dim)
        self.enc2_b2 = ResBlock1D(base * 2, base * 2, t_dim, t_dim)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, 2, 1)

        self.mid_b1 = ResBlock1D(base * 2, base * 2, t_dim, t_dim)
        self.mid_b2 = ResBlock1D(base * 2, base * 2, t_dim, t_dim)

        self.up2 = nn.ConvTranspose1d(base * 2, base * 2, 4, 2, 1)
        self.dec2_b1 = ResBlock1D(base * 4, base * 2, t_dim, t_dim)
        self.dec2_b2 = ResBlock1D(base * 2, base, t_dim, t_dim)

        self.up1 = nn.ConvTranspose1d(base, base, 4, 2, 1)
        self.dec1_b1 = ResBlock1D(base * 2, base, t_dim, t_dim)
        self.dec1_b2 = ResBlock1D(base, base, t_dim, t_dim)

        self.out = nn.Conv1d(base, img_ch, 3, padding=1)

    def forward(self, x, t, c):
        t = self.t_mlp(t)
        c = self.c_mlp(c)

        h1 = self.enc1_in(x)
        h1 = self.enc1_b1(h1, t, c)
        h1 = self.enc1_b2(h1, t, c)

        h = self.down1(h1)
        h2 = self.enc2_b1(h, t, c)
        h2 = self.enc2_b2(h2, t, c)
        h = self.down2(h2)

        h = self.mid_b1(h, t, c)
        h = self.mid_b2(h, t, c)

        h = self.up2(h)
        if h.shape[-1] != h2.shape[-1]:
            h = F.interpolate(h, size=h2.shape[-1], mode="linear", align_corners=False)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2_b1(h, t, c)
        h = self.dec2_b2(h, t, c)

        h = self.up1(h)
        if h.shape[-1] != h1.shape[-1]:
            h = F.interpolate(h, size=h1.shape[-1], mode="linear", align_corners=False)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1_b1(h, t, c)
        h = self.dec1_b2(h, t, c)

        return self.out(h)
