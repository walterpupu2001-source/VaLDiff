"""
2-D conditional UNet for delay-embedding diffusion models.

This single class subsumes the two near-identical UNet2D definitions in
the original code:

- ``use_upblock=False``: matches Block-2 Resampled-DE (uses
  ``nn.ConvTranspose2d`` for the up-sampling layers).
- ``use_upblock=True``: matches Block-2 Proposed (uses ``UpBlock2D``,
  i.e. bilinear-resize + conv, to avoid checkerboard artefacts).

The two modes produce DIFFERENT ``state_dict`` keys (because the up
layers have different internal parameters), so a Resampled-DE
checkpoint only loads cleanly with ``use_upblock=False`` and a Proposed
checkpoint only loads with ``use_upblock=True``. Within each mode, the
architecture is bit-identical to the corresponding Block-2 original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import ResBlock2D, SinusoidalEmbedding, UpBlock2D


class UNet2D(nn.Module):
    def __init__(
        self,
        img_ch: int = 1,
        base: int = 128,
        cond_dim: int = 13,
        t_dim: int = 128,
        use_upblock: bool = False,
    ):
        super().__init__()
        self.img_ch = img_ch
        self.use_upblock = use_upblock

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

        self.enc1_in = nn.Conv2d(img_ch, base, 3, padding=1)
        self.enc1_b1 = ResBlock2D(base, base, t_dim, t_dim)
        self.enc1_b2 = ResBlock2D(base, base, t_dim, t_dim)
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)

        self.enc2_b1 = ResBlock2D(base, base * 2, t_dim, t_dim)
        self.enc2_b2 = ResBlock2D(base * 2, base * 2, t_dim, t_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)

        self.mid_b1 = ResBlock2D(base * 2, base * 2, t_dim, t_dim)
        self.mid_b2 = ResBlock2D(base * 2, base * 2, t_dim, t_dim)

        # Up-sampling path: branches on use_upblock.
        if use_upblock:
            self.up2 = UpBlock2D(base * 2, base * 2)
        else:
            self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
        self.dec2_b1 = ResBlock2D(base * 4, base * 2, t_dim, t_dim)
        self.dec2_b2 = ResBlock2D(base * 2, base, t_dim, t_dim)

        if use_upblock:
            self.up1 = UpBlock2D(base, base)
        else:
            self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.dec1_b1 = ResBlock2D(base * 2, base, t_dim, t_dim)
        self.dec1_b2 = ResBlock2D(base, base, t_dim, t_dim)

        self.out = nn.Conv2d(base, img_ch, 3, padding=1)

    def forward(self, x, t, c):
        t_emb = self.t_mlp(t)
        c_emb = self.c_mlp(c)

        h1 = self.enc1_in(x)
        h1 = self.enc1_b1(h1, t_emb, c_emb)
        h1 = self.enc1_b2(h1, t_emb, c_emb)

        h = self.down1(h1)
        h2 = self.enc2_b1(h, t_emb, c_emb)
        h2 = self.enc2_b2(h2, t_emb, c_emb)

        h = self.down2(h2)
        h = self.mid_b1(h, t_emb, c_emb)
        h = self.mid_b2(h, t_emb, c_emb)

        h = self.up2(h)
        if h.shape[-2:] != h2.shape[-2:]:
            h = F.interpolate(h, size=h2.shape[-2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2_b1(h, t_emb, c_emb)
        h = self.dec2_b2(h, t_emb, c_emb)

        h = self.up1(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = F.interpolate(h, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1_b1(h, t_emb, c_emb)
        h = self.dec1_b2(h, t_emb, c_emb)

        return self.out(h)
