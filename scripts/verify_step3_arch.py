"""
Architecture equivalence test for Step 3a (UNet1D + StandardDDPM).

Strategy: rather than train and compare results (slow, stochastic),
we verify that the new module produces *bit-identical* outputs to the
original module for any given input — i.e. the refactor is purely
structural and the math is unchanged.

To do this without depending on the user's full Block-2 file, we embed
a verbatim copy of the original ``ResBlock1D`` / ``UNet1D`` and the
original ``StandardDDPM`` here. Both implementations are then:

1. Initialised with the same seed (so they produce the same random weights).
2. Fed the same input ``(x, t, c)``.
3. Their forward-pass outputs are compared.
4. DDPM noise schedules (alphas, alpha_bar, posterior_var) are also compared.

If the two implementations are mathematically equivalent, every check
should match to bit-precision (atol=0). Float-arithmetic order doesn't
matter for these because we're literally executing the same ops.

Usage:
    python scripts/verify_step3_arch.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force UTF-8 stdout/stderr (Windows-friendly).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet1d import UNet1D as NewUNet1D
from src.diffusion.ddpm import StandardDDPM as NewStandardDDPM


# ============================================================
# Verbatim copies of the original Block-2 1D-DDPM implementations
# ============================================================
class _OldSinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
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


class _OldResBlock1D(nn.Module):
    def __init__(self, ch, out_ch, t_dim, c_dim):
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


class _OldUNet1D(nn.Module):
    def __init__(self, img_ch=1, base=128, cond_dim=13, t_dim=128):
        super().__init__()
        self.t_mlp = nn.Sequential(
            _OldSinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.enc1_in = nn.Conv1d(img_ch, base, 3, padding=1)
        self.enc1_b1 = _OldResBlock1D(base, base, t_dim, t_dim)
        self.enc1_b2 = _OldResBlock1D(base, base, t_dim, t_dim)
        self.down1 = nn.Conv1d(base, base, 4, 2, 1)
        self.enc2_b1 = _OldResBlock1D(base, base * 2, t_dim, t_dim)
        self.enc2_b2 = _OldResBlock1D(base * 2, base * 2, t_dim, t_dim)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, 2, 1)
        self.mid_b1 = _OldResBlock1D(base * 2, base * 2, t_dim, t_dim)
        self.mid_b2 = _OldResBlock1D(base * 2, base * 2, t_dim, t_dim)
        self.up2 = nn.ConvTranspose1d(base * 2, base * 2, 4, 2, 1)
        self.dec2_b1 = _OldResBlock1D(base * 4, base * 2, t_dim, t_dim)
        self.dec2_b2 = _OldResBlock1D(base * 2, base, t_dim, t_dim)
        self.up1 = nn.ConvTranspose1d(base, base, 4, 2, 1)
        self.dec1_b1 = _OldResBlock1D(base * 2, base, t_dim, t_dim)
        self.dec1_b2 = _OldResBlock1D(base, base, t_dim, t_dim)
        self.out = nn.Conv1d(base, img_ch, 3, padding=1)

    def forward(self, x, t, c):
        t = self.t_mlp(t); c = self.c_mlp(c)
        h1 = self.enc1_in(x)
        h1 = self.enc1_b1(h1, t, c); h1 = self.enc1_b2(h1, t, c)
        h = self.down1(h1)
        h2 = self.enc2_b1(h, t, c); h2 = self.enc2_b2(h2, t, c)
        h = self.down2(h2)
        h = self.mid_b1(h, t, c); h = self.mid_b2(h, t, c)
        h = self.up2(h)
        if h.shape[-1] != h2.shape[-1]:
            h = F.interpolate(h, size=h2.shape[-1], mode="linear", align_corners=False)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2_b1(h, t, c); h = self.dec2_b2(h, t, c)
        h = self.up1(h)
        if h.shape[-1] != h1.shape[-1]:
            h = F.interpolate(h, size=h1.shape[-1], mode="linear", align_corners=False)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1_b1(h, t, c); h = self.dec1_b2(h, t, c)
        return self.out(h)


def _old_linear_beta_schedule(T, beta_start=1e-4, beta_end=1e-2):
    return torch.linspace(beta_start, beta_end, T)


class _OldStandardDDPM:
    """Verbatim copy of the original Block-2 1D-DDPM StandardDDPM."""

    def __init__(self, model, T=1000, device="cpu"):
        self.model = model
        self.T = T
        self.device = device
        betas_cpu = _old_linear_beta_schedule(T)
        self.betas = betas_cpu.to(device)
        self.alphas = (1.0 - betas_cpu).to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_var = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bar[t][:, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


# ============================================================
# Comparisons
# ============================================================
def check(name, ok, extra=""):
    flag = "✓" if ok else "✗"
    suffix = f"  {extra}" if extra else ""
    print(f"  {flag} {name}{suffix}")
    return ok


def main():
    all_ok = True
    device = "cpu"  # Deterministic CPU compare; same conclusion on GPU.
    torch.use_deterministic_algorithms(False)  # let cudnn deterministic stay default

    # ---- Identical weights from identical seed ----
    print("=" * 60)
    print("Step 1: Identical seed → identical initial weights")
    print("=" * 60)
    torch.manual_seed(123)
    new_model = NewUNet1D(img_ch=1, base=128, cond_dim=13, t_dim=128)
    torch.manual_seed(123)
    old_model = _OldUNet1D(img_ch=1, base=128, cond_dim=13, t_dim=128)

    new_sd = new_model.state_dict()
    old_sd = old_model.state_dict()
    ok = set(new_sd.keys()) == set(old_sd.keys())
    all_ok &= check("Parameter names match", ok, f"({len(new_sd)} params)")
    if not ok:
        missing = set(old_sd.keys()) - set(new_sd.keys())
        extra = set(new_sd.keys()) - set(old_sd.keys())
        print(f"    missing in new: {sorted(missing)[:5]}")
        print(f"    extra in new:   {sorted(extra)[:5]}")

    if ok:
        max_diff = 0.0
        for k in new_sd:
            d = (new_sd[k] - old_sd[k]).abs().max().item()
            if d > max_diff:
                max_diff = d
        all_ok &= check(
            "Initial weights numerically identical",
            max_diff == 0.0,
            f"max|Δ|={max_diff:.2e}",
        )

    # ---- Forward pass produces identical output ----
    print()
    print("=" * 60)
    print("Step 2: Same input → same forward-pass output")
    print("=" * 60)
    torch.manual_seed(0)
    x = torch.randn(2, 1, 384)
    t = torch.rand(2)
    c = torch.randn(2, 13)

    new_model.eval(); old_model.eval()
    with torch.no_grad():
        y_new = new_model(x, t, c)
        y_old = old_model(x, t, c)

    ok = y_new.shape == y_old.shape
    all_ok &= check(f"Output shapes match", ok, f"new={tuple(y_new.shape)}, old={tuple(y_old.shape)}")
    if ok:
        max_diff = (y_new - y_old).abs().max().item()
        all_ok &= check(
            "Output values bit-identical",
            max_diff == 0.0,
            f"max|Δ|={max_diff:.2e}",
        )

    # ---- DDPM noise schedule equivalence ----
    print()
    print("=" * 60)
    print("Step 3: DDPM noise-schedule equivalence")
    print("=" * 60)
    new_ddpm = NewStandardDDPM(new_model, T=1000, device=device)
    old_ddpm = _OldStandardDDPM(old_model, T=1000, device=device)
    for name in ("betas", "alphas", "alpha_bar", "alpha_bar_prev", "posterior_var"):
        a = getattr(new_ddpm, name)
        b = getattr(old_ddpm, name)
        d = (a - b).abs().max().item()
        all_ok &= check(f"{name}", d == 0.0, f"max|Δ|={d:.2e}")

    # ---- q_sample equivalence (with controlled noise) ----
    print()
    print("=" * 60)
    print("Step 4: q_sample equivalence")
    print("=" * 60)
    torch.manual_seed(7)
    x0 = torch.randn(4, 1, 384)
    t_idx = torch.tensor([0, 100, 500, 999], dtype=torch.long)
    fixed_noise = torch.randn_like(x0)
    a = new_ddpm.q_sample(x0, t_idx, noise=fixed_noise)
    b = old_ddpm.q_sample(x0, t_idx, noise=fixed_noise)
    d = (a - b).abs().max().item()
    all_ok &= check("q_sample(x0, t, fixed_noise)", d == 0.0, f"max|Δ|={d:.2e}")

    # ---- Verdict ----
    print()
    print("=" * 60)
    if all_ok:
        print("✓ NEW UNet1D + StandardDDPM IS BIT-IDENTICAL TO ORIGINAL")
        return 0
    print("✗ DIFFERENCES FOUND.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
