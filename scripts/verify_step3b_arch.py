"""
Architecture equivalence test for Step 3b (UNet2D + Proposed extras).

Verifies, by embedding the original Block-2 Resampled-DE / Proposed
UNet2D code in this file, that the refactored ``UNet2D`` (with
``use_upblock`` flag) is bit-identical to both original variants when
initialised with the same seed and given the same input.

Also verifies:
- ``build_mask_from_metas`` produces the expected mask.
- ``enforce_padding_constraint`` matches the original implementation.
- StandardDDPM masked-loss path matches the original Proposed train-step.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion.ddpm import StandardDDPM as NewStandardDDPM
from src.diffusion.proposed_constraints import (
    build_mask_from_metas,
    enforce_padding_constraint,
)
from src.models.unet2d import UNet2D as NewUNet2D


# ============================================================
# Verbatim copies of the original Block-2 implementations
# ============================================================
class _OldSinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, t):
        device = t.device; half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _OldResBlock2D(nn.Module):
    def __init__(self, ch, out_ch, t_dim, c_dim):
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


class _OldUpBlock2D(nn.Module):
    """Verbatim from Proposed Block-2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        x_up = self.up(x)
        h = F.silu(self.norm1(self.conv1(x_up)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x_up))


def _make_old_unet2d(use_upblock: bool, img_ch=1, base=128, cond_dim=13, t_dim=128):
    """Construct an *original-style* UNet2D for the requested mode.

    Replicates the Block-2 Resampled-DE class when ``use_upblock=False``,
    and the Block-2 Proposed class when ``use_upblock=True``.
    """
    class _OldUNet2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_ch = img_ch
            self.t_mlp = nn.Sequential(
                _OldSinusoidalEmbedding(t_dim),
                nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
                nn.Linear(t_dim * 4, t_dim),
            )
            self.c_mlp = nn.Sequential(
                nn.Linear(cond_dim, t_dim), nn.SiLU(),
                nn.Linear(t_dim, t_dim),
            )
            self.enc1_in = nn.Conv2d(img_ch, base, 3, padding=1)
            self.enc1_b1 = _OldResBlock2D(base, base, t_dim, t_dim)
            self.enc1_b2 = _OldResBlock2D(base, base, t_dim, t_dim)
            self.down1 = nn.Conv2d(base, base, 4, 2, 1)
            self.enc2_b1 = _OldResBlock2D(base, base * 2, t_dim, t_dim)
            self.enc2_b2 = _OldResBlock2D(base * 2, base * 2, t_dim, t_dim)
            self.down2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
            self.mid_b1 = _OldResBlock2D(base * 2, base * 2, t_dim, t_dim)
            self.mid_b2 = _OldResBlock2D(base * 2, base * 2, t_dim, t_dim)

            # CRITICAL: preserve the original creation order — up2, dec2_b1,
            # dec2_b2, then up1, dec1_b1, dec1_b2 — so the RNG stream that
            # produces initial weights matches the new UNet2D bit-for-bit.
            if use_upblock:
                self.up2 = _OldUpBlock2D(base * 2, base * 2)
            else:
                self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
            self.dec2_b1 = _OldResBlock2D(base * 4, base * 2, t_dim, t_dim)
            self.dec2_b2 = _OldResBlock2D(base * 2, base, t_dim, t_dim)

            if use_upblock:
                self.up1 = _OldUpBlock2D(base, base)
            else:
                self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
            self.dec1_b1 = _OldResBlock2D(base * 2, base, t_dim, t_dim)
            self.dec1_b2 = _OldResBlock2D(base, base, t_dim, t_dim)
            self.out = nn.Conv2d(base, img_ch, 3, padding=1)

        def forward(self, x, t, c):
            t_emb = self.t_mlp(t); c_emb = self.c_mlp(c)
            h1 = self.enc1_in(x)
            h1 = self.enc1_b1(h1, t_emb, c_emb); h1 = self.enc1_b2(h1, t_emb, c_emb)
            h = self.down1(h1)
            h2 = self.enc2_b1(h, t_emb, c_emb); h2 = self.enc2_b2(h2, t_emb, c_emb)
            h = self.down2(h2)
            h = self.mid_b1(h, t_emb, c_emb); h = self.mid_b2(h, t_emb, c_emb)
            h = self.up2(h)
            if h.shape[-2:] != h2.shape[-2:]:
                h = F.interpolate(h, size=h2.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, h2], dim=1)
            h = self.dec2_b1(h, t_emb, c_emb); h = self.dec2_b2(h, t_emb, c_emb)
            h = self.up1(h)
            if h.shape[-2:] != h1.shape[-2:]:
                h = F.interpolate(h, size=h1.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, h1], dim=1)
            h = self.dec1_b1(h, t_emb, c_emb); h = self.dec1_b2(h, t_emb, c_emb)
            return self.out(h)

    return _OldUNet2D()


# ---- Old Proposed train_step (verbatim) ----
def _old_build_mask(metas, img_size, device):
    masks = []
    for meta in metas:
        q = int(meta["orig_q"])
        q = max(1, min(q, img_size))
        m = torch.zeros(1, img_size, img_size, device=device, dtype=torch.float32)
        m[:, :, :q] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0)


@torch.no_grad()
def _old_enforce_padding(x, metas, img_size):
    B = x.shape[0]
    for b in range(B):
        q = int(metas[b]["orig_q"])
        q = max(1, min(q, img_size))
        if q < img_size:
            x[b, :, :, q:] = x[b, :, :, q-1:q]
    return x


# ============================================================
# Comparisons
# ============================================================
def check(name, ok, extra=""):
    flag = "✓" if ok else "✗"
    suffix = f"  {extra}" if extra else ""
    print(f"  {flag} {name}{suffix}")
    return ok


def compare_modules(new_model, old_model, mode_label):
    """Common test pattern: same seed → same weights → same forward."""
    print(f"\n{'-'*60}")
    print(f"UNet2D mode: {mode_label}")
    print(f"{'-'*60}")
    all_ok = True

    new_sd = new_model.state_dict()
    old_sd = old_model.state_dict()
    ok = set(new_sd.keys()) == set(old_sd.keys())
    all_ok &= check("Parameter names match", ok, f"({len(new_sd)} params)")
    if not ok:
        missing = set(old_sd.keys()) - set(new_sd.keys())
        extra = set(new_sd.keys()) - set(old_sd.keys())
        if missing: print(f"    missing in new (first 5): {sorted(missing)[:5]}")
        if extra:   print(f"    extra in new (first 5):   {sorted(extra)[:5]}")
        return all_ok

    max_diff = max((new_sd[k] - old_sd[k]).abs().max().item() for k in new_sd)
    all_ok &= check("Initial weights identical", max_diff == 0.0, f"max|Δ|={max_diff:.2e}")

    # Forward
    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32)
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
        all_ok &= check("Output values bit-identical", max_diff == 0.0, f"max|Δ|={max_diff:.2e}")
    return all_ok


def main():
    all_ok = True

    # ---- UNet2D with use_upblock=False (Resampled-DE original) ----
    torch.manual_seed(42)
    new_resampled = NewUNet2D(img_ch=1, base=128, cond_dim=13, t_dim=128, use_upblock=False)
    torch.manual_seed(42)
    old_resampled = _make_old_unet2d(use_upblock=False)
    all_ok &= compare_modules(new_resampled, old_resampled, "Resampled-DE (use_upblock=False)")

    # ---- UNet2D with use_upblock=True (Proposed original) ----
    torch.manual_seed(42)
    new_proposed = NewUNet2D(img_ch=1, base=128, cond_dim=13, t_dim=128, use_upblock=True)
    torch.manual_seed(42)
    old_proposed = _make_old_unet2d(use_upblock=True)
    all_ok &= compare_modules(new_proposed, old_proposed, "Proposed (use_upblock=True)")

    # ---- Padding-constraint helpers ----
    print(f"\n{'-'*60}")
    print("Padding-constraint helpers (Proposed)")
    print(f"{'-'*60}")
    img_size = 32
    metas = [{"orig_q": 20}, {"orig_q": 32}, {"orig_q": 5}]

    m_new = build_mask_from_metas(metas, img_size, "cpu")
    m_old = _old_build_mask(metas, img_size, "cpu")
    all_ok &= check("build_mask_from_metas", torch.equal(m_new, m_old))

    torch.manual_seed(0)
    x = torch.randn(3, 1, img_size, img_size)
    a = enforce_padding_constraint(x.clone(), metas, img_size)
    b = _old_enforce_padding(x.clone(), metas, img_size)
    all_ok &= check("enforce_padding_constraint", torch.equal(a, b))

    # ---- DDPM masked-loss equivalence ----
    print(f"\n{'-'*60}")
    print("Masked-loss DDPM equivalence (vs original Proposed train_step)")
    print(f"{'-'*60}")
    # Tiny model so we can compute quickly. Same model for both sides.
    torch.manual_seed(7)
    tiny = NewUNet2D(img_ch=1, base=8, cond_dim=4, t_dim=16, use_upblock=True).eval()
    ddpm = NewStandardDDPM(tiny, T=10, device="cpu")

    torch.manual_seed(0)
    x0 = torch.randn(2, 1, img_size, img_size)
    cond = torch.randn(2, 4)
    metas2 = [{"orig_q": 10}, {"orig_q": 28}]

    # Replicate the original Proposed train_step: enforce padding on x_t,
    # then masked MSE over the valid columns.
    def old_train_step(seed):
        torch.manual_seed(seed)
        t = torch.randint(0, ddpm.T, (x0.shape[0],), device=ddpm.device).long()
        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t, noise)
        x_t = _old_enforce_padding(x_t, metas2, img_size)
        t_in = t.float() / (ddpm.T - 1)
        eps_pred = ddpm.model(x_t, t_in, cond)
        mask = _old_build_mask(metas2, img_size, ddpm.device)
        return ((eps_pred - noise) ** 2 * mask).sum() / (mask.sum() + 1e-8)

    def new_train_step(seed):
        torch.manual_seed(seed)
        mask = build_mask_from_metas(metas2, img_size, ddpm.device)
        pad_fn = lambda x: enforce_padding_constraint(x, metas2, img_size)
        return ddpm.train_step(x0, cond, mask=mask, padding_fn=pad_fn)

    l_old = old_train_step(123).item()
    l_new = new_train_step(123).item()
    all_ok &= check("Masked-loss + padding train_step", l_old == l_new,
                    f"old={l_old:.10f}  new={l_new:.10f}  Δ={abs(l_old - l_new):.2e}")

    # ---- Verdict ----
    print()
    print("=" * 60)
    if all_ok:
        print("✓ STEP-3b CODE IS BIT-IDENTICAL TO ORIGINAL")
        return 0
    print("✗ DIFFERENCES FOUND.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
