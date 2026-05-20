"""
Generative-model-based augmentation — Block 3 verbatim.

Each function:
- Loads a pretrained checkpoint
- Generates synthetic CAPACITY curves (Ah scale) using the same procedure
  as Block 3's ``generate_*`` helpers
- Returns a flat list of synthetic capacity curves at their original lengths

Important differences from any earlier (broken) version of this module:

1. ALL 2-D DDPM sampling passes ``metas=...`` to the sampler, which
   ALWAYS applies the padding constraint (verbatim with Block 3's
   ``DDPMSampler.sample_2d``). This is true for both Proposed
   (use_upblock=True) AND Resampled-DE (use_upblock=False).

2. Output curves are in CAPACITY (Ah) — the SAME scale the generators
   were trained to reconstruct. Conversion to SOH (divide by Q_nominal)
   happens later in evaluate_downstream.py, AFTER concatenation with
   the real SOH training curves, matching Block 3's data flow.
"""

from typing import List

import numpy as np
import torch

from src.data.battery import ensure_1d_float
from src.delay_embedding import DelayEmbedding
from src.diffusion.ddpm import StandardDDPM
from src.diffusion.proposed_constraints import enforce_padding_constraint
from src.models.gan import GeneratorGAN
from src.models.unet1d import UNet1D
from src.models.unet2d import UNet2D
from src.models.vae import VAE


# Block 3 defaults
BASE_CH = 128
T_STEPS = 1000


# ============================================================
# Internal helper: interp_to_len (Block 3 util)
# ============================================================
def _interp_to_len(y, L):
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2:
        return np.full(L, y[0] if len(y) == 1 else 0.0, dtype=np.float64)
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, L)
    return np.interp(x_new, x_old, y).astype(np.float64)


# ============================================================
# Internal helper: ResampledDelayEmbedding fixed forward
# (Block 3 has this as a separate class; we expose it through
#  src.delay_embedding.DelayEmbedding.forward_fixed already.)
# ============================================================


# ============================================================
# Internal helper: 2D padding-aware sample using StandardDDPM
#
# This matches Block 3's DDPMSampler.sample_2d byte-for-byte:
# - Applies enforce_padding_constraint to initial noise
# - Applies enforce_padding_constraint after EVERY reverse step
# ============================================================
@torch.no_grad()
def _sample_2d_with_padding(ddpm: StandardDDPM, cond, img_size, batch_size, metas):
    """Block 3's DDPMSampler.sample_2d, expressed via StandardDDPM primitives."""
    ddpm.model.eval()
    x = torch.randn(batch_size, 1, img_size, img_size, device=ddpm.device)
    x = enforce_padding_constraint(x, metas, img_size)
    for i in reversed(range(ddpm.T)):
        t = torch.full((batch_size,), i, device=ddpm.device, dtype=torch.long)
        x = ddpm.p_sample(x, t, i, cond)
        x = enforce_padding_constraint(x, metas, img_size)
    return x


# ============================================================
# Generate DE-DDPM curves (Block 3 generate_de_ddpm)
# ============================================================
def generate_de_ddpm(model_path: str, batts, cond_num_mean, cond_num_std,
                     use_upblock: bool, num_aug_per_battery: int,
                     img_size: int = 32, device: str = "cuda") -> List[np.ndarray]:
    """
    Generate synthetic CAPACITY (Ah) curves using the adaptive-DE DDPM.

    Used by:
      - Proposed (use_upblock=True)

    Block 3 generate_de_ddpm verbatim. Returns capacity curves at each
    battery's original cycle length.
    """
    print(f"  Generating from {model_path} (UpBlock={use_upblock})...")
    model = UNet2D(base=BASE_CH, cond_dim=13, t_dim=128, use_upblock=use_upblock).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ddpm = StandardDDPM(model, T=T_STEPS, device=device)

    gmin = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).min()
    gmax = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).max()

    syn: List[np.ndarray] = []
    for b in batts:
        curve_orig = ensure_1d_float(b.capacity_curve)
        _, meta = DelayEmbedding.forward_to_square(curve_orig, img_size)
        cond = torch.from_numpy(
            ((b.get_condition_numeric() - cond_num_mean) / cond_num_std).astype(np.float32)
        ).unsqueeze(0).repeat(num_aug_per_battery, 1).to(device)
        metas = [meta] * num_aug_per_battery
        samples = _sample_2d_with_padding(ddpm, cond, img_size, num_aug_per_battery, metas).cpu().numpy()
        for k in range(num_aug_per_battery):
            syn.append(
                DelayEmbedding.inverse_from_square(
                    ((samples[k, 0] + 1) / 2) * (gmax - gmin) + gmin, meta
                )
            )
    return syn


# ============================================================
# Generate Resampled-DE curves (Block 3 generate_resampled_de)
# Used by DE-DDPM in evaluate_downstream (note: ALWAYS uses padding constraint).
# ============================================================
def generate_resampled_de(model_path: str, batts, cond_num_mean, cond_num_std,
                          num_aug_per_battery: int,
                          img_size: int = 32, seq_len_resampled: int = 500,
                          device: str = "cuda") -> List[np.ndarray]:
    """
    Generate synthetic CAPACITY curves using the fixed-DE DDPM, then
    interpolate each back to its battery's original length.

    Block 3 generate_resampled_de verbatim — including the (perhaps
    surprising) fact that the padding constraint IS applied during
    sampling even though Resampled-DE was trained without it.
    """
    print(f"  Generating Resampled-DE from {model_path}...")
    model = UNet2D(base=BASE_CH, cond_dim=13, t_dim=128, use_upblock=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ddpm = StandardDDPM(model, T=T_STEPS, device=device)

    gmin = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).min()
    gmax = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).max()

    syn: List[np.ndarray] = []
    for b in batts:
        curve_orig = ensure_1d_float(b.capacity_curve)
        orig_len = len(curve_orig)
        _, meta = DelayEmbedding.forward_fixed(
            _interp_to_len(curve_orig, seq_len_resampled), img_size, seq_len_resampled
        )
        meta["orig_len"] = orig_len
        cond = torch.from_numpy(
            ((b.get_condition_numeric() - cond_num_mean) / cond_num_std).astype(np.float32)
        ).unsqueeze(0).repeat(num_aug_per_battery, 1).to(device)
        metas = [meta] * num_aug_per_battery
        samples = _sample_2d_with_padding(ddpm, cond, img_size, num_aug_per_battery, metas).cpu().numpy()
        for k in range(num_aug_per_battery):
            curve_resampled = DelayEmbedding.inverse_fixed(
                ((samples[k, 0] + 1) / 2) * (gmax - gmin) + gmin, meta
            )
            syn.append(_interp_to_len(curve_resampled, orig_len))
    return syn


# ============================================================
# Generate 1D-DDPM curves (Block 3 generate_ddpm_1d)
# ============================================================
def generate_ddpm_1d(model_path: str, batts, cond_num_mean, cond_num_std,
                     num_aug_per_battery: int,
                     seq_len: int = 384, device: str = "cuda") -> List[np.ndarray]:
    """
    Generate synthetic CAPACITY curves using the conditional 1D-DDPM.
    Block 3 generate_ddpm_1d verbatim.
    """
    print(f"  Generating 1D-DDPM from {model_path}...")
    model = UNet1D(base=BASE_CH, cond_dim=13, t_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ddpm = StandardDDPM(model, T=T_STEPS, device=device)

    gmin = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).min()
    gmax = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts]).max()

    syn: List[np.ndarray] = []
    for b in batts:
        cond = torch.from_numpy(
            ((b.get_condition_numeric() - cond_num_mean) / cond_num_std).astype(np.float32)
        ).unsqueeze(0).repeat(num_aug_per_battery, 1).to(device)
        samples = ddpm.sample(cond, shape=(1, seq_len)).cpu().numpy()
        for k in range(num_aug_per_battery):
            curve = _interp_to_len(
                ((samples[k, 0] + 1) / 2) * (gmax - gmin) + gmin, len(b.capacity_curve)
            )
            syn.append(curve)
    return syn


# ============================================================
# Generate GAN curves (Block 3 generate_gan)
# ============================================================
def generate_gan(model_path: str, num_samples: int, train_curves: List[np.ndarray],
                 seq_len: int = 384, device: str = "cuda") -> List[np.ndarray]:
    """
    Generate ``num_samples`` synthetic CAPACITY curves from the trained GAN.
    Block 3 generate_gan verbatim.

    ``train_curves`` must be the list of REAL capacity (Ah) curves for the
    training batteries — used to (a) compute the global min/max for the
    [-1,1] -> Ah de-normalization, and (b) determine the per-output length
    via cyclic indexing into ``train_curves``.
    """
    print(f"  Generating GAN from {model_path}...")
    G = GeneratorGAN(latent_dim=128, seq_len=seq_len, ch=64).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    G.load_state_dict(ckpt["generator"])
    G.eval()
    gmin, gmax = np.concatenate(train_curves).min(), np.concatenate(train_curves).max()
    with torch.no_grad():
        fakes = G(torch.randn(num_samples, 128, device=device)).cpu().numpy()
    return [
        _interp_to_len(((f + 1) / 2) * (gmax - gmin) + gmin, len(train_curves[i % len(train_curves)]))
        for i, f in enumerate(fakes)
    ]


# ============================================================
# Generate VAE curves (Block 3 generate_vae)
# ============================================================
def generate_vae(model_path: str, num_samples: int, train_curves: List[np.ndarray],
                 seq_len: int = 384, device: str = "cuda") -> List[np.ndarray]:
    """
    Generate ``num_samples`` synthetic CAPACITY curves from the trained VAE.
    Block 3 generate_vae verbatim.
    """
    print(f"  Generating VAE from {model_path}...")
    model = VAE(seq_len=seq_len, latent_dim=32, ch=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    gmin, gmax = np.concatenate(train_curves).min(), np.concatenate(train_curves).max()
    with torch.no_grad():
        samples = model.sample(num_samples, device).cpu().numpy()
    return [
        _interp_to_len(((s + 1) / 2) * (gmax - gmin) + gmin, len(train_curves[i % len(train_curves)]))
        for i, s in enumerate(samples)
    ]
