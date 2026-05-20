"""
PyTorch datasets for the various generative-model variants.

- ``CurveDataset1D``           — used by 1D-DDPM (no DE).
- ``CurveDataset2D_ResampledDE`` — used by Resampled-DE / DE-DDPM (resample → fixed DE).
- ``CurveDataset2D_AutoDE``    — used by the Proposed method (adaptive DE, no resample).

All classes are copied verbatim from the original Block-2 scripts to
preserve numerical behaviour.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.battery import ensure_1d_float
from src.delay_embedding import DelayEmbedding


# ============================================================
# Resampling helper
# ============================================================
def interp_to_len(y, L: int) -> np.ndarray:
    """Linear interpolation of ``y`` to length ``L``. Verbatim from Block 2."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2:
        return np.full(L, y[0] if len(y) == 1 else 0.0, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, num=len(y))
    x_new = np.linspace(0.0, 1.0, num=L)
    return np.interp(x_new, x_old, y).astype(np.float64)


# ============================================================
# CurveDataset1D — used by 1D-DDPM
# ============================================================
class CurveDataset1D(Dataset):
    """1-D conditional capacity-curve dataset."""

    def __init__(
        self,
        batts,
        seq_len: int,
        cond_num_mean,
        cond_num_std,
        global_min=None,
        global_max=None,
        tag: str = "train",
    ):
        self.batts = batts
        self.seq_len = int(seq_len)
        self.tag = tag

        self.cond_num_mean = np.asarray(cond_num_mean, dtype=np.float64)
        self.cond_num_std = np.asarray(cond_num_std, dtype=np.float64)

        self.curves_rs = [interp_to_len(b.capacity_curve, self.seq_len) for b in batts]

        if global_min is None or global_max is None:
            all_vals = np.concatenate(self.curves_rs).astype(np.float64)
            self.global_min = float(np.min(all_vals))
            self.global_max = float(np.max(all_vals))
        else:
            self.global_min, self.global_max = float(global_min), float(global_max)

        self.cond_dim = 13

        lengths = [len(b.capacity_curve) for b in batts] if batts else [0]
        print(
            f"[{tag}] 1D Dataset: n_samples={len(batts)}, seq_len={self.seq_len}, "
            f"orig_len=[{min(lengths)}, {max(lengths)}], cond_dim={self.cond_dim}"
        )

    def __len__(self):
        return len(self.batts)

    def __getitem__(self, idx):
        b = self.batts[idx]
        y = self.curves_rs[idx]

        gmin, gmax = self.global_min, self.global_max
        if abs(gmax - gmin) < 1e-12:
            y_norm = np.zeros_like(y, dtype=np.float32)
        else:
            y_norm = 2.0 * ((y - gmin) / (gmax - gmin)) - 1.0
            y_norm = y_norm.astype(np.float32)

        num = b.get_condition_numeric()
        num_n = (num - self.cond_num_mean) / self.cond_num_std
        cond = num_n.astype(np.float32)

        meta = {
            "battery_name": b.battery_name,
            "seq_len": self.seq_len,
            "orig_len": len(b.capacity_curve),
            "gmin": gmin,
            "gmax": gmax,
        }
        return (
            torch.from_numpy(y_norm[None, :]).float(),
            torch.from_numpy(cond).float(),
            meta,
        )

    @staticmethod
    def denorm_curve(y_norm, meta):
        return ((y_norm + 1.0) / 2.0) * (meta["gmax"] - meta["gmin"]) + meta["gmin"]


# ============================================================
# CurveDataset2D_ResampledDE — used by Resampled-DE / DE-DDPM
# ============================================================
class CurveDataset2D_ResampledDE(Dataset):
    """Capacity curves are resampled to a fixed ``seq_len``, then delay-embedded
    to a fixed ``img_size × img_size`` image (fixed-mode DE).

    Conditioning is the z-scored 13-D physics-feature vector. Normalisation
    to ``[-1, 1]`` uses training-split global min/max.
    """

    def __init__(
        self,
        batts,
        img_size: int,
        seq_len: int,
        cond_num_mean,
        cond_num_std,
        global_min=None,
        global_max=None,
        tag: str = "train",
    ):
        self.batts = batts
        self.img_size = int(img_size)
        self.seq_len = int(seq_len)
        self.tag = tag

        self.cond_num_mean = np.asarray(cond_num_mean)
        self.cond_num_std = np.asarray(cond_num_std)

        # Resample, then DE.
        self.images, self.metas, self.orig_lengths = [], [], []
        for b in batts:
            y_orig = ensure_1d_float(b.capacity_curve)
            orig_len = len(y_orig)
            self.orig_lengths.append(orig_len)
            y_resampled = interp_to_len(y_orig, self.seq_len)
            img, meta = DelayEmbedding.forward_fixed(y_resampled, self.img_size, self.seq_len)
            meta["battery_name"] = b.battery_name
            meta["orig_len"] = orig_len
            self.images.append(img)
            self.metas.append(meta)

        if global_min is None or global_max is None:
            all_vals = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts])
            self.global_min = float(np.min(all_vals))
            self.global_max = float(np.max(all_vals))
        else:
            self.global_min, self.global_max = float(global_min), float(global_max)

        self.images_norm = []
        for img in self.images:
            if abs(self.global_max - self.global_min) < 1e-12:
                img_norm = np.zeros_like(img, dtype=np.float32)
            else:
                img_norm = 2.0 * ((img - self.global_min) / (self.global_max - self.global_min)) - 1.0
            self.images_norm.append(img_norm.astype(np.float32))

        self.num_dim = 13
        self.cond_dim = self.num_dim

        print(
            f"[{tag}] ★ Resampled-DE: 样本={len(batts)}, "
            f"原始长度=[{min(self.orig_lengths)}, {max(self.orig_lengths)}] → SEQ_LEN={self.seq_len}, "
            f"DE={self.img_size}×{self.img_size}, cond_dim={self.cond_dim}"
        )

    def __len__(self):
        return len(self.batts)

    def __getitem__(self, idx):
        b = self.batts[idx]
        img_norm = self.images_norm[idx]
        meta = self.metas[idx].copy()
        meta["gmin"], meta["gmax"] = self.global_min, self.global_max

        num = b.get_condition_numeric()
        num_n = (num - self.cond_num_mean) / self.cond_num_std
        cond = num_n.astype(np.float32)

        return (
            torch.from_numpy(img_norm[None, :, :]).float(),
            torch.from_numpy(cond).float(),
            meta,
        )

    @staticmethod
    def denorm_image(img_norm, meta):
        gmin, gmax = meta["gmin"], meta["gmax"]
        return ((img_norm + 1.0) / 2.0) * (gmax - gmin) + gmin

    @staticmethod
    def image_to_curve(img_denorm, meta):
        """Inverse DE: returns a length-``seq_len`` curve."""
        return DelayEmbedding.inverse_fixed(img_denorm, meta)


# ============================================================
# CurveDataset2D_AutoDE — used by Proposed
# ============================================================
class CurveDataset2D_AutoDE(Dataset):
    """Adaptive DE: each curve is delay-embedded *without* resampling, then
    padded (or zoomed) into an ``img_size × img_size`` image.

    ``meta['orig_q']`` records the number of valid columns — consumed by
    ``build_mask_from_metas`` / ``enforce_padding_constraint`` to keep
    the model from learning the padded regions.
    """

    def __init__(
        self,
        batts,
        img_size: int,
        cond_num_mean,
        cond_num_std,
        global_min=None,
        global_max=None,
        tag: str = "train",
    ):
        self.batts = batts
        self.img_size = int(img_size)
        self.tag = tag

        self.cond_num_mean = np.asarray(cond_num_mean)
        self.cond_num_std = np.asarray(cond_num_std)

        self.images, self.metas = [], []
        for b in batts:
            y = ensure_1d_float(b.capacity_curve)
            img, meta = DelayEmbedding.forward_to_square(y, self.img_size)
            meta["battery_name"] = b.battery_name
            self.images.append(img)
            self.metas.append(meta)

        if global_min is None or global_max is None:
            all_vals = np.concatenate([ensure_1d_float(b.capacity_curve) for b in batts])
            self.global_min = float(np.min(all_vals))
            self.global_max = float(np.max(all_vals))
        else:
            self.global_min, self.global_max = float(global_min), float(global_max)

        self.images_norm = []
        for img in self.images:
            if abs(self.global_max - self.global_min) < 1e-12:
                img_norm = np.zeros_like(img, dtype=np.float32)
            else:
                img_norm = 2.0 * ((img - self.global_min) / (self.global_max - self.global_min)) - 1.0
            self.images_norm.append(img_norm.astype(np.float32))

        self.num_dim = 13
        self.cond_dim = self.num_dim

        lengths = [len(b.capacity_curve) for b in batts]
        print(
            f"[{tag}] Auto-DE: 样本={len(batts)}, "
            f"原始长度=[{min(lengths)}, {max(lengths)}], "
            f"DE={self.img_size}×{self.img_size}, cond_dim={self.cond_dim}"
        )

    def __len__(self):
        return len(self.batts)

    def __getitem__(self, idx):
        b = self.batts[idx]
        img_norm = self.images_norm[idx]
        meta = self.metas[idx].copy()
        meta["gmin"], meta["gmax"] = self.global_min, self.global_max

        num = b.get_condition_numeric()
        num_n = (num - self.cond_num_mean) / self.cond_num_std
        cond = num_n.astype(np.float32)

        return (
            torch.from_numpy(img_norm[None, :, :]).float(),
            torch.from_numpy(cond).float(),
            meta,
        )

    @staticmethod
    def denorm_image(img_norm, meta):
        gmin, gmax = meta["gmin"], meta["gmax"]
        return ((img_norm + 1.0) / 2.0) * (gmax - gmin) + gmin

    @staticmethod
    def image_to_curve(img_denorm, meta):
        """Inverse adaptive DE: returns a length-``orig_len`` curve."""
        return DelayEmbedding.inverse_from_square(img_denorm, meta)


# ============================================================
# CurveDataset1D_NoCond — used by Vanilla GAN / VAE (unconditional)
# ============================================================
class CurveDataset1D_NoCond(Dataset):
    """Unconditional 1-D capacity-curve dataset.

    Differences from ``CurveDataset1D``:
    - No conditioning vector — ``__getitem__`` returns ``(x, meta)``.
    - Normalisation uses ``... / (gmax - gmin + 1e-12)`` rather than the
      explicit guard against zero range; preserved verbatim from Block 2.
    """

    def __init__(
        self,
        batts,
        seq_len: int,
        global_min=None,
        global_max=None,
        tag: str = "train",
    ):
        self.batts = batts
        self.seq_len = int(seq_len)
        self.curves_rs = [interp_to_len(b.capacity_curve, self.seq_len) for b in batts]

        if global_min is None or global_max is None:
            all_vals = np.concatenate(self.curves_rs).astype(np.float64)
            self.global_min = float(np.min(all_vals))
            self.global_max = float(np.max(all_vals))
        else:
            self.global_min, self.global_max = float(global_min), float(global_max)

        print(f"[{tag}] 样本={len(batts)}, seq_len={self.seq_len}")

    def __len__(self):
        return len(self.batts)

    def __getitem__(self, idx):
        b = self.batts[idx]
        y = self.curves_rs[idx]
        gmin, gmax = self.global_min, self.global_max
        # NOTE: ``+ 1e-12`` in denominator — preserved verbatim from Block 2.
        y_norm = 2.0 * ((y - gmin) / (gmax - gmin + 1e-12)) - 1.0

        meta = {
            "battery_name": b.battery_name,
            "gmin": gmin,
            "gmax": gmax,
            "orig_len": len(b.capacity_curve),
            "seq_len": self.seq_len,
        }
        return torch.from_numpy(y_norm.astype(np.float32)), meta

    @staticmethod
    def denorm_curve(y_norm, meta):
        return ((y_norm + 1.0) / 2.0) * (meta["gmax"] - meta["gmin"]) + meta["gmin"]


def no_cond_collate(batch):
    """Collate for ``CurveDataset1D_NoCond``: stacks tensors, keeps metas."""
    xs = torch.stack([b[0] for b in batch])
    ms = [b[1] for b in batch]
    return xs, ms


# ============================================================
# Collate
# ============================================================
def collate_keep_meta(batch):
    """Stack tensors but keep per-sample metadata as a list of dicts."""
    xs = torch.stack([b[0] for b in batch], dim=0)
    cs = torch.stack([b[1] for b in batch], dim=0)
    ms = [b[2] for b in batch]
    return xs, cs, ms
