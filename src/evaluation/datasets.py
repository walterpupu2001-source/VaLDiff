"""
Downstream-prediction datasets.

All classes are **copied verbatim** from the original Block-3 script
provided by the author, including the legacy ``EarlyToLateDataset`` and
``TrendPredictionDataset`` kept for back-compat. The downstream tasks
in the paper use ``SOHCurveDataset``.

Key behaviour (do NOT change):
- Each input SOH curve is FIRST resampled to a fixed length ``fixed_len``
  (default 500) via linear interpolation.
- Then it is split at ``int(fixed_len * input_ratio)`` into (X, Y).
- If a fitted ``SOHScaler`` is passed, BOTH X and Y are min-max normalized
  to [0, 1]; train_eval_loop inverse-transforms predictions back to the
  raw SOH scale before computing metrics.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def ensure_1d_float(x):
    """Drop NaN/Inf and flatten — matches Block 3's util."""
    arr = np.array(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    return arr


def interp_to_len(y, L):
    """Linear interpolation to length L — matches Block 3's util."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2:
        return np.full(L, y[0] if len(y) == 1 else 0.0, dtype=np.float64)
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, L)
    return np.interp(x_new, x_old, y).astype(np.float64)


# ============================================================
# SOHCurveDataset — Block 3 verbatim
# ============================================================
class SOHCurveDataset(Dataset):
    """
    Early SOH trajectory prediction.

    Input:  the first input_ratio portion of an SOH trajectory.
    Target: the remaining future SOH trajectory.

    SOH = Q(k) / Q_nominal (Q_nominal = 2.0 Ah for XJTU).
    If a fitted SOHScaler is provided, both X and Y are Min-Max normalized
    to [0, 1] using train-only statistics (with buffer). Inverse transform
    is performed in train_eval_loop before computing metrics, so reported
    MAE/RMSE/MAPE/R^2 are on the original SOH scale.
    """
    def __init__(self, curves, global_min=None, global_max=None,
                 fixed_len=500, input_ratio=0.5, scaler=None):
        self.samples = []
        self.scaler = scaler
        for curve in curves:
            soh = ensure_1d_float(curve)
            if len(soh) < 50:
                continue
            soh_resampled = interp_to_len(soh, fixed_len)
            # Apply Min-Max normalization to [0,1] (train-only stats) if scaler is given.
            if self.scaler is not None and self.scaler.fitted:
                soh_resampled = self.scaler.transform(soh_resampled)
            split_idx = int(fixed_len * input_ratio)
            X = soh_resampled[:split_idx].astype(np.float32)
            Y = soh_resampled[split_idx:].astype(np.float32)
            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx][0]), torch.from_numpy(self.samples[idx][1])


# ============================================================
# Legacy aliases — Block 3 verbatim
# ============================================================
class EarlyToLateDataset(Dataset):
    """
    Legacy dataset kept for compatibility.
    It now operates on SOH trajectories rather than min-max normalised capacity curves.
    Optionally applies SOHScaler (Min-Max to [0,1]) for normalization.
    """
    def __init__(self, curves, global_min=None, global_max=None,
                 fixed_len=500, input_ratio=0.3, pred_horizon=0.2, scaler=None):
        self.samples = []
        self.scaler = scaler
        for curve in curves:
            soh = ensure_1d_float(curve)
            if len(soh) < 50:
                continue
            soh_resampled = interp_to_len(soh, fixed_len)
            if self.scaler is not None and self.scaler.fitted:
                soh_resampled = self.scaler.transform(soh_resampled)
            split_idx = int(fixed_len * input_ratio)
            target_start_idx = int(fixed_len * (1 - pred_horizon))
            X = soh_resampled[:split_idx].astype(np.float32)
            Y = soh_resampled[target_start_idx:].astype(np.float32)
            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx][0]), torch.from_numpy(self.samples[idx][1])


class TrendPredictionDataset(Dataset):
    """
    Legacy trend-prediction dataset kept for compatibility; input curves are SOH trajectories.
    Optionally applies SOHScaler (Min-Max to [0,1]) for normalization on the X side
    (trend slope/avg targets are derived from the normalized series).
    """
    def __init__(self, curves, global_min=None, global_max=None,
                 fixed_len=500, input_ratio=0.4, scaler=None):
        self.samples = []
        self.scaler = scaler
        for curve in curves:
            soh = ensure_1d_float(curve)
            if len(soh) < 50:
                continue
            soh_resampled = interp_to_len(soh, fixed_len)
            if self.scaler is not None and self.scaler.fitted:
                soh_resampled = self.scaler.transform(soh_resampled)
            split_idx = int(fixed_len * input_ratio)
            X = soh_resampled[:split_idx].astype(np.float32)
            future = soh_resampled[split_idx:]
            avg_val = np.mean(future)
            slope = np.polyfit(np.linspace(0, 1, len(future)), future, 1)[0]
            Y = np.array([avg_val, slope], dtype=np.float32)
            self.samples.append((X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx][0]), torch.from_numpy(self.samples[idx][1])


# Backward-compatible alias used in the original code.
FullCurveDataset = SOHCurveDataset
