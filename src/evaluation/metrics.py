"""
Metric computation — mirrors Block 3's inline implementation EXACTLY.

In Block 3, metrics are computed inline inside ``train_eval_loop`` after
inverse-transforming through SOHScaler. This module exposes the same
computation as a reusable function, with identical formulas (including
the same handling of edge cases — zero std for Pearson, MAPE mask,
``+ 1e-8`` in R² denominator).
"""

import numpy as np


def compute_metrics(preds, targets):
    """Compute 7 metrics on already-inverse-transformed predictions/targets.

    Returns a dict identical in shape to Block 3's ``metrics`` dict.
    """
    preds_flat = np.asarray(preds).flatten()
    targets_flat = np.asarray(targets).flatten()

    mae = np.mean(np.abs(preds_flat - targets_flat))
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

    # MAPE: exclude near-zero targets.
    mask = np.abs(targets_flat) > 0.01
    mape = (np.mean(np.abs((targets_flat[mask] - preds_flat[mask]) / targets_flat[mask])) * 100
            if mask.sum() > 0 else 0)

    # R²
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Pearson
    if len(preds_flat) > 1 and np.std(preds_flat) > 1e-8:
        pearson = np.corrcoef(preds_flat, targets_flat)[0, 1]
        if np.isnan(pearson):
            pearson = 0
    else:
        pearson = 0

    max_ae = np.max(np.abs(preds_flat - targets_flat))
    med_ae = np.median(np.abs(preds_flat - targets_flat))

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2),
        'Pearson': float(pearson),
        'MaxAE': float(max_ae),
        'MedAE': float(med_ae),
    }
