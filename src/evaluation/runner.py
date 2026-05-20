"""
Downstream training + evaluation loop — Block 3 verbatim.

The original Block 3 helper trains a predictor with AdamW + MSE loss,
early-stops on validation MSE, restores the best state, inverse-transforms
predictions/targets through the SOHScaler before computing 7 metrics,
and optionally returns full prediction details for visualisation.

DO NOT alter:
- Optimizer (AdamW)
- Loss (MSE)
- Patience (15)
- Default device handling
- Order of operations in test inference
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Block 3 defaults
DOWNSTREAM_LR = 1e-3
DOWNSTREAM_EPOCHS = 100


def train_eval_loop(model, train_loader, val_loader, test_loader,
                    return_details=False, scaler=None,
                    epochs=DOWNSTREAM_EPOCHS, lr=DOWNSTREAM_LR,
                    device=None):
    """
    Block 3 verbatim. ``scaler`` should be a fitted SOHScaler (or None);
    if provided, predictions and targets are inverse-transformed back to
    raw SOH BEFORE computing metrics, so MAE/RMSE/MAPE/R^2 are on the
    original SOH scale.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_loss, best_state = float('inf'), None
    patience, counter = 15, 0

    for ep in range(epochs):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            loss = F.mse_loss(model(X), Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                val_loss += F.mse_loss(model(X), Y).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # ★ 收集所有预测和真实值
    all_preds, all_targets = [], []
    all_inputs = []  # 用于可视化

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(Y.cpu().numpy())
            all_inputs.append(X.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    inputs = np.vstack(all_inputs)

    # ★ Inverse transform back to original SOH scale before computing metrics.
    if scaler is not None and scaler.fitted:
        preds = scaler.inverse_transform(preds)
        targets = scaler.inverse_transform(targets)
        inputs = scaler.inverse_transform(inputs)

    # ★ 计算完整指标 (on original SOH scale)
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    mae = np.mean(np.abs(preds_flat - targets_flat))
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

    # MAPE (避免除零) - SOH 在 ~[0.7, 1.0]，阈值 0.01 安全
    mask = np.abs(targets_flat) > 0.01
    mape = np.mean(np.abs((targets_flat[mask] - preds_flat[mask]) / targets_flat[mask])) * 100 if mask.sum() > 0 else 0

    # R²
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Pearson相关系数
    if len(preds_flat) > 1 and np.std(preds_flat) > 1e-8:
        pearson = np.corrcoef(preds_flat, targets_flat)[0, 1]
        if np.isnan(pearson):
            pearson = 0
    else:
        pearson = 0

    # MaxAE, MedAE
    max_ae = np.max(np.abs(preds_flat - targets_flat))
    med_ae = np.median(np.abs(preds_flat - targets_flat))

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Pearson': pearson,
        'MaxAE': max_ae,
        'MedAE': med_ae,
    }

    if return_details:
        details = {
            'inputs': inputs,           # 输入序列 (N, input_dim) - on SOH scale
            'predictions': preds,       # 预测值 (N, output_dim) - on SOH scale
            'targets': targets,         # 真实值 (N, output_dim) - on SOH scale
            'metrics': metrics,
        }
        return metrics, details

    return metrics
