"""
Architecture equivalence test for Step 3d (Block-3 downstream).

The classes prefixed with ``_Old`` below are **verbatim copies** from
the user's original Block 3 script (the one used to produce paper
results). The verifier confirms that each refactored class in
``src.evaluation.predictors`` / ``src.augmentation.classical`` /
``src.evaluation.scaler`` / ``src.evaluation.metrics`` matches its
``_Old`` counterpart bit-for-bit.

To audit the verifier itself: open this file side-by-side with the
Block 3 source and confirm the ``_Old*`` definitions are identical
character-for-character. If they are, then `max|Δ|=0` between
"_Old" and "new" really does mean "new code = original code".
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------- Refactored versions under test -------
from src.augmentation.classical import (
    aug_gaussian_noise as new_gauss,
    aug_time_warping as new_warp,
)
from src.evaluation.metrics import compute_metrics as new_compute_metrics
from src.evaluation.predictors import (
    CNNPredictor as NewCNN,
    InformerPredictor as NewInformer,
    LSTMPredictor as NewLSTM,
    MLPPredictor as NewMLP,
    PatchTSTPredictor as NewPatchTST,
    RNNPredictor as NewRNN,
    TransformerPredictor as NewTransformer,
    iTransformerPredictor as NewiTransformer,
)
from src.evaluation.scaler import SOHScaler as NewSOHScaler


# ============================================================
# === VERBATIM COPIES FROM BLOCK 3 BELOW ===
# (do not modify — these define what "equivalent" means)
# ============================================================


# --- Block 3 predictors verbatim ---
class _OldMLPPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x): return self.net(x)


class _OldCNNPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (input_size // 4), 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        return self.fc(x.view(x.size(0), -1))


class _OldRNNPredictor(nn.Module):
    """GRU-based RNN Predictor"""
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_size)
    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class _OldLSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, output_size)
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class _OldTransformerPredictor(nn.Module):
    """Vanilla Transformer Encoder (Vaswani et al., NeurIPS 2017)"""
    def __init__(self, input_size, output_size, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1)) + self.pos_enc
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class _OldPatchTSTPredictor(nn.Module):
    def __init__(self, input_size, output_size, patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        self.num_patches = max(1, (input_size - patch_len) // stride + 1)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model * self.num_patches, output_size)
    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len), mode='replicate')
            seq_len = self.patch_len
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        actual_num_patches = patches.shape[1]
        x = self.patch_embedding(patches)
        if actual_num_patches <= self.num_patches:
            x = x + self.pos_enc[:, :actual_num_patches, :]
        else:
            pos_enc_interp = F.interpolate(
                self.pos_enc.transpose(1, 2), size=actual_num_patches,
                mode='linear', align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc_interp
        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(batch_size, -1)
        expected_dim = self.fc.in_features
        if x.shape[1] != expected_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), expected_dim).squeeze(1)
        return self.fc(x)


class _OldProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_k
        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        u = max(1, L // self.factor)
        M = scores.max(dim=-1)[0] - scores.mean(dim=-1)
        top_indices = M.topk(u, dim=-1)[1]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class _OldInformerPredictor(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, n_heads=4, n_layers=2, factor=5):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList()
        self.distill_convs = nn.ModuleList()
        current_len = input_size
        for i in range(n_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'attn': _OldProbSparseAttention(d_model, n_heads, factor),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }))
            if i < n_layers - 1:
                self.distill_convs.append(
                    nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
                )
                current_len = (current_len + 1) // 2
        self.final_len = current_len
        self.fc = nn.Linear(d_model * current_len, output_size)
    def forward(self, x):
        B, L = x.shape
        x = self.input_proj(x.unsqueeze(-1))
        if L <= self.input_size:
            x = x + self.pos_enc[:, :L, :]
        else:
            pos_enc_interp = F.interpolate(
                self.pos_enc.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc_interp
        for i, layer in enumerate(self.encoder_layers):
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))
            if i < len(self.distill_convs):
                x = self.distill_convs[i](x.transpose(1, 2)).transpose(1, 2)
                x = F.gelu(x)
        x = x.reshape(B, -1)
        expected_dim = self.fc.in_features
        if x.shape[1] != expected_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), expected_dim).squeeze(1)
        return self.fc(x)


class _OldiTransformerPredictor(nn.Module):
    def __init__(self, input_size, output_size, n_segments=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.n_segments = n_segments
        self.segment_len = max(1, input_size // n_segments)
        self.actual_input = self.segment_len * n_segments
        self.variate_embed = nn.Linear(self.segment_len, d_model)
        self.variate_pos = nn.Parameter(torch.randn(1, n_segments, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model * n_segments, output_size)
    def forward(self, x):
        B, L = x.shape
        if L < self.actual_input:
            x = F.pad(x, (0, self.actual_input - L), mode='replicate')
        elif L > self.actual_input:
            x = F.interpolate(x.unsqueeze(1), size=self.actual_input, mode='linear', align_corners=False).squeeze(1)
        x = x.view(B, self.n_segments, self.segment_len)
        x = self.variate_embed(x)
        x = x + self.variate_pos
        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(B, -1)
        return self.fc(x)


# --- Block 3 SOHScaler verbatim ---
def _old_ensure_1d_float(x):
    arr = np.array(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    return arr


class _OldSOHScaler:
    def __init__(self, buffer=0.05, eps=1e-8):
        self.min_val = None
        self.max_val = None
        self.raw_min = None
        self.raw_max = None
        self.buffer = float(buffer)
        self.eps = float(eps)
        self.fitted = False
    def fit(self, curves):
        all_vals = np.concatenate([_old_ensure_1d_float(c) for c in curves])
        self.raw_min = float(all_vals.min())
        self.raw_max = float(all_vals.max())
        self.min_val = self.raw_min - self.buffer
        self.max_val = self.raw_max + self.buffer
        self.fitted = True
        return self
    def transform(self, x):
        assert self.fitted
        x = np.asarray(x, dtype=np.float64)
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)
    def inverse_transform(self, x):
        assert self.fitted
        x = np.asarray(x, dtype=np.float64)
        return x * (self.max_val - self.min_val + self.eps) + self.min_val


# --- Block 3 classical augs verbatim ---
def _old_aug_gaussian_noise(curves, sigma=0.02, base_seed=2026, num_aug_per_battery=5):
    rng = np.random.RandomState(base_seed)
    aug = []
    for c in curves:
        scale = (c.max() - c.min()) * sigma
        for _ in range(num_aug_per_battery):
            noise = rng.normal(0, scale, len(c))
            aug.append(c + noise)
    return aug


def _old_aug_time_warping(curves, sigma=0.2, num_knots=4, base_seed=2026, num_aug_per_battery=5):
    rng = np.random.RandomState(base_seed)
    aug = []
    for c in curves:
        L = len(c)
        for _ in range(num_aug_per_battery):
            knot_xs = np.linspace(0, L - 1, num_knots + 2)
            knot_ys = np.cumsum(np.abs(rng.normal(1.0, sigma, num_knots + 2)))
            knot_ys = knot_ys / knot_ys[-1] * (L - 1)
            spline = CubicSpline(knot_xs, knot_ys)
            t_warped = np.clip(spline(np.arange(L)), 0, L - 1)
            aug.append(np.interp(np.arange(L), t_warped, c))
    return aug


# ============================================================
# === Comparison helpers ===
# ============================================================
def check(name, ok, extra=""):
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name}{('  ' + extra) if extra else ''}")
    return ok


def compare_predictor(NewCls, OldCls, label, input_size=150, output_size=350,
                      batch_size=4, seed=42, extra_kwargs=None):
    """Verify NewCls and OldCls produce bit-identical state_dicts and outputs."""
    extra_kwargs = extra_kwargs or {}
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    all_ok = True

    torch.manual_seed(seed)
    new_model = NewCls(input_size=input_size, output_size=output_size, **extra_kwargs)
    torch.manual_seed(seed)
    old_model = OldCls(input_size=input_size, output_size=output_size, **extra_kwargs)

    n_sd = new_model.state_dict(); o_sd = old_model.state_dict()
    ok = set(n_sd.keys()) == set(o_sd.keys())
    all_ok &= check("Parameter names match", ok, f"({len(n_sd)} params)")
    if not ok:
        m = set(o_sd.keys()) - set(n_sd.keys()); e = set(n_sd.keys()) - set(o_sd.keys())
        if m: print(f"    missing in new (first 5): {sorted(m)[:5]}")
        if e: print(f"    extra in new (first 5):   {sorted(e)[:5]}")
        return False

    max_d = max((n_sd[k] - o_sd[k]).abs().max().item() for k in n_sd)
    all_ok &= check("Initial weights identical", max_d == 0.0, f"max|Δ|={max_d:.2e}")

    new_model.eval(); old_model.eval()
    torch.manual_seed(0)
    x = torch.randn(batch_size, input_size)

    torch.manual_seed(123)
    with torch.no_grad():
        y_new = new_model(x)
    torch.manual_seed(123)
    with torch.no_grad():
        y_old = old_model(x)
    ok = y_new.shape == y_old.shape == (batch_size, output_size)
    all_ok &= check("Output shape matches", ok, f"shape={tuple(y_new.shape)}")
    if ok:
        max_d = (y_new - y_old).abs().max().item()
        all_ok &= check("Output values bit-identical", max_d == 0.0, f"max|Δ|={max_d:.2e}")
    return all_ok


def main():
    all_ok = True

    # ----- 8 predictors -----
    # Use Block 3's actual task dimensions (input_size=150, output_size=350 for SOH_30to70).
    all_ok &= compare_predictor(NewMLP, _OldMLPPredictor, "MLP")
    all_ok &= compare_predictor(NewCNN, _OldCNNPredictor, "CNN")
    all_ok &= compare_predictor(NewRNN, _OldRNNPredictor, "RNN (GRU)")
    all_ok &= compare_predictor(NewLSTM, _OldLSTMPredictor, "LSTM")
    all_ok &= compare_predictor(NewTransformer, _OldTransformerPredictor, "Transformer")
    all_ok &= compare_predictor(NewPatchTST, _OldPatchTSTPredictor, "PatchTST")
    all_ok &= compare_predictor(NewInformer, _OldInformerPredictor, "Informer")
    all_ok &= compare_predictor(NewiTransformer, _OldiTransformerPredictor, "iTransformer")

    # ----- SOHScaler -----
    print(f"\n{'-'*60}\nSOHScaler\n{'-'*60}")
    np.random.seed(0)
    train_curves = [np.random.uniform(0.7, 1.0, 100) for _ in range(20)]
    test_curve = np.random.uniform(0.65, 1.05, 100)
    s_new = NewSOHScaler(buffer=0.05).fit(train_curves)
    s_old = _OldSOHScaler(buffer=0.05).fit(train_curves)
    all_ok &= check("raw_min identical", s_new.raw_min == s_old.raw_min)
    all_ok &= check("raw_max identical", s_new.raw_max == s_old.raw_max)
    all_ok &= check("min_val identical", s_new.min_val == s_old.min_val)
    all_ok &= check("max_val identical", s_new.max_val == s_old.max_val)
    n_norm, o_norm = s_new.transform(test_curve), s_old.transform(test_curve)
    all_ok &= check("transform identical", np.array_equal(n_norm, o_norm),
                    f"max|Δ|={np.abs(n_norm - o_norm).max():.2e}")
    n_inv, o_inv = s_new.inverse_transform(n_norm), s_old.inverse_transform(o_norm)
    all_ok &= check("inverse_transform identical", np.array_equal(n_inv, o_inv),
                    f"max|Δ|={np.abs(n_inv - o_inv).max():.2e}")

    # ----- compute_metrics -----
    print(f"\n{'-'*60}\ncompute_metrics\n{'-'*60}")
    np.random.seed(1)
    targets = np.random.uniform(0.6, 1.0, 500)
    preds = targets + np.random.randn(500) * 0.05
    m_new = new_compute_metrics(preds, targets)

    # Inline Block-3 computation
    preds_flat = preds.flatten().astype(np.float64)
    targets_flat = targets.flatten().astype(np.float64)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))
    mask = np.abs(targets_flat) > 0.01
    mape = (np.mean(np.abs((targets_flat[mask] - preds_flat[mask]) / targets_flat[mask])) * 100
            if mask.sum() > 0 else 0)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    if len(preds_flat) > 1 and np.std(preds_flat) > 1e-8:
        pearson = np.corrcoef(preds_flat, targets_flat)[0, 1]
        if np.isnan(pearson): pearson = 0
    else:
        pearson = 0
    max_ae = np.max(np.abs(preds_flat - targets_flat))
    med_ae = np.median(np.abs(preds_flat - targets_flat))

    all_ok &= check("MAE",     m_new["MAE"] == mae)
    all_ok &= check("RMSE",    m_new["RMSE"] == rmse)
    all_ok &= check("MAPE",    m_new["MAPE"] == mape)
    all_ok &= check("R2",      m_new["R2"] == r2)
    all_ok &= check("Pearson", m_new["Pearson"] == pearson)
    all_ok &= check("MaxAE",   m_new["MaxAE"] == max_ae)
    all_ok &= check("MedAE",   m_new["MedAE"] == med_ae)

    # ----- Classical augmentation -----
    print(f"\n{'-'*60}\nClassical augmentation\n{'-'*60}")
    np.random.seed(0)
    base_curves = [np.linspace(2.0, 1.0, 200) + np.random.randn(200) * 0.01 for _ in range(3)]

    new_g = new_gauss(base_curves, sigma=0.02, num_aug_per_battery=5, base_seed=2026)
    old_g = _old_aug_gaussian_noise(base_curves, sigma=0.02, base_seed=2026, num_aug_per_battery=5)
    diffs = [np.abs(a - b).max() for a, b in zip(new_g, old_g)]
    all_ok &= check("Gaussian noise output bit-identical", max(diffs) == 0.0,
                    f"max|Δ|={max(diffs):.2e}")

    new_w = new_warp(base_curves, sigma=0.2, num_knots=4, num_aug_per_battery=5, base_seed=2026)
    old_w = _old_aug_time_warping(base_curves, sigma=0.2, num_knots=4, base_seed=2026, num_aug_per_battery=5)
    diffs = [np.abs(a - b).max() for a, b in zip(new_w, old_w)]
    all_ok &= check("TimeWarping output bit-identical", max(diffs) == 0.0,
                    f"max|Δ|={max(diffs):.2e}")

    print()
    print("=" * 60)
    if all_ok:
        print("✓ STEP-3d CODE IS BIT-IDENTICAL TO BLOCK 3 ORIGINAL")
        return 0
    print("✗ DIFFERENCES FOUND.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
