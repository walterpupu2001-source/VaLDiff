"""
Step-3d unit tests: predictors, scaler, datasets, metrics, augmentation.

All tests use the Block-3 API (q_nominal, fixed_len/input_ratio,
num_aug_per_battery, base_seed). Do NOT introduce keyword arguments
that don't exist in Block 3.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation.classical import (
    aug_gaussian_noise,
    aug_time_warping,
)
from src.evaluation.datasets import SOHCurveDataset
from src.evaluation.metrics import compute_metrics
from src.evaluation.predictors import (
    PREDICTOR_REGISTRY,
    CNNPredictor,
    InformerPredictor,
    LSTMPredictor,
    MLPPredictor,
    PatchTSTPredictor,
    RNNPredictor,
    TransformerPredictor,
    iTransformerPredictor,
)
from src.evaluation.scaler import (
    SOHScaler,
    capacity_list_to_soh,
    capacity_to_soh_curve,
)


# ============================================================
# SOHScaler
# ============================================================
class TestSOHScaler:
    def test_capacity_to_soh(self):
        cap = np.array([2.0, 1.8, 1.6])
        soh = capacity_to_soh_curve(cap, q_nominal=2.0)
        assert np.allclose(soh, [1.0, 0.9, 0.8])

    def test_unfit_raises_on_transform(self):
        s = SOHScaler()
        with pytest.raises(AssertionError):
            s.transform(np.array([0.5]))

    def test_fit_then_transform_inverts(self):
        curves = [np.random.uniform(0.7, 1.0, 50) for _ in range(10)]
        s = SOHScaler(buffer=0.05).fit(curves)
        for c in curves:
            assert np.allclose(s.inverse_transform(s.transform(c)), c, atol=1e-9)

    def test_buffer_extends_range(self):
        curves = [np.array([0.7, 1.0])]
        s = SOHScaler(buffer=0.05).fit(curves)
        assert abs(s.min_val - 0.65) < 1e-9
        assert abs(s.max_val - 1.05) < 1e-9

    def test_fitted_flag_set(self):
        s = SOHScaler(buffer=0.05)
        assert s.fitted is False
        s.fit([np.array([0.7, 1.0])])
        assert s.fitted is True


# ============================================================
# SOHCurveDataset (Block 3 API: fixed_len + input_ratio)
# ============================================================
class TestSOHCurveDataset:
    def test_resample_then_split_at_input_ratio(self):
        # Block 3 first resamples each curve to fixed_len, then splits at input_ratio.
        curves = [np.linspace(1.0, 0.7, 100) for _ in range(3)]
        ds = SOHCurveDataset(curves, None, None, fixed_len=500, input_ratio=0.3)
        assert len(ds) == 3
        x, y = ds[0]
        # input_size = int(500 * 0.3) = 150; output_size = 500 - 150 = 350
        assert x.shape == (150,)
        assert y.shape == (350,)

    def test_scaler_applied_to_both_x_and_y(self):
        curves = [np.linspace(1.0, 0.7, 100) for _ in range(3)]
        s = SOHScaler(buffer=0.05).fit(curves)
        ds = SOHCurveDataset(curves, None, None, fixed_len=500, input_ratio=0.5, scaler=s)
        x, y = ds[0]
        # After min-max normalization, all values should be in roughly [0.05/0.35, 1-0.05/0.35]
        # i.e. between ~0.1 and ~0.9
        assert x.min() > 0.0 and x.max() < 1.0
        assert y.min() > 0.0 and y.max() < 1.0

    def test_short_curves_filtered(self):
        # Block 3: if len(soh) < 50, skip.
        curves = [np.linspace(1.0, 0.7, 30), np.linspace(1.0, 0.7, 100)]
        ds = SOHCurveDataset(curves, None, None, fixed_len=500, input_ratio=0.5)
        assert len(ds) == 1


# ============================================================
# compute_metrics
# ============================================================
class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.linspace(0.7, 1.0, 100)
        m = compute_metrics(y, y)
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["MAPE"] == 0.0
        assert abs(m["R2"] - 1.0) < 1e-6

    def test_keys(self):
        y = np.linspace(0.7, 1.0, 100)
        m = compute_metrics(y + 0.1, y)
        for k in ("MAE", "RMSE", "MAPE", "R2", "Pearson", "MaxAE", "MedAE"):
            assert k in m

    def test_mape_excludes_near_zero(self):
        # Block 3 uses mask = |targets| > 0.01 to avoid div-by-zero.
        targets = np.array([0.0, 0.0, 0.5, 0.5])
        preds = np.array([0.1, 0.2, 0.45, 0.55])
        m = compute_metrics(preds, targets)
        expected = float(np.mean(np.abs([0.05 / 0.5, 0.05 / 0.5])) * 100)
        assert abs(m["MAPE"] - expected) < 1e-9


# ============================================================
# Predictors — shape / registry
# ============================================================
class TestPredictorShapes:
    @pytest.mark.parametrize("name,cls", [
        ("MLP", MLPPredictor), ("CNN", CNNPredictor),
        ("RNN", RNNPredictor), ("LSTM", LSTMPredictor),
        ("Transformer", TransformerPredictor),
        ("PatchTST", PatchTSTPredictor),
        ("Informer", InformerPredictor),
        ("iTransformer", iTransformerPredictor),
    ])
    def test_forward_shape(self, name, cls):
        model = cls(input_size=64, output_size=32)
        model.eval()
        x = torch.randn(2, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 32), f"{name} output shape was {y.shape}"

    def test_registry_complete(self):
        expected = {"MLP", "CNN", "RNN", "LSTM", "Transformer",
                    "PatchTST", "Informer", "iTransformer"}
        assert set(PREDICTOR_REGISTRY.keys()) == expected


# ============================================================
# Classical augmentations (Block 3 API: num_aug_per_battery, base_seed)
# ============================================================
class TestClassicalAug:
    def test_gaussian_output_length(self):
        curves = [np.ones(50) for _ in range(3)]
        out = aug_gaussian_noise(curves, sigma=0.01, num_aug_per_battery=4)
        assert len(out) == 12  # 3 × 4
        for c in out:
            assert len(c) == 50

    def test_gaussian_deterministic_with_base_seed(self):
        curves = [np.ones(50) for _ in range(2)]
        a = aug_gaussian_noise(curves, sigma=0.01, num_aug_per_battery=3, base_seed=7)
        b = aug_gaussian_noise(curves, sigma=0.01, num_aug_per_battery=3, base_seed=7)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_gaussian_scale_is_range_scaled(self):
        # Block 3 uses scale = (c.max() - c.min()) * sigma.
        # For constant curves, scale=0 → noise=0 → output == input.
        curves = [np.full(50, 0.8)]  # range = 0
        out = aug_gaussian_noise(curves, sigma=0.5, num_aug_per_battery=2)
        for c in out:
            assert np.allclose(c, 0.8)

    def test_time_warping_preserves_length(self):
        curves = [np.linspace(2.0, 1.0, 100) for _ in range(2)]
        out = aug_time_warping(curves, sigma=0.2, num_knots=4, num_aug_per_battery=3)
        assert len(out) == 6
        for c in out:
            assert len(c) == 100


# ============================================================
# capacity_list_to_soh
# ============================================================
class TestCapacityList:
    def test_returns_same_count(self):
        cap_curves = [np.array([2.0, 1.9]), np.array([2.0, 1.8, 1.6])]
        soh = capacity_list_to_soh(cap_curves, q_nominal=2.0)
        assert len(soh) == 2
        assert np.allclose(soh[0], [1.0, 0.95])
        assert np.allclose(soh[1], [1.0, 0.9, 0.8])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
