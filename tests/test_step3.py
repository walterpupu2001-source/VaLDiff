"""
Step-3 unit tests for the model + diffusion + dataset modules.

These don't require any real data — synthetic tensors and a stub
BatteryRaw are sufficient. Heavier numerical-equivalence checks against
the original code live in ``scripts/verify_step3_arch.py``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import CurveDataset1D, collate_keep_meta, interp_to_len
from src.diffusion.ddpm import StandardDDPM, linear_beta_schedule
from src.models.common import ResBlock1D, SinusoidalEmbedding
from src.models.unet1d import UNet1D


# ============================================================
# interp_to_len
# ============================================================
class TestInterpToLen:
    def test_target_length(self):
        out = interp_to_len(np.linspace(0, 1, 100), 50)
        assert out.shape == (50,) and out.dtype == np.float64

    def test_endpoints_preserved(self):
        y = np.array([2.0, 1.0])
        out = interp_to_len(y, 10)
        assert out[0] == 2.0 and out[-1] == 1.0

    def test_empty_returns_zeros(self):
        out = interp_to_len(np.array([]), 5)
        assert out.shape == (5,) and (out == 0).all()

    def test_single_value_broadcast(self):
        out = interp_to_len(np.array([3.14]), 4)
        assert (out == 3.14).all()


# ============================================================
# SinusoidalEmbedding
# ============================================================
class TestSinusoidalEmbedding:
    def test_output_shape(self):
        emb = SinusoidalEmbedding(dim=128)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 128)

    def test_deterministic(self):
        emb = SinusoidalEmbedding(dim=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out1 = emb(t)
        out2 = emb(t)
        assert torch.equal(out1, out2)


# ============================================================
# ResBlock1D
# ============================================================
class TestResBlock1D:
    def test_residual_when_same_channels(self):
        blk = ResBlock1D(8, 8, t_dim=16, c_dim=16)
        assert isinstance(blk.skip, torch.nn.Identity)

    def test_1x1_skip_when_different_channels(self):
        blk = ResBlock1D(8, 16, t_dim=16, c_dim=16)
        assert isinstance(blk.skip, torch.nn.Conv1d)
        assert blk.skip.kernel_size == (1,)

    def test_forward_shape(self):
        blk = ResBlock1D(8, 16, t_dim=32, c_dim=32)
        x = torch.randn(2, 8, 100)
        t = torch.randn(2, 32)
        c = torch.randn(2, 32)
        y = blk(x, t, c)
        assert y.shape == (2, 16, 100)


# ============================================================
# UNet1D
# ============================================================
class TestUNet1D:
    def test_forward_shape(self):
        m = UNet1D(img_ch=1, base=32, cond_dim=13, t_dim=64)
        x = torch.randn(2, 1, 384)
        t = torch.rand(2)
        c = torch.randn(2, 13)
        y = m(x, t, c)
        assert y.shape == x.shape

    def test_parameter_count_reasonable(self):
        m = UNet1D(img_ch=1, base=128, cond_dim=13, t_dim=128)
        n = sum(p.numel() for p in m.parameters())
        # Order of magnitude check: a few million params for base=128.
        assert 1_000_000 < n < 20_000_000

    def test_state_dict_keys_stable(self):
        """If keys change, pre-trained checkpoints won't load — guard against that."""
        m = UNet1D(img_ch=1, base=128, cond_dim=13, t_dim=128)
        keys = sorted(m.state_dict().keys())
        # Spot-check a few keys we expect.
        assert "enc1_in.weight" in keys
        assert "out.weight" in keys
        # Existence of t_mlp / c_mlp linear layers
        assert any(k.startswith("t_mlp.") for k in keys)
        assert any(k.startswith("c_mlp.") for k in keys)


# ============================================================
# StandardDDPM
# ============================================================
class TestStandardDDPM:
    def test_schedule_shapes(self):
        m = UNet1D(base=32)
        ddpm = StandardDDPM(m, T=1000, device="cpu")
        assert ddpm.betas.shape == (1000,)
        assert ddpm.alpha_bar.shape == (1000,)
        assert ddpm.alpha_bar_prev.shape == (1000,)
        assert ddpm.alpha_bar_prev[0] == 1.0
        assert ddpm.posterior_var.shape == (1000,)

    def test_q_sample_at_t_zero(self):
        """At t=0, q_sample should give back roughly x0 (alpha_bar≈1)."""
        m = UNet1D(base=32)
        ddpm = StandardDDPM(m, T=1000, device="cpu")
        x0 = torch.randn(2, 1, 64)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.zeros_like(x0)  # no noise
        x_t = ddpm.q_sample(x0, t, noise=noise)
        # alpha_bar[0] = 1 - betas[0] ≈ 0.9999, so x_t ≈ sqrt(0.9999) * x0
        assert torch.allclose(x_t, x0 * np.sqrt(1 - 1e-4), atol=1e-5)

    def test_train_step_runs_and_returns_scalar_loss(self):
        torch.manual_seed(0)
        m = UNet1D(base=32, cond_dim=13)
        ddpm = StandardDDPM(m, T=100, device="cpu")
        x0 = torch.randn(4, 1, 64)
        c = torch.randn(4, 13)
        loss = ddpm.train_step(x0, c)
        assert loss.dim() == 0 and loss.requires_grad

    def test_train_step_masked_loss_uses_only_masked_region(self):
        torch.manual_seed(0)
        m = UNet1D(base=16, cond_dim=4)
        ddpm = StandardDDPM(m, T=10, device="cpu")
        x0 = torch.randn(2, 1, 32)
        c = torch.randn(2, 4)
        all_ones = torch.ones_like(x0)
        torch.manual_seed(1); l_full = ddpm.train_step(x0, c)
        torch.manual_seed(1); l_masked = ddpm.train_step(x0, c, mask=all_ones)
        # Masked loss with all-ones mask should match unmasked loss closely.
        assert abs(l_full.item() - l_masked.item()) < 1e-5


# ============================================================
# CurveDataset1D
# ============================================================
def _fake_batt(name: str, n_cycles: int, batch_id: int = 1):
    """Stub BatteryRaw exposing only the fields CurveDataset1D consumes."""
    return SimpleNamespace(
        battery_name=name,
        batch_id=batch_id,
        capacity_curve=np.linspace(2.0, 1.0, n_cycles).astype(np.float64),
        physics_features={
            k: float(i) for i, k in enumerate([
                "voltage_mean", "voltage_std", "voltage_min", "voltage_max",
                "voltage_kurtosis", "voltage_skewness",
                "current_mean", "current_std",
                "current_kurtosis", "current_skewness",
                "temp_mean", "temp_std", "temp_max",
            ])
        },
        get_condition_numeric=lambda: np.arange(13, dtype=np.float64),
    )


class TestCurveDataset1D:
    def test_basic_shapes(self, capsys):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset1D(batts, seq_len=64, cond_num_mean=cmean, cond_num_std=cstd)
        capsys.readouterr()  # silence

        assert len(ds) == 3
        x, c, meta = ds[0]
        assert x.shape == (1, 64)
        assert c.shape == (13,)
        assert meta["seq_len"] == 64
        assert meta["orig_len"] == 200

    def test_normalisation_in_range(self):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset1D(batts, seq_len=64, cond_num_mean=cmean, cond_num_std=cstd)
        for i in range(len(ds)):
            x, _, _ = ds[i]
            assert -1.0 <= x.min().item() and x.max().item() <= 1.0

    def test_denorm_curve_inverts(self):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset1D(batts, seq_len=64, cond_num_mean=cmean, cond_num_std=cstd)
        x, _, meta = ds[0]
        x_back = CurveDataset1D.denorm_curve(x.numpy()[0], meta)
        # Should approximately recover the resampled curve.
        assert x_back.min() >= 1.0 - 1e-6
        assert x_back.max() <= 2.0 + 1e-6


# ============================================================
# collate_keep_meta
# ============================================================
class TestCollate:
    def test_stacks_tensors_and_preserves_metas(self):
        batch = [
            (torch.zeros(1, 4), torch.zeros(3), {"a": 1}),
            (torch.ones(1, 4),  torch.ones(3),  {"a": 2}),
        ]
        xs, cs, ms = collate_keep_meta(batch)
        assert xs.shape == (2, 1, 4) and cs.shape == (2, 3)
        assert ms == [{"a": 1}, {"a": 2}]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
