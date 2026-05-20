"""
Step-3b unit tests for the 2-D model + dataset + padding-constraint modules.

These don't require real data — synthetic tensors and stub batteries
suffice. Heavier numerical-equivalence checks against the original
implementations live in ``scripts/verify_step3b_arch.py``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import (
    CurveDataset2D_AutoDE,
    CurveDataset2D_ResampledDE,
    collate_keep_meta,
)
from src.diffusion.ddpm import StandardDDPM
from src.diffusion.proposed_constraints import (
    build_mask_from_metas,
    enforce_padding_constraint,
    make_padding_fn,
)
from src.models.common import ResBlock2D, UpBlock2D
from src.models.unet2d import UNet2D


# ============================================================
# Test fixtures
# ============================================================
def _fake_batt(name: str, n_cycles: int, batch_id: int = 1):
    return SimpleNamespace(
        battery_name=name,
        batch_id=batch_id,
        capacity_curve=np.linspace(2.0, 1.0, n_cycles).astype(np.float64),
        get_condition_numeric=lambda: np.arange(13, dtype=np.float64),
    )


# ============================================================
# ResBlock2D
# ============================================================
class TestResBlock2D:
    def test_residual_when_same_channels(self):
        blk = ResBlock2D(8, 8, t_dim=16, c_dim=16)
        assert isinstance(blk.skip, torch.nn.Identity)

    def test_1x1_skip_when_different_channels(self):
        blk = ResBlock2D(8, 16, t_dim=16, c_dim=16)
        assert isinstance(blk.skip, torch.nn.Conv2d)
        assert blk.skip.kernel_size == (1, 1)

    def test_forward_shape(self):
        blk = ResBlock2D(4, 8, t_dim=16, c_dim=16)
        x = torch.randn(2, 4, 32, 32)
        t = torch.randn(2, 16)
        c = torch.randn(2, 16)
        assert blk(x, t, c).shape == (2, 8, 32, 32)


# ============================================================
# UpBlock2D
# ============================================================
class TestUpBlock2D:
    def test_doubles_spatial_dims(self):
        blk = UpBlock2D(8, 8)
        x = torch.randn(2, 8, 16, 16)
        assert blk(x).shape == (2, 8, 32, 32)

    def test_changes_channels(self):
        blk = UpBlock2D(16, 8)
        x = torch.randn(2, 16, 8, 8)
        assert blk(x).shape == (2, 8, 16, 16)


# ============================================================
# UNet2D
# ============================================================
class TestUNet2D:
    def test_forward_shape_resampled(self):
        m = UNet2D(img_ch=1, base=32, cond_dim=13, t_dim=64, use_upblock=False)
        x = torch.randn(2, 1, 32, 32)
        t = torch.rand(2); c = torch.randn(2, 13)
        assert m(x, t, c).shape == x.shape

    def test_forward_shape_proposed(self):
        m = UNet2D(img_ch=1, base=32, cond_dim=13, t_dim=64, use_upblock=True)
        x = torch.randn(2, 1, 32, 32)
        t = torch.rand(2); c = torch.randn(2, 13)
        assert m(x, t, c).shape == x.shape

    def test_state_dict_keys_differ_between_modes(self):
        """Sanity: the two modes really do produce different state_dicts."""
        m_r = UNet2D(base=32, cond_dim=13, t_dim=64, use_upblock=False)
        m_p = UNet2D(base=32, cond_dim=13, t_dim=64, use_upblock=True)
        assert set(m_r.state_dict().keys()) != set(m_p.state_dict().keys())

    def test_resampled_up_layer_is_convtranspose(self):
        m = UNet2D(base=32, use_upblock=False)
        assert isinstance(m.up1, torch.nn.ConvTranspose2d)
        assert isinstance(m.up2, torch.nn.ConvTranspose2d)

    def test_proposed_up_layer_is_upblock(self):
        m = UNet2D(base=32, use_upblock=True)
        assert isinstance(m.up1, UpBlock2D)
        assert isinstance(m.up2, UpBlock2D)


# ============================================================
# Padding-constraint helpers
# ============================================================
class TestPaddingConstraints:
    def test_mask_shape_and_values(self):
        metas = [{"orig_q": 20}, {"orig_q": 32}, {"orig_q": 1}]
        m = build_mask_from_metas(metas, img_size=32, device="cpu")
        assert m.shape == (3, 1, 32, 32)
        # Sample 0: cols 0..20 should be 1, the rest 0
        assert (m[0, 0, :, :20] == 1).all()
        assert (m[0, 0, :, 20:] == 0).all()
        # Sample 1: all 1 (orig_q == img_size)
        assert (m[1] == 1).all()
        # Sample 2: only col 0 is 1
        assert (m[2, 0, :, 0] == 1).all() and (m[2, 0, :, 1:] == 0).all()

    def test_padding_constraint_copies_last_column(self):
        metas = [{"orig_q": 5}]
        torch.manual_seed(0)
        x = torch.randn(1, 1, 8, 8)
        col4 = x[0, 0, :, 4].clone()
        x2 = enforce_padding_constraint(x.clone(), metas, img_size=8)
        # Every padded column should equal column 4
        for j in range(5, 8):
            assert torch.equal(x2[0, 0, :, j], col4)

    def test_make_padding_fn_returns_callable(self):
        fn = make_padding_fn([{"orig_q": 4}], img_size=8)
        assert callable(fn)
        x = torch.zeros(1, 1, 8, 8); x[0, 0, :, 3] = 7.0
        out = fn(x)
        assert (out[0, 0, :, 4:] == 7.0).all()


# ============================================================
# CurveDataset2D_ResampledDE
# ============================================================
class TestCurveDataset2DResampled:
    def test_basic_shapes(self, capsys):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset2D_ResampledDE(
            batts, img_size=32, seq_len=500,
            cond_num_mean=cmean, cond_num_std=cstd,
        )
        capsys.readouterr()
        assert len(ds) == 3
        x, c, meta = ds[0]
        assert x.shape == (1, 32, 32)
        assert c.shape == (13,)
        assert meta["seq_len"] == 500
        assert meta["orig_len"] == 200

    def test_normalisation_in_range(self):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset2D_ResampledDE(
            batts, 32, 500, cmean, cstd
        )
        for i in range(len(ds)):
            x, _, _ = ds[i]
            assert -1.0 <= x.min().item() and x.max().item() <= 1.0


# ============================================================
# CurveDataset2D_AutoDE
# ============================================================
class TestCurveDataset2DAuto:
    def test_basic_shapes_and_orig_q(self, capsys):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        cmean = np.zeros(13); cstd = np.ones(13)
        ds = CurveDataset2D_AutoDE(
            batts, img_size=32, cond_num_mean=cmean, cond_num_std=cstd,
        )
        capsys.readouterr()
        assert len(ds) == 3
        x, c, meta = ds[0]
        assert x.shape == (1, 32, 32)
        assert "orig_q" in meta
        assert 1 <= meta["orig_q"] <= 32


# ============================================================
# DDPM dim-agnostic broadcasting for 2D
# ============================================================
class TestDDPM2D:
    def test_q_sample_works_for_2d(self):
        m = UNet2D(base=8, cond_dim=4, t_dim=16, use_upblock=False)
        ddpm = StandardDDPM(m, T=100, device="cpu")
        x0 = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0, 50], dtype=torch.long)
        noise = torch.zeros_like(x0)
        x_t = ddpm.q_sample(x0, t, noise=noise)
        assert x_t.shape == x0.shape

    def test_train_step_2d_with_mask(self):
        torch.manual_seed(0)
        m = UNet2D(base=8, cond_dim=4, t_dim=16, use_upblock=True)
        ddpm = StandardDDPM(m, T=10, device="cpu")
        x0 = torch.randn(2, 1, 32, 32)
        c = torch.randn(2, 4)
        metas = [{"orig_q": 10}, {"orig_q": 20}]
        mask = build_mask_from_metas(metas, 32, "cpu")
        pad_fn = make_padding_fn(metas, 32)
        loss = ddpm.train_step(x0, c, mask=mask, padding_fn=pad_fn)
        assert loss.dim() == 0 and loss.requires_grad


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
