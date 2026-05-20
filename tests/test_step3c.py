"""
Step-3c unit tests for GAN + VAE + unconditional dataset.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import CurveDataset1D_NoCond, no_cond_collate
from src.models.gan import DiscriminatorGAN, GeneratorGAN
from src.models.vae import VAE, Decoder, Encoder, vae_loss


def _fake_batt(name, n_cycles):
    return SimpleNamespace(
        battery_name=name,
        capacity_curve=np.linspace(2.0, 1.0, n_cycles).astype(np.float64),
    )


# ============================================================
# GAN
# ============================================================
class TestGeneratorGAN:
    def test_output_shape(self):
        G = GeneratorGAN(latent_dim=64, seq_len=384, ch=16)
        # BatchNorm needs at least batch size 2 in train mode.
        G.eval()
        z = torch.randn(2, 64)
        out = G(z)
        assert out.shape == (2, 384)

    def test_output_is_bounded_by_tanh(self):
        G = GeneratorGAN(latent_dim=64, seq_len=384, ch=16)
        G.eval()
        z = torch.randn(2, 64)
        out = G(z)
        assert out.min().item() >= -1.0 and out.max().item() <= 1.0


class TestDiscriminatorGAN:
    def test_output_shape(self):
        D = DiscriminatorGAN(seq_len=384, ch=16)
        x = torch.randn(2, 384)
        out = D(x)
        assert out.shape == (2,)

    def test_output_is_sigmoid_range(self):
        D = DiscriminatorGAN(seq_len=384, ch=16)
        x = torch.randn(2, 384)
        out = D(x)
        assert (out >= 0).all() and (out <= 1).all()


# ============================================================
# VAE
# ============================================================
class TestEncoder:
    def test_outputs_mu_and_logvar(self):
        enc = Encoder(seq_len=384, latent_dim=8, ch=16)
        x = torch.randn(2, 384)
        mu, logvar = enc(x)
        assert mu.shape == (2, 8) and logvar.shape == (2, 8)


class TestDecoder:
    def test_output_shape_and_range(self):
        dec = Decoder(seq_len=384, latent_dim=8, ch=16)
        z = torch.randn(2, 8)
        out = dec(z)
        assert out.shape == (2, 384)
        assert out.min().item() >= -1.0 and out.max().item() <= 1.0


class TestVAE:
    def test_forward_shapes(self):
        m = VAE(seq_len=384, latent_dim=8, ch=16)
        x = torch.randn(2, 384)
        recon, mu, logvar = m(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 8) and logvar.shape == (2, 8)

    def test_sample_returns_correct_shape(self):
        m = VAE(seq_len=384, latent_dim=8, ch=16)
        out = m.sample(4, "cpu")
        assert out.shape == (4, 384)

    def test_reparameterize_is_differentiable(self):
        m = VAE(seq_len=384, latent_dim=4, ch=8)
        x = torch.randn(2, 384, requires_grad=True)
        recon, mu, logvar = m(x)
        loss, _, _ = vae_loss(x, recon, mu, logvar)
        loss.backward()
        # Some parameter should have a gradient.
        assert any(p.grad is not None for p in m.parameters())


class TestVaeLoss:
    def test_kl_zero_when_unit_gaussian(self):
        """KL(N(0, I) || N(0, I)) should be 0."""
        x = torch.randn(2, 384)
        recon = x.clone()
        mu = torch.zeros(2, 8)
        logvar = torch.zeros(2, 8)
        _, _, kl = vae_loss(x, recon, mu, logvar)
        assert abs(kl.item()) < 1e-6


# ============================================================
# CurveDataset1D_NoCond
# ============================================================
class TestCurveDataset1DNoCond:
    def test_basic_shapes(self, capsys):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        ds = CurveDataset1D_NoCond(batts, seq_len=64)
        capsys.readouterr()
        assert len(ds) == 3
        x, meta = ds[0]
        assert x.shape == (64,)
        assert meta["seq_len"] == 64
        assert meta["orig_len"] == 200

    def test_no_condition_returned(self):
        """Only (x, meta) — no condition vector, unlike CurveDataset1D."""
        batts = [_fake_batt(f"b{i}", 200) for i in range(2)]
        ds = CurveDataset1D_NoCond(batts, seq_len=64)
        item = ds[0]
        assert len(item) == 2  # tensor + meta only

    def test_normalisation_in_range(self):
        batts = [_fake_batt(f"b{i}", 200) for i in range(3)]
        ds = CurveDataset1D_NoCond(batts, seq_len=64)
        for i in range(len(ds)):
            x, _ = ds[i]
            assert -1.0 <= x.min().item() and x.max().item() <= 1.0

    def test_collate_pairs(self):
        batts = [_fake_batt(f"b{i}", 200) for i in range(4)]
        ds = CurveDataset1D_NoCond(batts, seq_len=64)
        batch = [ds[i] for i in range(2)]
        xs, ms = no_cond_collate(batch)
        assert xs.shape == (2, 64)
        assert len(ms) == 2 and isinstance(ms[0], dict)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
