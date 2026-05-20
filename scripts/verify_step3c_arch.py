"""
Architecture equivalence test for Step 3c (GAN + VAE).

Same strategy as the earlier verifiers: embed verbatim copies of the
original Block-2 ``Generator`` / ``Discriminator`` / ``Encoder`` / ``Decoder``
/ ``VAE`` here, initialise both old and new versions with the same seed,
and confirm bit-identical weights and forward outputs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gan import GeneratorGAN as NewGenerator
from src.models.gan import DiscriminatorGAN as NewDiscriminator
from src.models.vae import VAE as NewVAE
from src.models.vae import vae_loss as new_vae_loss


# ============================================================
# Verbatim copies of the original Block-2 implementations
# ============================================================
class _OldGenerator(nn.Module):
    def __init__(self, latent_dim=128, seq_len=384, ch=64):
        super().__init__()
        self.init_len = seq_len // 16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ch * 8 * self.init_len),
            nn.BatchNorm1d(ch * 8 * self.init_len),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm1d(ch * 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm1d(ch * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.init_len)
        return self.conv(x).squeeze(1)


class _OldDiscriminator(nn.Module):
    def __init__(self, seq_len=384, ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(ch, ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(ch * 2, ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Conv1d(ch * 4, ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ch * 8 * (seq_len // 16), 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x).squeeze(1)


class _OldEncoder(nn.Module):
    def __init__(self, seq_len=384, latent_dim=32, ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch * 2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(ch * 2, ch * 4, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(ch * 4, ch * 8, 4, 2, 1), nn.LeakyReLU(0.2),
        )
        flat = ch * 8 * (seq_len // 16)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class _OldDecoder(nn.Module):
    def __init__(self, seq_len=384, latent_dim=32, ch=64):
        super().__init__()
        self.init_len = seq_len // 16; self.ch = ch
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ch * 8 * self.init_len),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.ch * 8, self.init_len)
        return self.conv(h).squeeze(1)


class _OldVAE(nn.Module):
    def __init__(self, seq_len=384, latent_dim=32, ch=64):
        super().__init__()
        self.encoder = _OldEncoder(seq_len, latent_dim, ch)
        self.decoder = _OldDecoder(seq_len, latent_dim, ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def _old_vae_loss(x, x_recon, mu, logvar, kl_weight=0.0005):
    recon = F.mse_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


# ============================================================
# Comparison helpers
# ============================================================
def check(name, ok, extra=""):
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name}{('  ' + extra) if extra else ''}")
    return ok


def compare_module_pair(new_model, old_model, label):
    print(f"\n{'-'*60}")
    print(label)
    print(f"{'-'*60}")
    all_ok = True
    new_sd, old_sd = new_model.state_dict(), old_model.state_dict()
    ok = set(new_sd.keys()) == set(old_sd.keys())
    all_ok &= check("Parameter names match", ok, f"({len(new_sd)} params)")
    if not ok:
        missing = set(old_sd.keys()) - set(new_sd.keys())
        extra = set(new_sd.keys()) - set(old_sd.keys())
        if missing: print(f"    missing in new (first 5): {sorted(missing)[:5]}")
        if extra:   print(f"    extra in new (first 5):   {sorted(extra)[:5]}")
        return all_ok
    max_diff = max((new_sd[k] - old_sd[k]).abs().max().item() for k in new_sd)
    all_ok &= check("Initial weights identical", max_diff == 0.0, f"max|Δ|={max_diff:.2e}")
    return all_ok


def main():
    all_ok = True

    # ---- GeneratorGAN ----
    torch.manual_seed(42)
    new_g = NewGenerator(latent_dim=128, seq_len=384, ch=64)
    torch.manual_seed(42)
    old_g = _OldGenerator(latent_dim=128, seq_len=384, ch=64)
    all_ok &= compare_module_pair(new_g, old_g, "GAN: Generator (use_upblock irrelevant)")

    new_g.eval(); old_g.eval()
    torch.manual_seed(0)
    z = torch.randn(2, 128)
    with torch.no_grad():
        y_new, y_old = new_g(z), old_g(z)
    all_ok &= check("  Generator output bit-identical",
                    torch.equal(y_new, y_old),
                    f"max|Δ|={(y_new - y_old).abs().max().item():.2e}")

    # ---- DiscriminatorGAN ----
    torch.manual_seed(42)
    new_d = NewDiscriminator(seq_len=384, ch=64)
    torch.manual_seed(42)
    old_d = _OldDiscriminator(seq_len=384, ch=64)
    all_ok &= compare_module_pair(new_d, old_d, "GAN: Discriminator")

    new_d.eval(); old_d.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 384)
    with torch.no_grad():
        y_new, y_old = new_d(x), old_d(x)
    all_ok &= check("  Discriminator output bit-identical",
                    torch.equal(y_new, y_old),
                    f"max|Δ|={(y_new - y_old).abs().max().item():.2e}")

    # ---- VAE ----
    torch.manual_seed(42)
    new_v = NewVAE(seq_len=384, latent_dim=32, ch=64)
    torch.manual_seed(42)
    old_v = _OldVAE(seq_len=384, latent_dim=32, ch=64)
    all_ok &= compare_module_pair(new_v, old_v, "VAE")

    # VAE has stochastic reparameterisation — but with the same seed,
    # both versions should produce identical noise → identical outputs.
    new_v.eval(); old_v.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 384)
    torch.manual_seed(123)
    new_recon, new_mu, new_logvar = new_v(x)
    torch.manual_seed(123)
    old_recon, old_mu, old_logvar = old_v(x)
    all_ok &= check("  VAE forward: mu bit-identical",
                    torch.equal(new_mu, old_mu),
                    f"max|Δ|={(new_mu - old_mu).abs().max().item():.2e}")
    all_ok &= check("  VAE forward: logvar bit-identical",
                    torch.equal(new_logvar, old_logvar),
                    f"max|Δ|={(new_logvar - old_logvar).abs().max().item():.2e}")
    all_ok &= check("  VAE forward: reconstruction bit-identical",
                    torch.equal(new_recon, old_recon),
                    f"max|Δ|={(new_recon - old_recon).abs().max().item():.2e}")

    # ---- vae_loss equivalence ----
    print(f"\n{'-'*60}")
    print("vae_loss equivalence")
    print(f"{'-'*60}")
    torch.manual_seed(0)
    x = torch.randn(4, 384)
    x_recon = torch.randn(4, 384)
    mu = torch.randn(4, 32)
    logvar = torch.randn(4, 32)
    L_new, R_new, K_new = new_vae_loss(x, x_recon, mu, logvar, kl_weight=0.0005)
    L_old, R_old, K_old = _old_vae_loss(x, x_recon, mu, logvar, kl_weight=0.0005)
    all_ok &= check("Total loss", L_new.item() == L_old.item(),
                    f"new={L_new.item():.10f} old={L_old.item():.10f}")
    all_ok &= check("Reconstruction loss", R_new.item() == R_old.item())
    all_ok &= check("KL loss", K_new.item() == K_old.item())

    # ---- Verdict ----
    print()
    print("=" * 60)
    if all_ok:
        print("✓ STEP-3c CODE IS BIT-IDENTICAL TO ORIGINAL")
        return 0
    print("✗ DIFFERENCES FOUND.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
