"""
1-D Vanilla β-VAE (unconditional baseline).

Verbatim port of the Block-2 Vanilla-VAE classes. The β coefficient is
the ``kl_weight`` argument to ``vae_loss``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """4-layer strided-conv encoder, producing ``(mu, logvar)``."""

    def __init__(self, seq_len: int = 384, latent_dim: int = 32, ch: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 4, 2, 1),         # /2
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch * 2, 4, 2, 1),    # /4
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch * 2, ch * 4, 4, 2, 1),  # /8
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch * 4, ch * 8, 4, 2, 1),  # /16
            nn.LeakyReLU(0.2),
        )

        flat_dim = ch * 8 * (seq_len // 16)
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        # x: [B, seq_len]
        x = x.unsqueeze(1)  # [B, 1, seq_len]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """4-layer transposed-conv decoder, reconstructs a length-``seq_len`` curve."""

    def __init__(self, seq_len: int = 384, latent_dim: int = 32, ch: int = 64):
        super().__init__()
        self.init_len = seq_len // 16
        self.ch = ch

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ch * 8 * self.init_len),
            nn.LeakyReLU(0.2),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(ch * 8, ch * 4, 4, 2, 1),  # ×2
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 4, ch * 2, 4, 2, 1),  # ×4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1),      # ×8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(ch, 1, 4, 2, 1),           # ×16
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.ch * 8, self.init_len)
        x = self.conv(h)
        return x.squeeze(1)  # [B, seq_len]


class VAE(nn.Module):
    """Standard VAE with re-parameterisation trick."""

    def __init__(self, seq_len: int = 384, latent_dim: int = 32, ch: int = 64):
        super().__init__()
        self.encoder = Encoder(seq_len, latent_dim, ch)
        self.decoder = Decoder(seq_len, latent_dim, ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples: int, device):
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)


def vae_loss(x, x_recon, mu, logvar, kl_weight: float = 0.0005):
    """Standard β-VAE loss: mean MSE reconstruction + ``kl_weight`` × KL."""
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
