"""
1-D Vanilla GAN (unconditional baseline).

Verbatim port of the Block-2 Vanilla-GAN classes. Class names are
suffixed with ``GAN`` so they don't collide with the VAE's ``Decoder``
when both are imported into the same script (e.g. Block 3).

Notes for users loading old checkpoints:
- The old checkpoint dict has keys ``"generator"`` and ``"discriminator"``,
  produced by ``torch.save({"generator": G.state_dict(), ...})``.
- The *contents* of those state_dicts (parameter names) are unchanged,
  so ``G.load_state_dict(ckpt["generator"])`` works without modification.
"""

import torch
import torch.nn as nn


class GeneratorGAN(nn.Module):
    """Unconditional 1-D generator.

    Maps a latent noise vector of size ``latent_dim`` to a sequence of
    length ``seq_len``. The initial spatial dimension is ``seq_len // 16``;
    four ``ConvTranspose1d`` blocks then upsample by ×2 each (16× total).
    """

    def __init__(self, latent_dim: int = 128, seq_len: int = 384, ch: int = 64):
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


class DiscriminatorGAN(nn.Module):
    """Unconditional 1-D discriminator with spectral-norm constraints.

    Spectral norm is applied to every weight matrix to stabilise training
    (no clipping, no gradient penalty needed). Output is a sigmoid
    probability in ``[0, 1]``.
    """

    def __init__(self, seq_len: int = 384, ch: int = 64):
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
        x = x.unsqueeze(1)  # [B, seq] → [B, 1, seq]
        x = self.conv(x)
        return self.fc(x).squeeze(1)
