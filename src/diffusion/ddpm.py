"""
Standard DDPM (denoising diffusion probabilistic model).

Mathematically equivalent to the implementations in Block-2 1D-DDPM /
Resampled-DE / Proposed scripts. Differences in style:

- Uses ``while a_bar.dim() < x0.dim(): a_bar.unsqueeze_(-1)`` for dim
  alignment (same approach as Resampled-DE / Proposed; the 1D-only
  version originally hard-coded ``[:, None, None]`` — numerically
  identical for 1D inputs because the broadcast shape ends up the same).
- ``train_step`` accepts optional ``mask`` and ``padding_fn`` so the
  same class serves both the unmasked 1D-DDPM/Resampled-DE pipelines
  and the Proposed pipeline that needs masked-MSE + padding constraint.
- ``sample`` is generic over input shape; callers supply the shape.

When ``mask`` and ``padding_fn`` are both ``None`` (the default), the
behaviour reduces to the original 1D-DDPM ``train_step`` byte-for-byte.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 1e-2):
    """Linear β schedule, identical to the original implementations."""
    return torch.linspace(beta_start, beta_end, T)


class StandardDDPM:
    """Training- and sampling-side helper around an ε-prediction network."""

    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        device: str = "cpu",
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
    ):
        self.model = model
        self.T = T
        self.device = device

        betas_cpu = linear_beta_schedule(T, beta_start, beta_end)
        self.betas = betas_cpu.to(device)
        # Note: ``self.alphas = (1.0 - betas_cpu).to(device)`` matches the
        # 1D-DDPM original. Computing ``1 - self.betas`` on-device would
        # be numerically identical but we preserve the original line.
        self.alphas = (1.0 - betas_cpu).to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_var = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """Draw x_t ~ q(x_t | x_0) for a batch of (x0, t)."""
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bar[t]
        while a_bar.dim() < x0.dim():
            a_bar = a_bar.unsqueeze(-1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self,
        x0,
        cond,
        *,
        mask: Optional[torch.Tensor] = None,
        padding_fn: Optional[Callable] = None,
    ):
        """Single training step.

        Parameters
        ----------
        x0, cond : tensors
            Clean input and condition vector.
        mask : optional, same shape as x0
            If given, apply masked MSE (sum(mask * sq_err) / sum(mask)).
            Used by Proposed for the "ignore padding columns" trick.
        padding_fn : optional callable(x_t) -> x_t
            If given, apply to ``x_t`` before passing to the network.
            Used by Proposed for the padding-constraint trick.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        if padding_fn is not None:
            x_t = padding_fn(x_t)

        t_in = t.float() / (self.T - 1)
        eps_pred = self.model(x_t, t_in, cond)

        if mask is not None:
            return ((eps_pred - noise) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        return F.mse_loss(eps_pred, noise)

    # ------------------------------------------------------------------
    # Reverse process (sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond):
        """Single reverse-process step."""
        t_in = t.float() / (self.T - 1)
        eps_pred = self.model(x, t_in, cond)

        a = self.alphas[t]
        a_bar = self.alpha_bar[t]
        beta = self.betas[t]
        while a.dim() < x.dim():
            a = a.unsqueeze(-1)
            a_bar = a_bar.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        x0_pred = (x - torch.sqrt(1.0 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if t_index > 0:
            a_bar_prev = self.alpha_bar_prev[t]
            while a_bar_prev.dim() < x.dim():
                a_bar_prev = a_bar_prev.unsqueeze(-1)
            coef_x0 = beta * torch.sqrt(a_bar_prev) / (1.0 - a_bar)
            coef_xt = torch.sqrt(a) * (1.0 - a_bar_prev) / (1.0 - a_bar)
            mean = coef_x0 * x0_pred + coef_xt * x
            var = self.posterior_var[t]
            while var.dim() < x.dim():
                var = var.unsqueeze(-1)
            return mean + torch.sqrt(var) * torch.randn_like(x)
        return x0_pred

    @torch.no_grad()
    def sample(
        self,
        cond,
        shape,
        *,
        padding_fn: Optional[Callable] = None,
    ):
        """Generate a batch by full reverse diffusion.

        Parameters
        ----------
        cond : (B, cond_dim) tensor
        shape : tuple
            Sample shape excluding batch, e.g. ``(1, seq_len)`` for 1D
            or ``(1, H, W)`` for 2D.
        padding_fn : optional, called after each reverse step
            For Proposed's per-step padding constraint.
        """
        self.model.eval()
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, *shape, device=self.device)
        if padding_fn is not None:
            x = padding_fn(x)
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, i, cond)
            if padding_fn is not None:
                x = padding_fn(x)
        return x
