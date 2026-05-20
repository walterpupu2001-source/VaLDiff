"""
Padding-constraint helpers for the Proposed method.

The Proposed UNet2D operates on adaptive Delay-Embedding images that
have been padded out to a fixed ``n x n`` shape. The number of *valid*
columns is recorded in ``meta['orig_q']`` (variable per battery). To
prevent the model from "learning" the padded columns:

1. ``build_mask_from_metas`` creates a binary mask used by
   ``StandardDDPM.train_step(..., mask=...)`` so that MSE is only
   computed on the valid columns.

2. ``enforce_padding_constraint`` overwrites the padded columns of a
   noisy / partially-denoised image with the *last valid column*,
   preventing the diffusion process from drifting in those regions.
   It is applied to ``x_t`` before every forward pass during training,
   and after every reverse step during sampling.

Both functions are copied verbatim from the Proposed Block-2 script.
"""

from typing import List

import torch


def build_mask_from_metas(metas: List[dict], img_size: int, device) -> torch.Tensor:
    """Return a mask of shape ``[B, 1, H, W]``.

    For each sample, columns ``[0:orig_q]`` are 1 (valid), the rest 0
    (padded). ``orig_q`` is clamped to ``[1, img_size]`` for safety.
    """
    masks = []
    for meta in metas:
        q = int(meta["orig_q"])
        q = max(1, min(q, img_size))
        m = torch.zeros(1, img_size, img_size, device=device, dtype=torch.float32)
        m[:, :, :q] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0)  # [B, 1, H, W]


@torch.no_grad()
def enforce_padding_constraint(
    x: torch.Tensor, metas: List[dict], img_size: int
) -> torch.Tensor:
    """Copy the last-valid column into every padded column, in-place.

    Block-3 verbatim: uses ``meta.get('orig_q', meta.get('actual_q', img_size))``
    so the SAME constraint helper works for both adaptive-DE meta dicts
    (which have ``orig_q``) and fixed-DE meta dicts (which have ``actual_q``).

    Parameters
    ----------
    x : (B, C, H, W) tensor
    metas : list of dicts; each must contain either ``orig_q`` or ``actual_q``.
    img_size : int — the image side ``W`` (also ``H``).
    """
    if metas is None:
        return x
    B = x.shape[0]
    for b in range(B):
        meta = metas[b] if b < len(metas) else metas[0]
        q = meta.get("orig_q", meta.get("actual_q", img_size))
        q = max(1, min(int(q), img_size))
        if q < img_size:
            x[b, :, :, q:] = x[b, :, :, q - 1 : q]
    return x


def make_padding_fn(metas: List[dict], img_size: int):
    """Convenience: build a closure suitable to pass as ``padding_fn``."""
    def _apply(x: torch.Tensor) -> torch.Tensor:
        return enforce_padding_constraint(x, metas, img_size)
    return _apply
