"""
Deterministic seeding.

Mirrors the original ``set_seed`` from Block 3 verbatim, including the call
order (``torch`` → ``np.random`` → ``random`` → ``torch.cuda``). Changing
the call order can change the state of downstream RNG streams, so do not
reorder.
"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for ``torch``, ``numpy``, and ``random``.

    Call order is preserved exactly from the original code. ``torch`` is
    imported lazily so this function still works in environments without
    PyTorch installed (e.g. data-prep-only).
    """
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        torch = None

    np.random.seed(seed)
    random.seed(seed)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
