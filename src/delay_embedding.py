"""
Delay Embedding (DE) for 1-D capacity trajectories.

Two modes are exposed, corresponding to the two model families in the paper.
The two modes use **different parameter-computation formulas** (one floor,
one ceil) — this is intentional and preserved from the original code.

Adaptive mode  (used by the Proposed method)
--------------------------------------------
- Input trajectory of variable length L; no resampling is performed.
- ``compute_params_adaptive`` uses ``ceil``.
- ``forward_to_square`` pads with the last column when ``q < n`` and
  zoom-shrinks when ``q > n``, producing a final ``n x n`` image.
- The variable ``orig_q`` is stored in the meta dict so the inverse and
  the Padding-Constraint mask can recover the valid columns.

Fixed mode  (used by DE-DDPM / Resampled-DE)
--------------------------------------------
- Input trajectory is pre-resampled to a constant length ``seq_len``
  (500 in the paper).
- ``compute_params_fixed`` uses **floor** division (i.e. the formula that
  motivated the choice of 500: with n=32, m=15, exactly 32 complete
  windows tile the sequence, plus one trailing edge-padded column).
- The variable ``actual_q`` is stored in the meta dict.

All low-level operations (``forward``, ``inverse``) are shared.
"""

from typing import Tuple

import numpy as np


class DelayEmbedding:
    """Static collection of DE operations (adaptive + fixed modes)."""

    # ============================================================
    # Low-level shared primitives
    # ============================================================
    @staticmethod
    def forward(x: np.ndarray, n: int, m: int) -> np.ndarray:
        """Slide an ``n``-window with hop ``m`` across ``x``.

        Returns an ``(n, q)`` matrix where columns are successive windows.
        If ``len(x) < n``, ``x`` is edge-padded to length ``n``.
        The last window is edge-padded if it would extend past ``len(x)``.

        ``q`` is computed inside as ``ceil((L - n) / m) + 1``.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        L = len(x)
        if L < n:
            x = np.pad(x, (0, n - L), mode='edge'); L = n
        q = int(np.ceil((L - n) / m)) + 1
        img = np.zeros((n, q), dtype=np.float64)
        for j in range(q):
            start = j * m
            end = start + n
            if end <= L:
                img[:, j] = x[start:end]
            else:
                valid = L - start
                img[:valid, j] = x[start:L]
                img[valid:, j] = x[-1]
        return img

    @staticmethod
    def inverse(img: np.ndarray, n: int, m: int, target_len: int) -> np.ndarray:
        """Reconstruct a 1-D trajectory from an ``(n, q)`` DE matrix.

        Each output sample is the average of all DE columns covering it.
        Used by both modes' inverse routines.
        """
        img = np.asarray(img, dtype=np.float64)
        _, q = img.shape
        max_len = n + (q - 1) * m
        x = np.zeros(max_len, dtype=np.float64)
        weights = np.zeros(max_len, dtype=np.float64)
        for j in range(q):
            start = j * m
            x[start:start+n] += img[:, j]
            weights[start:start+n] += 1.0
        x = x / np.maximum(weights, 1e-8)
        return x[:target_len]

    # ============================================================
    # Adaptive mode (Proposed) — ceil-based parameters
    # ============================================================
    @staticmethod
    def compute_params_adaptive(seq_len: int, n: int) -> Tuple[int, int]:
        """Adaptive-mode parameter computation (used by ``forward_to_square``).

        Uses ``ceil`` so that every sample is covered by at least one window.
        """
        if seq_len <= n:
            return 1, seq_len
        m = max(1, int(np.ceil((seq_len - n) / (n - 1))))
        q = int(np.ceil((seq_len - n) / m)) + 1
        return m, q

    @staticmethod
    def forward_to_square(x: np.ndarray, n: int) -> Tuple[np.ndarray, dict]:
        """Adaptive forward: DE → ``n x n`` (pad-or-shrink as needed).

        Returns
        -------
        img_square : (n, n) ndarray
        meta : dict with keys ``orig_len``, ``n``, ``m``, ``orig_q``.
            ``orig_q`` is the number of *valid* (non-padded) columns.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        L = len(x)
        m, _ = DelayEmbedding.compute_params_adaptive(L, n)
        img = DelayEmbedding.forward(x, n, m)
        actual_q = img.shape[1]

        if actual_q < n:
            img_square = np.zeros((n, n), dtype=np.float64)
            img_square[:, :actual_q] = img
            for j in range(actual_q, n):
                img_square[:, j] = img[:, -1]
        elif actual_q > n:
            from scipy.ndimage import zoom
            img_square = zoom(img, (1, n / actual_q), order=1)
            actual_q = n
        else:
            img_square = img

        meta = {'orig_len': L, 'n': n, 'm': m, 'orig_q': min(actual_q, n)}
        return img_square, meta

    @staticmethod
    def inverse_from_square(img_square: np.ndarray, meta: dict) -> np.ndarray:
        """Adaptive inverse: ``n x n`` → 1-D trajectory of length ``orig_len``."""
        n, m, orig_q, orig_len = meta['n'], meta['m'], meta['orig_q'], meta['orig_len']
        img = img_square[:, :orig_q]
        return DelayEmbedding.inverse(img, n, m, orig_len)

    # ============================================================
    # Fixed mode (Resampled-DE) — floor-based parameters
    # ============================================================
    @staticmethod
    def compute_params_fixed(seq_len: int, n: int) -> Tuple[int, int]:
        """Fixed-mode parameter computation (used by ``forward_fixed``).

        Uses ``//`` (floor); for ``seq_len=500``, ``n=32`` this yields
        ``m=15, q=32`` (exactly 32 complete windows tile the sequence).
        The underlying ``forward`` adds one trailing edge-padded column.
        """
        if seq_len <= n:
            return 1, seq_len
        m = max(1, (seq_len - n) // (n - 1))
        q = (seq_len - n) // m + 1
        return m, q

    @staticmethod
    def forward_fixed(x: np.ndarray, n: int, seq_len: int) -> Tuple[np.ndarray, dict]:
        """Fixed forward: input already resampled to ``seq_len``, DE to ``n x n``.

        Returns
        -------
        img_square : (n, n) ndarray
        meta : dict with keys ``seq_len``, ``n``, ``m``, ``actual_q``.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        assert len(x) == seq_len, f"Input length {len(x)} != seq_len {seq_len}"

        m, _ = DelayEmbedding.compute_params_fixed(seq_len, n)
        img = DelayEmbedding.forward(x, n, m)
        actual_q = img.shape[1]

        if actual_q < n:
            img_square = np.zeros((n, n), dtype=np.float64)
            img_square[:, :actual_q] = img
            for j in range(actual_q, n):
                img_square[:, j] = img[:, -1]
        elif actual_q > n:
            from scipy.ndimage import zoom
            img_square = zoom(img, (1, n / actual_q), order=1)
        else:
            img_square = img

        meta = {'seq_len': seq_len, 'n': n, 'm': m, 'actual_q': min(actual_q, n)}
        return img_square, meta

    @staticmethod
    def inverse_fixed(img_square: np.ndarray, meta: dict) -> np.ndarray:
        """Fixed inverse: ``n x n`` → 1-D trajectory of length ``seq_len``."""
        n, m, actual_q, seq_len = meta['n'], meta['m'], meta['actual_q'], meta['seq_len']
        img = img_square[:, :actual_q]
        return DelayEmbedding.inverse(img, n, m, seq_len)
