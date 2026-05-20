"""
Step-1 verification tests.

These tests use hand-computed expected values (no real .mat data needed)
to verify that the foundational components produce byte-identical outputs
to the original Block-1 / Block-2 / Block-3 implementations.

Run either as:
    pytest tests/ -v
or:
    python tests/test_step1.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``src`` importable when running the file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.battery import ensure_1d_float, detect_batch_id_from_path
from src.delay_embedding import DelayEmbedding
from src.utils.seed import set_seed


# ============================================================
# ensure_1d_float
# ============================================================
class TestEnsure1dFloat:
    def test_removes_nan(self):
        x = np.array([1.0, np.nan, 2.0, 3.0])
        out = ensure_1d_float(x)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_removes_inf(self):
        x = np.array([1.0, np.inf, 2.0, -np.inf, 3.0])
        out = ensure_1d_float(x)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_flattens_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = ensure_1d_float(x)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_dtype_is_float64(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        out = ensure_1d_float(x)
        assert out.dtype == np.float64

    def test_empty_input(self):
        out = ensure_1d_float(np.array([]))
        assert out.size == 0


# ============================================================
# detect_batch_id_from_path
# ============================================================
class TestDetectBatchId:
    def test_typical_windows_path(self):
        assert detect_batch_id_from_path(r"C:\foo\Batch-1\bar.mat") == 1
        assert detect_batch_id_from_path(r"C:\foo\Batch-5\bar.mat") == 5

    def test_typical_unix_path(self):
        assert detect_batch_id_from_path("/data/Batch-3/x.mat") == 3

    def test_lowercase(self):
        assert detect_batch_id_from_path("data/batch-2/x.mat") == 2

    def test_no_batch_returns_zero(self):
        assert detect_batch_id_from_path("/some/random/file.mat") == 0

    def test_out_of_range_returns_zero(self):
        # Batch-6 is not recognised
        assert detect_batch_id_from_path("/data/Batch-6/x.mat") == 0


# ============================================================
# DelayEmbedding: low-level forward()
# ============================================================
class TestDelayEmbeddingForward:
    """Hand-computed expected outputs."""

    def test_simple_3x3_no_padding(self):
        # x = [1, 2, 3, 4, 5, 6, 7], n=3, m=2
        # q = ceil((7-3)/2) + 1 = 2 + 1 = 3
        # col 0: x[0:3] = [1, 2, 3]
        # col 1: x[2:5] = [3, 4, 5]
        # col 2: x[4:7] = [5, 6, 7]
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        img = DelayEmbedding.forward(x, n=3, m=2)
        expected = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ])
        np.testing.assert_array_equal(img, expected)

    def test_last_window_edge_padded(self):
        # x = [1, 2, 3, 4, 5, 6], n=3, m=2
        # q = ceil((6-3)/2) + 1 = ceil(1.5) + 1 = 3
        # col 0: x[0:3] = [1, 2, 3]
        # col 1: x[2:5] = [3, 4, 5]
        # col 2: start=4, end=7 > L=6 → valid=2, take x[4:6]=[5,6], then edge-pad with x[-1]=6
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        img = DelayEmbedding.forward(x, n=3, m=2)
        expected = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 6.0],  # 6 = edge-pad value
        ])
        np.testing.assert_array_equal(img, expected)

    def test_short_input_edge_padded(self):
        # x = [1, 2], n=3, m=1 → pad to length 3 with edge value → [1, 2, 2]
        # q = ceil((3-3)/1)+1 = 1, col 0 = [1, 2, 2]
        x = np.array([1.0, 2.0])
        img = DelayEmbedding.forward(x, n=3, m=1)
        expected = np.array([[1.0], [2.0], [2.0]])
        np.testing.assert_array_equal(img, expected)


# ============================================================
# DelayEmbedding: parameter computation (the SEQ_LEN=500 question!)
# ============================================================
class TestDelayEmbeddingParams:
    def test_fixed_mode_500_32(self):
        """The exact case used by Resampled-DE / DE-DDPM in the paper."""
        m, q = DelayEmbedding.compute_params_fixed(500, 32)
        assert m == 15, f"Expected m=15, got {m}"
        assert q == 32, f"Expected q=32 (planned), got {q}"

    def test_fixed_mode_short(self):
        # seq_len <= n → m=1, q=seq_len
        m, q = DelayEmbedding.compute_params_fixed(10, 32)
        assert (m, q) == (1, 10)

    def test_adaptive_mode_uses_ceil(self):
        """For L=500 with ceil-based math, m=16 (different from fixed's m=15)."""
        m, q = DelayEmbedding.compute_params_adaptive(500, 32)
        # ceil(468/31) = ceil(15.097) = 16
        # q = ceil(468/16) + 1 = ceil(29.25) + 1 = 30 + 1 = 31
        assert m == 16, f"Expected m=16 (ceil), got {m}"
        assert q == 31, f"Expected q=31, got {q}"

    def test_adaptive_vs_fixed_differ(self):
        """Crucial: the two modes give different results, on purpose."""
        m_a, _ = DelayEmbedding.compute_params_adaptive(500, 32)
        m_f, _ = DelayEmbedding.compute_params_fixed(500, 32)
        assert m_a != m_f, "Adaptive and fixed modes must differ — preserve both."


# ============================================================
# DelayEmbedding: forward → inverse round-trips
# ============================================================
class TestDelayEmbeddingRoundTrip:
    def test_fixed_round_trip_smooth_curve(self):
        """Smooth curve survives forward+inverse with bounded error.

        Note: ``inverse_fixed`` returns 497 samples (NOT 500) when seq_len=500.
        With n=32 and m=15, the embedding covers indices 0..32+31*15-1 = 0..496,
        so only 497 positions can be reconstructed. The original Block-2 code
        masks this by calling ``interp_to_len(curve, real_len)`` immediately
        after inverse_fixed during evaluation. We preserve this behaviour
        verbatim — do not "fix" it.
        """
        x = np.linspace(2.0, 1.0, 500)  # typical battery capacity decay shape
        img, meta = DelayEmbedding.forward_fixed(x, n=32, seq_len=500)

        assert img.shape == (32, 32)
        assert meta == {"seq_len": 500, "n": 32, "m": 15, "actual_q": 32}

        x_back = DelayEmbedding.inverse_fixed(img, meta)
        # Documented quirk: 497 not 500 (see docstring above).
        assert x_back.shape == (497,)
        # The DE inverse uses overlap-averaging which is approximate but
        # smoothes nicely on a monotone curve. Loose tolerance is intentional.
        assert np.max(np.abs(x_back - x[:497])) < 0.05

    def test_adaptive_round_trip_smooth_curve(self):
        x = np.linspace(2.0, 1.0, 600)  # arbitrary cycle count
        img, meta = DelayEmbedding.forward_to_square(x, n=32)

        assert img.shape == (32, 32)
        assert meta["orig_len"] == 600
        assert meta["n"] == 32
        assert "orig_q" in meta and "m" in meta

        x_back = DelayEmbedding.inverse_from_square(img, meta)
        assert x_back.shape == (600,)
        assert np.max(np.abs(x_back - x)) < 0.05

    def test_adaptive_short_curve_pads_to_32(self):
        """Adaptive mode on a short curve pads to n x n with edge-repeat."""
        x = np.linspace(2.0, 1.0, 50)
        img, meta = DelayEmbedding.forward_to_square(x, n=32)
        assert img.shape == (32, 32)
        assert meta["orig_q"] < 32  # some columns are padding

    def test_inverse_dtype(self):
        x = np.linspace(2.0, 1.0, 500)
        img, meta = DelayEmbedding.forward_fixed(x, n=32, seq_len=500)
        x_back = DelayEmbedding.inverse_fixed(img, meta)
        assert x_back.dtype == np.float64


# ============================================================
# set_seed
# ============================================================
class TestSetSeed:
    def test_numpy_reproducible(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible(self):
        import random
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_torch_reproducible_if_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)


# ============================================================
# Allow ``python tests/test_step1.py`` execution
# ============================================================
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
