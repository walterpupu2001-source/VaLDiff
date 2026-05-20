"""
Battery representation loaded from a .mat file.

This module is the single source of truth for the `BatteryRaw` class. All
training and evaluation scripts in the project import it from here, so
that pickle files produced by `scripts/prepare_data.py` reference a stable
class location (``src.data.battery.BatteryRaw``).

The contents of the class — particularly capacity-loading logic, V/I/T
aggregation, and the 13-dim physics-feature definition — are copied
verbatim from the original Block-1 implementation. Refactoring is purely
structural: numerical results must be byte-identical to the original.
"""

import os

import numpy as np
import scipy.io
from scipy.stats import kurtosis, skew


# ============================================================
# Helpers
# ============================================================
def ensure_1d_float(x):
    """Return ``x`` as a 1-D ``float64`` array with NaN / non-finite values removed.

    The order of the two filters (``~isnan`` followed by ``isfinite``) is preserved
    from the original code for full byte-level reproducibility, even though
    ``isfinite`` alone would suffice.
    """
    arr = np.array(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    return arr


def detect_batch_id_from_path(path):
    """Detect batch ID (1-5) from a file path by searching for ``batch-{k}``."""
    p = path.lower()
    for k in range(1, 6):
        if f"batch-{k}" in p:
            return k
    return 0


# ============================================================
# BatteryRaw
# ============================================================
class BatteryRaw:
    """A single battery cell loaded from a ``.mat`` file.

    Attributes
    ----------
    mat_path : str
        Path the data was loaded from.
    battery_name : str
        File-name (no extension), used as a stable identifier.
    batch_id : int
        Batch ID (1-5), or 0 if not detectable from path.
    capacity_curve : np.ndarray, shape (n_cycles,)
        Discharge (or charge) capacity per cycle in Ah.
    voltages, currents, temps : np.ndarray, shape (n_cycles,)
        Per-cycle means of voltage / current / temperature, truncated to
        the length of ``capacity_curve``.
    physics_features : dict
        13 scalar physics features used as conditioning input.
    """

    def __init__(self, mat_path: str):
        self.mat_path = mat_path
        self.battery_name = os.path.splitext(os.path.basename(mat_path))[0]
        self.batch_id = detect_batch_id_from_path(mat_path)

        mat = scipy.io.loadmat(mat_path)
        cap, src = self._load_capacity(mat)
        v, i, t = self._load_vit(mat)

        self.capacity_curve = cap
        self.voltages = v[:len(cap)]
        self.currents = i[:len(cap)]
        self.temps = t[:len(cap)]

        print(f"电池: {self.battery_name} | batch={self.batch_id} | 长度={len(self.capacity_curve)}")
        self.physics_features = self._build_physics_features()

    # --------------------------------------------------------
    # .mat loading (verbatim from Block 1)
    # --------------------------------------------------------
    def _load_capacity(self, mat):
        if "summary" in mat and mat["summary"].size > 0:
            s = mat["summary"][0, 0]
            names = s.dtype.names
            if "discharge_capacity_Ah" in names:
                cap = ensure_1d_float(s["discharge_capacity_Ah"]); src = "summary.discharge"
            elif "charge_capacity_Ah" in names:
                cap = ensure_1d_float(s["charge_capacity_Ah"]); src = "summary.charge"
            else:
                cap, src = None, "summary.unknown"
            if cap is not None and cap.size > 0:
                while cap.size > 20 and abs(cap[-1]) < 1e-9:
                    cap = cap[:-1]
                return cap.astype(np.float64), src
        if "data" in mat and mat["data"].size > 0:
            data = mat["data"]; N = data.shape[1]
            caps = []
            for j in range(N):
                rec = data[0, j]
                c = ensure_1d_float(rec["capacity_Ah"])
                caps.append(float(c[-1]) if c.size else 0.0)
            caps = np.array(caps, dtype=np.float64)
            while caps.size > 20 and abs(caps[-1]) < 1e-9:
                caps = caps[:-1]
            return caps, "data"
        return np.array([], dtype=np.float64), "none"

    def _load_vit(self, mat):
        if "data" not in mat or mat["data"].size == 0:
            return np.array([]), np.array([]), np.array([])
        data = mat["data"]; N = data.shape[1]
        v_list, i_list, t_list = [], [], []
        for j in range(N):
            rec = data[0, j]
            v = ensure_1d_float(rec["voltage_V"])
            c = ensure_1d_float(rec["current_A"])
            tt = ensure_1d_float(rec["temperature_C"])
            v_list.append(v.mean() if v.size else 0.0)
            i_list.append(c.mean() if c.size else 0.0)
            t_list.append(tt.mean() if tt.size else 0.0)
        return np.array(v_list), np.array(i_list), np.array(t_list)

    # --------------------------------------------------------
    # Feature extraction (verbatim from Block 1)
    # --------------------------------------------------------
    def _build_physics_features(self):
        def safe(a):
            a = np.array(a, dtype=np.float64); a = a[~np.isnan(a)]
            return np.array([0.0]) if a.size == 0 else a
        v, c, t = safe(self.voltages), safe(self.currents), safe(self.temps)
        return {
            "voltage_mean": float(v.mean()), "voltage_std": float(v.std()),
            "voltage_min": float(v.min()), "voltage_max": float(v.max()),
            "voltage_kurtosis": float(kurtosis(v)), "voltage_skewness": float(skew(v)),
            "current_mean": float(c.mean()), "current_std": float(c.std()),
            "current_kurtosis": float(kurtosis(c)), "current_skewness": float(skew(c)),
            "temp_mean": float(t.mean()), "temp_std": float(t.std()), "temp_max": float(t.max()),
        }

    # --------------------------------------------------------
    # Public conditioning interface
    # --------------------------------------------------------
    def get_condition_numeric(self):
        """Return the 13-D conditioning vector in fixed order."""
        p = self.physics_features
        return np.array([
            p["voltage_mean"], p["voltage_std"], p["voltage_min"], p["voltage_max"],
            p["voltage_kurtosis"], p["voltage_skewness"],
            p["current_mean"], p["current_std"], p["current_kurtosis"], p["current_skewness"],
            p["temp_mean"], p["temp_std"], p["temp_max"],
        ], dtype=np.float64)

    def get_condition_onehot(self):
        """Return a 5-D one-hot encoding of ``batch_id`` (zeros if unknown)."""
        oh = np.zeros(5, dtype=np.float64)
        if 1 <= self.batch_id <= 5:
            oh[self.batch_id - 1] = 1.0
        return oh
