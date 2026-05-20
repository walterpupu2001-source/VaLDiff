"""
SOH (State of Health) Min-Max scaler with buffer — Block 3 verbatim.

Fit ONLY on training data; adds a buffer (default 0.05) below min and
above max so val/test curves slightly outside the train range still
produce well-behaved normalized values. All splits use the SAME scaler;
metrics are inverse-transformed back to raw SOH before being computed.
"""

import numpy as np


# Constants from Block 3.
Q_NOMINAL_AH = 2.0
SOH_EPS = 1e-8


def ensure_1d_float(x):
    """Drop NaN/Inf — Block 3 util reused by SOHScaler.fit."""
    arr = np.array(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    return arr


def capacity_to_soh_curve(capacity_curve, q_nominal=Q_NOMINAL_AH):
    """
    Convert a capacity trajectory in Ah to a capacity-based SOH trajectory.

    SOH is represented as a fraction: SOH(k) = Q(k) / Q_nominal.
    For the XJTU dataset, Q_nominal = 2.0 Ah.
    """
    q = ensure_1d_float(capacity_curve)
    if q_nominal <= 0:
        raise ValueError("q_nominal must be positive.")
    return (q / (q_nominal + SOH_EPS)).astype(np.float64)


def capacity_list_to_soh(curves, q_nominal=Q_NOMINAL_AH):
    """Convert a list of capacity trajectories in Ah to SOH trajectories."""
    return [capacity_to_soh_curve(c, q_nominal=q_nominal) for c in curves]


class SOHScaler:
    """
    Min-Max normalization for SOH trajectories, mapping to [0, 1].
    Verbatim copy of Block 3's SOHScaler (attribute names included).

    - Fit ONLY on training curves to avoid data leakage.
    - Adds a small buffer (default 0.05) below min and above max so that
      val/test curves slightly outside the train range still produce
      well-behaved normalized values.
    - All methods share the same scaler so that comparisons remain fair.
    - Metrics are computed AFTER inverse-normalization, so MAE/RMSE/
      MAPE/R^2 keep their physical meaning on the original SOH scale.
    """
    def __init__(self, buffer=0.05, eps=1e-8):
        self.min_val = None       # buffered lower bound
        self.max_val = None       # buffered upper bound
        self.raw_min = None       # raw train min (for inspection)
        self.raw_max = None       # raw train max (for inspection)
        self.buffer = float(buffer)
        self.eps = float(eps)
        self.fitted = False

    def fit(self, curves):
        all_vals = np.concatenate([ensure_1d_float(c) for c in curves])
        self.raw_min = float(all_vals.min())
        self.raw_max = float(all_vals.max())
        self.min_val = self.raw_min - self.buffer
        self.max_val = self.raw_max + self.buffer
        self.fitted = True
        return self

    def transform(self, x):
        assert self.fitted, "SOHScaler not fitted yet."
        x = np.asarray(x, dtype=np.float64)
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)

    def inverse_transform(self, x):
        assert self.fitted, "SOHScaler not fitted yet."
        x = np.asarray(x, dtype=np.float64)
        return x * (self.max_val - self.min_val + self.eps) + self.min_val
