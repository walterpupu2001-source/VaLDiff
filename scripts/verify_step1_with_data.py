"""
Optional Step-1 verification against real data.

Compares the new ``src.data.battery.BatteryRaw`` against the original
Block-1 implementation on one or more real ``.mat`` files. Prints a
diff of all 13 physics features plus the capacity-curve sum/length/min/max.

Usage
-----
    # Pass file paths explicitly:
    python scripts/verify_step1_with_data.py path/to/one_battery.mat
    python scripts/verify_step1_with_data.py data/raw/Batch-1/*.mat

    # OR run with no args — auto-discovers .mat files in data/raw/Batch-1/:
    python scripts/verify_step1_with_data.py

If any value differs by more than the ``ATOL`` tolerance, the script exits
with code 1. If no .mat files are found at all (no args + empty default
directory), exits 0 with a friendly "skipped" message — so wrapper
scripts like verify_all.bat keep going instead of aborting.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.battery import BatteryRaw as BatteryRawNew

# Tolerance for "identical". Pure structural refactor → expect exact 0.
ATOL = 1e-12


# ============================================================
# Original BatteryRaw (copied from your Block 1) for comparison
# ============================================================
# We keep this here as a self-contained reference. If you have moved on
# from Block 1 already, this still works — the import order ensures the
# new class is registered under src.data.battery and this local copy lives
# in __main__, so they don't conflict.
import os
import scipy.io
from scipy.stats import kurtosis, skew


def _ensure_1d_float(x):
    arr = np.array(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[np.isfinite(arr)]
    return arr


def _detect_batch_id(path):
    p = path.lower()
    for k in range(1, 6):
        if f"batch-{k}" in p:
            return k
    return 0


class BatteryRawOriginal:
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.battery_name = os.path.splitext(os.path.basename(mat_path))[0]
        self.batch_id = _detect_batch_id(mat_path)

        mat = scipy.io.loadmat(mat_path)
        cap, _ = self._load_capacity(mat)
        v, i, t = self._load_vit(mat)

        self.capacity_curve = cap
        self.voltages = v[:len(cap)]
        self.currents = i[:len(cap)]
        self.temps = t[:len(cap)]
        self.physics_features = self._build_features()

    def _load_capacity(self, mat):
        if "summary" in mat and mat["summary"].size > 0:
            s = mat["summary"][0, 0]
            names = s.dtype.names
            if "discharge_capacity_Ah" in names:
                cap = _ensure_1d_float(s["discharge_capacity_Ah"]); src = "summary.discharge"
            elif "charge_capacity_Ah" in names:
                cap = _ensure_1d_float(s["charge_capacity_Ah"]); src = "summary.charge"
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
                c = _ensure_1d_float(rec["capacity_Ah"])
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
            v = _ensure_1d_float(rec["voltage_V"])
            c = _ensure_1d_float(rec["current_A"])
            tt = _ensure_1d_float(rec["temperature_C"])
            v_list.append(v.mean() if v.size else 0.0)
            i_list.append(c.mean() if c.size else 0.0)
            t_list.append(tt.mean() if tt.size else 0.0)
        return np.array(v_list), np.array(i_list), np.array(t_list)

    def _build_features(self):
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


# ============================================================
# Comparison
# ============================================================
def compare_one(mat_path):
    print(f"\n{'='*60}")
    print(f"Comparing: {mat_path}")
    print('='*60)

    print("\n[Original]")
    orig = BatteryRawOriginal(mat_path)
    print("\n[New]")
    new = BatteryRawNew(mat_path)

    all_ok = True

    # Capacity curve
    cap_ok = (
        len(orig.capacity_curve) == len(new.capacity_curve)
        and np.allclose(orig.capacity_curve, new.capacity_curve, atol=ATOL)
    )
    flag = "✓" if cap_ok else "✗"
    print(f"\n{flag} capacity_curve  len_old={len(orig.capacity_curve)} len_new={len(new.capacity_curve)}")
    if not cap_ok:
        diff = np.max(np.abs(orig.capacity_curve - new.capacity_curve))
        print(f"   max |diff| = {diff}")
        all_ok = False

    # Physics features
    print("\nPhysics features:")
    for key in orig.physics_features:
        v_old = orig.physics_features[key]
        v_new = new.physics_features[key]
        equal = abs(v_old - v_new) < ATOL
        flag = "✓" if equal else "✗"
        print(f"  {flag} {key:<22}  old={v_old:>14.8f}  new={v_new:>14.8f}")
        if not equal:
            all_ok = False

    # Batch ID
    bid_ok = orig.batch_id == new.batch_id
    flag = "✓" if bid_ok else "✗"
    print(f"\n{flag} batch_id  old={orig.batch_id}  new={new.batch_id}")
    if not bid_ok:
        all_ok = False

    return all_ok


if __name__ == "__main__":
    paths = sys.argv[1:]

    # If no args, auto-discover .mat files under data/raw/Batch-1/ (relative
    # to the project root, NOT cwd) — so wrapper scripts and IDE runs both
    # find them.
    if not paths:
        default_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "Batch-1"
        if default_dir.is_dir():
            paths = sorted(str(p) for p in default_dir.glob("*.mat"))
        if not paths:
            print(
                f"[skip] verify_step1_with_data.py: no .mat files found in "
                f"{default_dir}\n"
                f"        To run this verifier, either copy .mat files into\n"
                f"        data/raw/Batch-1/ or pass paths explicitly:\n"
                f"          python scripts/verify_step1_with_data.py <path>.mat"
            )
            sys.exit(0)  # 0 → wrappers can continue past this step
        print(f"[auto] verify_step1_with_data.py: discovered {len(paths)} .mat file(s) in {default_dir}")

    results = [compare_one(p) for p in paths]
    print("\n" + "=" * 60)
    if all(results):
        print(f"✓ ALL {len(paths)} FILES MATCH BIT-FOR-BIT")
        sys.exit(0)
    else:
        n_fail = sum(1 for r in results if not r)
        print(f"✗ {n_fail}/{len(paths)} FILES DIFFER - see output above")
        sys.exit(1)
