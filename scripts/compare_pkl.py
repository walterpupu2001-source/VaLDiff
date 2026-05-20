"""
Byte-equivalence check between the original ``prepared_data.pkl`` and
the one freshly produced by ``scripts/prepare_data.py``.

Why this script exists
----------------------
The original Block 1 pickled the data with ``BatteryRaw`` living in
``__main__`` (because Block 1 was a top-level script). The new
``prepare_data.py`` pickles with ``BatteryRaw`` living in
``src.data.battery``.

When this comparison script runs:
- ``from src.data.battery import BatteryRaw`` makes the class importable
  via ``src.data.battery.BatteryRaw`` — that lets us load the *new* pkl.
- The bound name ``BatteryRaw`` in this script's own namespace also makes
  ``__main__.BatteryRaw`` resolvable — that lets us load the *old* pkl.

So both pickles deserialize into the same Python class, and we can do
field-by-field comparison.

Usage
-----
    python scripts/compare_pkl.py <path_to_old_pkl> <path_to_new_pkl>

For example:
    python scripts/compare_pkl.py ^
        "C:\\Users\\z5454160\\OneDrive - UNSW\\Desktop\\NewJournal\\upload_to_katana\\prepared_data.pkl" ^
        outputs\\prepared_data.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The import below has THREE jobs:
# 1) Make ``src.data.battery.BatteryRaw`` available (for new pkl).
# 2) Bind name ``BatteryRaw`` in __main__ (for old pkl, pickled from there).
# 3) Make ``ensure_1d_float`` / ``detect_batch_id_from_path`` similarly
#    available, in case they were pickled too.
from src.data.battery import BatteryRaw  # noqa: F401


# Tolerance for "identical". Pure structural refactor → expect bit-exact.
ATOL = 1e-12


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def array_equal_within(a, b, atol=ATOL):
    """True iff shapes match and values are within ``atol``."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=atol, rtol=0)


def compare_battery_pair(b_old, b_new):
    """Compare two ``BatteryRaw`` instances field-by-field.

    Returns
    -------
    (ok, list_of_diff_messages)
    """
    diffs = []

    # Identifiers
    if b_old.battery_name != b_new.battery_name:
        diffs.append(f"battery_name: '{b_old.battery_name}' vs '{b_new.battery_name}'")
    if b_old.batch_id != b_new.batch_id:
        diffs.append(f"batch_id: {b_old.batch_id} vs {b_new.batch_id}")

    # Note: mat_path is INTENTIONALLY not compared. The original pkl was
    # generated with absolute OneDrive paths; the new one with relative
    # ``data/raw/...`` paths. They naturally differ.

    # Arrays
    for attr in ("capacity_curve", "voltages", "currents", "temps"):
        o = getattr(b_old, attr)
        n = getattr(b_new, attr)
        if not array_equal_within(o, n):
            diffs.append(
                f"{attr}: shapes {np.asarray(o).shape} vs {np.asarray(n).shape}, "
                f"max |diff|={np.max(np.abs(np.asarray(o)-np.asarray(n))) if np.asarray(o).shape==np.asarray(n).shape else 'N/A'}"
            )

    # Physics features (dict of scalars)
    for k in b_old.physics_features:
        v_old = b_old.physics_features[k]
        v_new = b_new.physics_features.get(k, None)
        if v_new is None:
            diffs.append(f"physics_features.{k}: missing in new")
            continue
        if abs(v_old - v_new) > ATOL:
            diffs.append(
                f"physics_features.{k}: old={v_old:.12f} new={v_new:.12f} "
                f"diff={v_old - v_new:.2e}"
            )

    return len(diffs) == 0, diffs


def main(old_path, new_path):
    print(f"Loading old pkl: {old_path}")
    old = load_pkl(old_path)
    print(f"Loading new pkl: {new_path}")
    new = load_pkl(new_path)

    print("\n" + "=" * 60)
    print("Top-level pkl structure")
    print("=" * 60)
    print(f"  old keys: {sorted(old.keys())}")
    print(f"  new keys: {sorted(new.keys())}")

    all_ok = True

    # ---- Split sizes ----
    print("\n" + "=" * 60)
    print("Split sizes")
    print("=" * 60)
    for split in ("train_batts", "val_batts", "test_batts"):
        n_old, n_new = len(old[split]), len(new[split])
        ok = n_old == n_new
        flag = "✓" if ok else "✗"
        print(f"  {flag} {split:<12}  old={n_old:>4}   new={n_new:>4}")
        if not ok:
            all_ok = False

    # ---- Condition stats ----
    print("\n" + "=" * 60)
    print("Condition statistics")
    print("=" * 60)
    for key in ("cond_num_mean", "cond_num_std"):
        a, b = old[key], new[key]
        if not array_equal_within(a, b):
            flag = "✗"
            diff = np.max(np.abs(np.asarray(a) - np.asarray(b))) if np.asarray(a).shape == np.asarray(b).shape else float("nan")
            print(f"  {flag} {key:<14}  shape_old={np.asarray(a).shape}  shape_new={np.asarray(b).shape}  max|diff|={diff:.2e}")
            all_ok = False
        else:
            print(f"  ✓ {key:<14}  shape={np.asarray(a).shape}  matches within {ATOL}")

    # ---- Per-split membership and battery-wise content ----
    for split in ("train_batts", "val_batts", "test_batts"):
        print("\n" + "=" * 60)
        print(f"Split: {split}")
        print("=" * 60)

        if len(old[split]) != len(new[split]):
            print(f"  ⚠ Skipping content compare — lengths differ.")
            continue

        names_old = [b.battery_name for b in old[split]]
        names_new = [b.battery_name for b in new[split]]

        if set(names_old) != set(names_new):
            missing_in_new = set(names_old) - set(names_new)
            extra_in_new = set(names_new) - set(names_old)
            print(f"  ✗ Membership differs.")
            if missing_in_new:
                print(f"    Missing in new: {sorted(missing_in_new)}")
            if extra_in_new:
                print(f"    Extra in new:   {sorted(extra_in_new)}")
            all_ok = False
            continue

        if names_old != names_new:
            print(f"  ⚠ Order differs (membership identical).")
            print(f"    old order: {names_old}")
            print(f"    new order: {names_new}")
            # Don't immediately fail — compare by name instead of order.
            order_matches = False
        else:
            order_matches = True

        # Compare each battery by name lookup (handles order differences).
        new_by_name = {b.battery_name: b for b in new[split]}
        n_ok = 0
        for b_old in old[split]:
            b_new = new_by_name[b_old.battery_name]
            ok, diffs = compare_battery_pair(b_old, b_new)
            if ok:
                n_ok += 1
            else:
                print(f"  ✗ {b_old.battery_name}:")
                for d in diffs:
                    print(f"      {d}")
                all_ok = False
        if order_matches:
            print(f"  ✓ Order matches, content matches for {n_ok}/{len(old[split])} batteries")
        else:
            print(f"  ⚠ Order differs but content matches for {n_ok}/{len(old[split])} batteries")
            # Order difference is a real concern — see explanation below.
            all_ok = False

    # ---- Verdict ----
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ NEW PKL IS BIT-FOR-BIT EQUIVALENT TO OLD")
        return 0
    else:
        print("✗ DIFFERENCES FOUND — see details above.")
        print()
        print("Common explanations if you see differences:")
        print("  - Different set of .mat files in data/raw/ vs the original run.")
        print("  - The original pkl was generated with a different RNG seed.")
        print("  - Splits ORDER differs but content matches → train_batts order")
        print("    is consumed by DataLoaders downstream, so this matters.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
