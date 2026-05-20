"""
Step-2 verification tests.

Covers:
- ``gather_mat_files``: ordering, deduplication, missing dirs
- ``stratified_split_by_batch``: edge cases (n=1, n=2, n>=3) and seeded determinism
- End-to-end smoke test: synthetic .mat files → prepare_data.py → loadable pkl

Run:
    pytest tests/ -v
"""

import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import gather_mat_files, stratified_split_by_batch


# ============================================================
# gather_mat_files
# ============================================================
class TestGatherMatFiles:
    def test_empty_dirs(self):
        assert gather_mat_files([]) == []

    def test_missing_dir_silently_skipped(self, tmp_path):
        # No files exist anywhere
        assert gather_mat_files([str(tmp_path / "nope")]) == []

    def test_picks_up_mat_files(self, tmp_path):
        (tmp_path / "a.mat").touch()
        (tmp_path / "b.MAT").touch()         # uppercase ext also accepted
        (tmp_path / "c.txt").touch()         # ignored
        paths = gather_mat_files([str(tmp_path)])
        names = [os.path.basename(p) for p in paths]
        assert sorted(names) == ["a.mat", "b.MAT"]

    def test_dedup_keeps_newer_mtime(self, tmp_path):
        d1 = tmp_path / "d1"; d1.mkdir()
        d2 = tmp_path / "d2"; d2.mkdir()
        p1 = d1 / "battery.mat"; p1.touch()
        time.sleep(0.05)  # ensure mtime is later
        p2 = d2 / "battery.mat"; p2.touch()
        paths = gather_mat_files([str(d1), str(d2)])
        assert len(paths) == 1
        # Whichever path won, mtime must be newer than the loser.
        assert os.path.getmtime(paths[0]) == max(os.path.getmtime(p1), os.path.getmtime(p2))

    def test_final_order_is_alphabetical_by_path(self, tmp_path):
        d1 = tmp_path / "Batch-2"; d1.mkdir()
        d2 = tmp_path / "Batch-1"; d2.mkdir()
        (d1 / "z.mat").touch()
        (d2 / "a.mat").touch()
        paths = gather_mat_files([str(d1), str(d2)])
        # Sorted alphabetically by full path → Batch-1/a.mat before Batch-2/z.mat
        assert "Batch-1" in paths[0] and "a.mat" in paths[0]
        assert "Batch-2" in paths[1] and "z.mat" in paths[1]


# ============================================================
# stratified_split_by_batch
# ============================================================
def _make_fake_batt(name, batch_id):
    """Minimal stand-in for BatteryRaw — only batch_id is used by the split."""
    return SimpleNamespace(battery_name=name, batch_id=batch_id)


class TestStratifiedSplit:
    def test_n_one_goes_to_train(self):
        batts = [_make_fake_batt("a", 1)]
        tr, va, te = stratified_split_by_batch(batts)
        assert len(tr) == 1 and len(va) == 0 and len(te) == 0

    def test_n_two_split_1_1_0(self):
        batts = [_make_fake_batt(f"b{i}", 1) for i in range(2)]
        tr, va, te = stratified_split_by_batch(batts)
        assert len(tr) == 1 and len(va) == 1 and len(te) == 0

    def test_n_three_split_1_1_1(self):
        batts = [_make_fake_batt(f"b{i}", 1) for i in range(3)]
        tr, va, te = stratified_split_by_batch(batts)
        assert len(tr) >= 1 and len(va) >= 1 and len(te) >= 1
        assert len(tr) + len(va) + len(te) == 3

    def test_split_sums_to_total(self):
        batts = [_make_fake_batt(f"b{i}", (i % 3) + 1) for i in range(20)]
        tr, va, te = stratified_split_by_batch(batts)
        assert len(tr) + len(va) + len(te) == 20

    def test_seeded_determinism(self):
        batts1 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        batts2 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        a = stratified_split_by_batch(batts1, seed=42)
        b = stratified_split_by_batch(batts2, seed=42)
        for la, lb in zip(a, b):
            assert [x.battery_name for x in la] == [x.battery_name for x in lb]

    def test_different_seeds_give_different_orders(self):
        batts1 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        batts2 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        a = stratified_split_by_batch(batts1, seed=42)
        b = stratified_split_by_batch(batts2, seed=7)
        # Train sets should differ in order for different seeds.
        assert [x.battery_name for x in a[0]] != [x.battery_name for x in b[0]]

    def test_module_random_state_does_not_affect_split(self):
        """The split uses its own RNG — global random.seed must not influence it."""
        batts1 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        batts2 = [_make_fake_batt(f"b{i}", 1) for i in range(10)]
        random.seed(1)
        a = stratified_split_by_batch(batts1, seed=42)
        random.seed(9999)
        b = stratified_split_by_batch(batts2, seed=42)
        for la, lb in zip(a, b):
            assert [x.battery_name for x in la] == [x.battery_name for x in lb]


# ============================================================
# End-to-end smoke test
# ============================================================
def _make_synthetic_mat(path: Path, n_cycles: int = 100):
    """Create a synthetic .mat file with the same structure scipy.loadmat would
    read from your real data: a ``data`` field with per-cycle V/I/T/cap records.
    """
    # Construct one structured record per cycle.
    records = []
    for k in range(n_cycles):
        rec = (
            np.linspace(3.0, 4.2, 50).reshape(-1, 1),       # voltage_V
            np.full((50, 1), -1.0),                          # current_A
            np.linspace(25.0, 26.0, 50).reshape(-1, 1),     # temperature_C
            np.array([[2.0 - 0.001 * k]]),                   # capacity_Ah (vector)
        )
        records.append(rec)

    arr = np.empty((1, n_cycles), dtype=[
        ("voltage_V", "O"),
        ("current_A", "O"),
        ("temperature_C", "O"),
        ("capacity_Ah", "O"),
    ])
    for j, (v, i, t, c) in enumerate(records):
        arr[0, j]["voltage_V"] = v
        arr[0, j]["current_A"] = i
        arr[0, j]["temperature_C"] = t
        arr[0, j]["capacity_Ah"] = c

    scipy.io.savemat(str(path), {"data": arr})


class TestPrepareDataEndToEnd:
    """Run ``scripts/prepare_data.py`` against synthetic data and check the
    resulting pkl loads cleanly and has the expected structure.
    """

    def test_full_pipeline_on_synthetic_data(self, tmp_path):
        project_root = Path(__file__).resolve().parents[1]

        # Build a fake project rooted at tmp_path, pointing back at real src/
        fake_root = tmp_path / "fake_project"
        (fake_root / "config").mkdir(parents=True)
        (fake_root / "outputs").mkdir(parents=True)
        (fake_root / "data" / "raw").mkdir(parents=True)

        # Symlink src and scripts so the script imports the real code.
        # (On Windows, os.symlink may need admin — fall back to copying.)
        for sub in ("src", "scripts"):
            src = project_root / sub
            dst = fake_root / sub
            try:
                os.symlink(src, dst, target_is_directory=True)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copytree(src, dst)

        # Synthetic batches with batch-specific names so dedup doesn't collapse them.
        for k in range(1, 5):
            (fake_root / "data" / "raw" / f"Batch-{k}").mkdir()
        for k in range(4):
            _make_synthetic_mat(fake_root / "data" / "raw" / "Batch-1" / f"b1_cell-{k}.mat", n_cycles=80)
        for k in range(3):
            _make_synthetic_mat(fake_root / "data" / "raw" / "Batch-2" / f"b2_cell-{k}.mat", n_cycles=80)

        # Minimal config pointing at the synthetic data.
        config_text = (
            "seed: 42\n"
            "paths:\n"
            "  data_root: data/raw\n"
            "  outputs: outputs\n"
            "  prepared_pkl: outputs/prepared_data.pkl\n"
            "split:\n"
            "  ratios: [0.7, 0.15, 0.15]\n"
            "batches: ['Batch-1', 'Batch-2', 'Batch-3', 'Batch-4']\n"
            "soh:\n"
            "  q_nominal_ah: 2.0\n"
        )
        (fake_root / "config" / "default.yaml").write_text(config_text)

        # Run prepare_data.py inside the fake project.
        # ``encoding="utf-8"`` ensures Chinese output decodes correctly on
        # Windows where the default subprocess encoding is cp1252.
        result = subprocess.run(
            [sys.executable, str(fake_root / "scripts" / "prepare_data.py")],
            cwd=str(fake_root),
            capture_output=True, text=True, timeout=120,
            encoding="utf-8",
        )
        assert result.returncode == 0, (
            f"prepare_data.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Inspect the produced pkl.
        pkl_path = fake_root / "outputs" / "prepared_data.pkl"
        assert pkl_path.exists()

        # IMPORTANT: load with src.data.battery.BatteryRaw importable in __main__.
        from src.data.battery import BatteryRaw  # noqa: F401
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        assert set(data.keys()) == {
            "train_batts", "val_batts", "test_batts",
            "cond_num_mean", "cond_num_std",
        }
        assert data["cond_num_mean"].shape == (13,)
        assert data["cond_num_std"].shape == (13,)
        total = len(data["train_batts"]) + len(data["val_batts"]) + len(data["test_batts"])
        assert total == 7  # 4 + 3 synthetic cells


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
