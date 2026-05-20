"""
Data preparation (Block 1 of the original notebook).

Walks ``data/raw/Batch-{1..5}/``, loads each ``.mat`` file into a
``BatteryRaw``, splits the resulting list into train / val / test
stratified by batch, computes the conditioning vector mean & std on the
training split, and pickles everything to ``outputs/prepared_data.pkl``.

Behaviour is byte-identical to the original Block 1:
- Same RNG seed (42) and call ordering (``random.seed`` before
  ``np.random.seed`` — preserved from the original).
- Same split ratios (0.7 / 0.15 / 0.15) and per-batch shuffle.
- Same conditioning-stat formula (mean over train, std + 1e-8).
- Same pickle key names so existing Block-2/Block-3 scripts keep working.

Usage:
    cd <project root>
    python scripts/prepare_data.py
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np

# Force UTF-8 stdout/stderr so Chinese print messages don't crash on Windows
# when this script is run with captured output (e.g. via subprocess or piping).
# Harmless when stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Make ``src`` importable when running this file as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.battery import BatteryRaw
from src.data.split import gather_mat_files, stratified_split_by_batch
from src.utils.config import load_config


def main(config_path: str = "config/default.yaml"):
    cfg = load_config(config_path)

    # Reproducibility — preserve Block-1 call order exactly.
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("BLOCK 1: 数据准备")
    print("=" * 60)

    # Resolve batch directories relative to the current working directory,
    # so users can run from anywhere as long as the cwd contains data/raw/.
    cwd = Path.cwd()
    data_root = cwd / cfg["paths"]["data_root"]
    base_dirs = [str(data_root / b) for b in cfg["batches"]]

    print(f"\n数据根目录: {data_root}")
    for d in base_dirs:
        marker = "✓" if Path(d).is_dir() else "✗"
        print(f"  {marker} {d}")

    mat_paths = gather_mat_files(base_dirs)
    if not mat_paths:
        raise SystemExit("未找到 .mat 文件 — 请把 .mat 文件放到 data/raw/Batch-*/ 下")

    print(f"\n共找到 .mat 文件: {len(mat_paths)}")
    all_batts = [BatteryRaw(p) for p in mat_paths]

    ratios = tuple(cfg["split"]["ratios"])
    train_batts, val_batts, test_batts = stratified_split_by_batch(
        all_batts, ratios=ratios, seed=seed,
    )

    print(f"\n训练集: {len(train_batts)}, 验证集: {len(val_batts)}, 测试集: {len(test_batts)}")

    # Conditioning stats from training set (verbatim).
    cond_num_train = np.stack(
        [b.get_condition_numeric() for b in train_batts], axis=0,
    )
    cond_num_mean = cond_num_train.mean(axis=0)
    cond_num_std = cond_num_train.std(axis=0) + 1e-8

    data_to_save = {
        "train_batts": train_batts,
        "val_batts": val_batts,
        "test_batts": test_batts,
        "cond_num_mean": cond_num_mean,
        "cond_num_std": cond_num_std,
    }

    save_path = cwd / cfg["paths"]["prepared_pkl"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"\n✓ 数据已保存到: {save_path}")


if __name__ == "__main__":
    main()
