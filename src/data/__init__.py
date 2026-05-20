"""Data loading and battery representation."""
from src.data.battery import (
    BatteryRaw,
    ensure_1d_float,
    detect_batch_id_from_path,
)
from src.data.split import (
    gather_mat_files,
    stratified_split_by_batch,
)
from src.data.datasets import (
    CurveDataset1D,
    CurveDataset1D_NoCond,
    CurveDataset2D_ResampledDE,
    CurveDataset2D_AutoDE,
    collate_keep_meta,
    no_cond_collate,
    interp_to_len,
)

__all__ = [
    "BatteryRaw",
    "ensure_1d_float",
    "detect_batch_id_from_path",
    "gather_mat_files",
    "stratified_split_by_batch",
    "CurveDataset1D",
    "CurveDataset1D_NoCond",
    "CurveDataset2D_ResampledDE",
    "CurveDataset2D_AutoDE",
    "collate_keep_meta",
    "no_cond_collate",
    "interp_to_len",
]
