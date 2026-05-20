"""Downstream-evaluation utilities (predictors, datasets, scaler, metrics, runner)."""
from src.evaluation.datasets import (
    EarlyToLateDataset,
    FullCurveDataset,
    SOHCurveDataset,
    TrendPredictionDataset,
    ensure_1d_float,
    interp_to_len,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.predictors import (
    CNNPredictor,
    InformerPredictor,
    LSTMPredictor,
    MLPPredictor,
    PREDICTOR_REGISTRY,
    PatchTSTPredictor,
    ProbSparseAttention,
    RNNPredictor,
    TransformerPredictor,
    iTransformerPredictor,
)
from src.evaluation.runner import train_eval_loop
from src.evaluation.scaler import (
    Q_NOMINAL_AH,
    SOHScaler,
    capacity_list_to_soh,
    capacity_to_soh_curve,
)

__all__ = [
    "compute_metrics",
    "SOHScaler",
    "Q_NOMINAL_AH",
    "capacity_to_soh_curve",
    "capacity_list_to_soh",
    "SOHCurveDataset",
    "EarlyToLateDataset",
    "TrendPredictionDataset",
    "FullCurveDataset",
    "ensure_1d_float",
    "interp_to_len",
    "train_eval_loop",
    "PREDICTOR_REGISTRY",
    "MLPPredictor",
    "CNNPredictor",
    "RNNPredictor",
    "LSTMPredictor",
    "TransformerPredictor",
    "PatchTSTPredictor",
    "InformerPredictor",
    "ProbSparseAttention",
    "iTransformerPredictor",
]
