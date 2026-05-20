"""Data-augmentation methods: classical (perturbation-based) + generative (model-based)."""
from src.augmentation.classical import (
    aug_gaussian_noise,
    aug_time_warping,
)

__all__ = [
    "aug_gaussian_noise",
    "aug_time_warping",
]
