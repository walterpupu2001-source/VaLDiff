"""
Classical augmentation methods — Block 3 verbatim (minus Scaling, which the
paper does not include).

Two methods, both using a fresh ``np.random.RandomState(BASE_SEED)``
inside the function (so the order of calls in evaluate_downstream
determines the random sequences). DO NOT alter these — any change in
RNG initialisation or formula will diverge from paper results.
"""

import numpy as np
from scipy.interpolate import CubicSpline


# Defaults from Block 3.
BASE_SEED = 2026
NUM_AUG_PER_BATTERY = 5


def aug_gaussian_noise(curves, sigma=0.02, num_aug_per_battery=NUM_AUG_PER_BATTERY, base_seed=BASE_SEED):
    """
    高斯噪声扰动 (Gaussian Noise Injection)
    参考: Fan et al., 2022 - 用于模拟容量恢复、容量骤降、传感器测量误差
    在LSTM和GRU模型上验证，可提高模型泛化能力和SOH预测准确率

    Block-3 verbatim: noise scale = (c.max() - c.min()) * sigma.
    Uses ``rng.normal(0, scale, len(c))`` (not ``rng.randn(...) * scale`` —
    these produce different sequences from the same RandomState).
    """
    rng = np.random.RandomState(base_seed)
    aug = []
    for c in curves:
        c = np.asarray(c, dtype=np.float64).ravel()
        scale = (c.max() - c.min()) * sigma  # 噪声幅度与数据范围成比例
        for _ in range(num_aug_per_battery):
            noise = rng.normal(0, scale, len(c))
            aug.append(c + noise)
    return aug


def aug_time_warping(curves, sigma=0.2, num_knots=4,
                     num_aug_per_battery=NUM_AUG_PER_BATTERY, base_seed=BASE_SEED):
    """
    时间扭曲 (Time Warping)
    参考: Kim et al., 2020 - 使用DTW将参考电池曲线弹性匹配到目标电池
    可以模拟不同电池间退化路径的差异，提升对不同退化路径的泛化能力

    Block-3 verbatim: REAL time-axis warping via cumsum of |N(1, sigma)|
    multipliers, normalised to span [0, L-1], then a CubicSpline maps
    time indices to warped times, and ``np.interp`` samples the curve.
    """
    rng = np.random.RandomState(base_seed)
    aug = []
    for c in curves:
        c = np.asarray(c, dtype=np.float64).ravel()
        L = len(c)
        for _ in range(num_aug_per_battery):
            # 生成平滑的时间扭曲曲线
            knot_xs = np.linspace(0, L - 1, num_knots + 2)
            knot_ys = np.cumsum(np.abs(rng.normal(1.0, sigma, num_knots + 2)))
            knot_ys = knot_ys / knot_ys[-1] * (L - 1)  # 归一化到[0, L-1]
            spline = CubicSpline(knot_xs, knot_ys)
            t_warped = np.clip(spline(np.arange(L)), 0, L - 1)
            # 插值得到扭曲后的曲线
            aug.append(np.interp(np.arange(L), t_warped, c))
    return aug
