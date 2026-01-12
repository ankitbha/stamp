import numpy as np
from typing import Tuple

# =============================================================================
# Normalization Utils
# =============================================================================

def time_split_series(series: np.ndarray, train_frac: float, gap: int) -> Tuple[np.ndarray, np.ndarray]:
    # series: [S, T]
    S, T = series.shape
    T_train = int(train_frac * T)
    t0_val = min(T, T_train + gap)
    train = series[:, :T_train]
    val = series[:, t0_val:]
    if val.shape[1] < 2:
        raise ValueError(f"Validation split too small: train T={train.shape[1]}, val T={val.shape[1]}.")
    return train, val


def compute_normalization_stats(train_series: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    # per-sensor mean/std over time axis
    mu = train_series.mean(axis=1, keepdims=True)
    sd = train_series.std(axis=1, keepdims=True) + eps
    return mu, sd


def apply_normalization(series: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (series - mu) / sd