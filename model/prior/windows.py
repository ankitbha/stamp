import numpy as np
from typing import Tuple

# =============================================================================
# Event-biased window sampling
# =============================================================================

def compute_activity_scores(series: np.ndarray, window_K: int) -> np.ndarray:
    """
    Activity score a(t) for each valid start t in [0, T-(K+1)].
    Using mean absolute derivative over the window:
      a(t) = mean_{i, tau in window} |d_i(tau+1) - d_i(tau)|
    """
    S, T = series.shape
    K = int(window_K)
    max_start = T - (K + 1)
    if max_start < 0:
        raise ValueError(f"Series too short for K={K}: T={T}")

    diff = np.abs(np.diff(series, axis=1))  # [S, T-1]

    # window sum over time for each start: use cumulative sum
    csum = np.cumsum(diff, axis=1)  # [S, T-1]
    # sum over [t, t+K-1] in diff space (length K)
    scores = np.zeros((max_start + 1,), dtype=np.float64)
    for t in range(max_start + 1):
        t0 = t
        t1 = t + K  # exclusive in csum indexing
        # sum_i (csum[i,t1-1] - csum[i,t0-1])
        if t0 == 0:
            wsum = csum[:, t1 - 1].sum()
        else:
            wsum = (csum[:, t1 - 1] - csum[:, t0 - 1]).sum()
        scores[t] = wsum / (S * K + 1e-12)

    return scores.astype(np.float64)


def sample_window_starts_event_biased(
    activity_scores: np.ndarray,
    num_windows: int,
    gamma: float,
    eps: float,
    uniform_mix_eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample window starts with p(t) ∝ a(t)^gamma + eps, mixed with uniform with weight eta.
    """
    a = np.asarray(activity_scores, dtype=np.float64)
    a = np.maximum(a, 0.0)
    p = (a ** float(gamma)) + float(eps)
    p = p / (p.sum() + 1e-18)

    eta = float(uniform_mix_eta)
    if eta > 0.0:
        u = np.ones_like(p) / float(len(p))
        p = (1.0 - eta) * p + eta * u

    starts = rng.choice(len(p), size=int(num_windows), replace=True, p=p).astype(np.int64)
    return starts


def make_windows_from_starts(series: np.ndarray, starts: np.ndarray, window_K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    series: [S, T]
    starts: [B] start indices
    Returns:
      X: [B, S, K]   (inputs)
      Y: [B, S, K]   (targets next-step)
    """
    S, T = series.shape
    K = int(window_K)
    B = int(len(starts))
    X = np.empty((B, S, K), dtype=np.float32)
    Y = np.empty((B, S, K), dtype=np.float32)
    for b, t0 in enumerate(starts):
        t0 = int(t0)
        X[b] = series[:, t0 : t0 + K]
        Y[b] = series[:, t0 + 1 : t0 + K + 1]
    return X, Y