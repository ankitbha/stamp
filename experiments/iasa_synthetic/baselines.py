"""Grouping rules and estimators that the synthetic study compares with IASA."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _components(J: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(J))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    out: dict[int, list[int]] = {}
    for j in range(J):
        out.setdefault(find(j), []).append(j)
    return sorted((sorted(g) for g in out.values()), key=lambda g: g[0])


def pairwise_coherence_grouping(A: np.ndarray, tau_rho: float = 0.99) -> list[list[int]]:
    """Connected components of the graph joining columns with |cos| > tau_rho."""
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    C = np.abs((A / norms).T @ (A / norms))
    J = A.shape[1]
    edges = [(i, j) for i in range(J) for j in range(i + 1, J) if C[i, j] > tau_rho]
    return _components(J, edges)


def bkw_grouping(A: np.ndarray, condition_index: float = 30.0, proportion: float = 0.5) -> list[list[int]]:
    """Belsley-Kuh-Welsch collinearity sets, merged into a partition.

    Columns are scaled to unit length.  For every singular value with
    condition index sigma_1 / sigma_k above the threshold, the columns whose
    variance-decomposition proportion on that singular value exceeds
    ``proportion`` form a near-dependency; overlapping sets are merged.
    """
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    X = A / norms
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    J = X.shape[1]
    S = np.maximum(S, S[0] * 1e-15)
    phi = (Vt.T ** 2) / (S[None, :] ** 2)
    prop = phi / phi.sum(axis=1, keepdims=True)
    edges: list[tuple[int, int]] = []
    for k in range(len(S)):
        if S[0] / S[k] <= condition_index:
            continue
        members = [j for j in range(J) if prop[j, k] > proportion]
        edges.extend((members[0], m) for m in members[1:])
    return _components(J, edges)


def fit_nnls(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    c, _ = nnls(A, y, maxiter=50 * A.shape[1] + 1000)
    return c


def fit_ridge(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    J = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(J), A.T @ y)


def ridge_gcv(A: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Ridge with lambda chosen by generalized cross-validation."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    uy = U.T @ y
    N = A.shape[0]
    grid = np.logspace(-8, 2, 41) * S[0] ** 2 if grid is None else grid
    best = (np.inf, grid[0])
    rss_out = float(y @ y - uy @ uy)
    for lam in grid:
        f = S**2 / (S**2 + lam)
        rss = float(np.sum(((1 - f) * uy) ** 2)) + rss_out
        gcv = rss / (N - f.sum()) ** 2
        if gcv < best[0]:
            best = (gcv, lam)
    lam = best[1]
    return fit_ridge(A, y, lam), float(lam)


def fit_tsvd(A: np.ndarray, y: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Truncated-SVD estimate and the retained right singular vectors."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > threshold
    c = Vt[keep].T @ ((U[:, keep].T @ y) / S[keep])
    return c, Vt[keep]


def tsvd_gcv(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Truncated SVD with the truncation rank chosen by generalized cross-validation."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    uy = U.T @ y
    N = A.shape[0]
    rss_out = float(y @ y - uy @ uy)
    best_k, best = 1, np.inf
    for k in range(1, len(S) + 1):
        if N - k <= 0:
            break
        g = (float(np.sum(uy[k:] ** 2)) + rss_out) / (N - k) ** 2
        if g < best:
            best_k, best = k, g
    c = Vt[:best_k].T @ (uy[:best_k] / S[:best_k])
    return c, Vt[:best_k], best_k


def gaussian_posterior(
    A: np.ndarray, y: np.ndarray, sigma: float, prior_mean: np.ndarray, prior_sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance under c ~ N(prior_mean, prior_sd^2 I), eps ~ N(0, sigma^2 I)."""
    J = A.shape[1]
    precision = A.T @ A / sigma**2 + np.eye(J) / prior_sd**2
    cov = np.linalg.inv(precision)
    mean = cov @ (A.T @ y / sigma**2 + prior_mean / prior_sd**2)
    return mean, cov


def residual_bootstrap_nnls(
    rng: np.random.Generator, A: np.ndarray, y: np.ndarray, n_boot: int = 200
) -> np.ndarray:
    """NNLS refits on fitted values plus resampled centered residuals."""
    c = fit_nnls(A, y)
    fitted = A @ c
    r = y - fitted
    r = r - r.mean()
    out = np.empty((n_boot, A.shape[1]))
    for b in range(n_boot):
        out[b] = fit_nnls(A, fitted + rng.choice(r, size=r.shape[0], replace=True))
    return out
