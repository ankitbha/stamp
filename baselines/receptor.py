"""Baseline source-apportionment methods for comparison against IASA.

These are deliberately *identifiability-blind* reference methods used to show what
IASA's identifiability layer adds. Each returns a confident point-estimate split
with no visibility/coherence gating, no conservative grouping, and no merge/flag,
which IASA replaces with an identifiable-resolution report on the same data.

    B1 ``plain_nnls`` -- IASA's own projected system (``H_tilde``, ``Y_tilde``),
        solved by plain nonnegative least squares, reported with a naive
        covariance CI. This is the pure identifiability-layer ablation: same fit
        as IASA, minus the diagnostics/grouping.
    B2 ``pmf_nmf`` -- PMF-style receptor model (nonnegative matrix factorization,
        the standard PMF surrogate) on the sensor x time observation matrix, with
        factors assigned to inventory groups by best-case spatial correlation. It
        uses neither the transport operator nor the inventories in the fit.
    B3 ``cmb_nnls`` -- Chemical-mass-balance-style NNLS on the *raw* (unprojected)
        response ``H_lag`` with no background projection ``P_Q^perp``, so
        source-like background variation leaks directly into source coefficients.

All three are kept charitable (well-tuned, best-case factor assignment) so the
contrast reads as "even a fair baseline cannot see the non-identifiability."
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls


def _as_2d(A) -> np.ndarray:
    return np.asarray(A, dtype=np.float64).reshape(np.asarray(A).shape[0], -1)


def _nnls_with_mask(A: np.ndarray, y: np.ndarray,
                    fixed_zero_indices: Sequence[int] = ()) -> tuple[np.ndarray, list[int]]:
    """Nonnegative least squares that honors a fixed-zero column mask (as IASA does)."""
    J = A.shape[1]
    drop = set(int(j) for j in fixed_zero_indices)
    keep = [j for j in range(J) if j not in drop]
    c = np.zeros(J, dtype=np.float64)
    if keep:
        ck, _ = nnls(A[:, keep], y)
        c[keep] = ck
    return c, keep


def _naive_covariance_std(A: np.ndarray, c: np.ndarray, keep: Sequence[int],
                          residual: np.ndarray, sigma: float | None,
                          active_tol: float = 1e-8) -> np.ndarray:
    """Plug-in frequentist CI a practitioner would report: sigma^2 (A_S^T A_S)^-1
    on the active (positive) set, with sigma estimated from the residual if not
    supplied. Deliberately ignores visibility/coherence -- that is the point."""
    J = A.shape[1]
    coef_std = np.zeros(J, dtype=np.float64)
    active = [j for j in keep if c[j] > active_tol]
    if not active:
        return coef_std
    Aa = A[:, active]
    N = A.shape[0]
    dof = max(N - len(active), 1)
    sig2 = (sigma ** 2) if sigma is not None else float(residual @ residual) / dof
    gram = Aa.T @ Aa
    cov = sig2 * np.linalg.pinv(gram)
    std_active = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    for j, s in zip(active, std_active):
        coef_std[j] = float(s)
    return coef_std


def _nnls_estimate(A: np.ndarray, y: np.ndarray, *, method: str,
                   sigma: float | None = None,
                   fixed_zero_indices: Sequence[int] = ()) -> dict[str, Any]:
    A = _as_2d(A)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    c, keep = _nnls_with_mask(A, y, fixed_zero_indices)
    residual = y - A @ c
    residual_norm = float(np.linalg.norm(residual))
    coef_std = _naive_covariance_std(A, c, keep, residual, sigma)
    return {
        "method": method,
        "c_hat": c.tolist(),
        "coef_std": coef_std.tolist(),
        "residual_norm": residual_norm,
        # Baselines emit no identifiability verdict and never merge/flag.
        "identifiability_flag": None,
        "report_groups": [[int(j)] for j in range(A.shape[1])],
    }


def plain_nnls(H_tilde, Y_tilde, *, sigma: float | None = None,
               fixed_zero_indices: Sequence[int] = ()) -> dict[str, Any]:
    """B1: plain NNLS on IASA's projected system; naive CI; no diagnostics."""
    return _nnls_estimate(H_tilde, Y_tilde, method="plain_nnls_B1",
                          sigma=sigma, fixed_zero_indices=fixed_zero_indices)


def cmb_nnls(H_lag, Y, *, sigma: float | None = None,
             fixed_zero_indices: Sequence[int] = ()) -> dict[str, Any]:
    """B3: chemical-mass-balance NNLS on the raw (unprojected) response."""
    return _nnls_estimate(H_lag, Y, method="cmb_B3",
                          sigma=sigma, fixed_zero_indices=fixed_zero_indices)


def _observation_matrix(Y: np.ndarray, row_index: Sequence[dict[str, Any]]) -> np.ndarray:
    """Reshape the stacked sensor-time observation vector into a [T, S] matrix
    keyed by (time_index, sensor_index) for factorization."""
    times = sorted({int(r["time_index"]) for r in row_index})
    sensors = sorted({int(r["sensor_index"]) for r in row_index})
    t_pos = {t: i for i, t in enumerate(times)}
    s_pos = {s: i for i, s in enumerate(sensors)}
    M = np.full((len(times), len(sensors)), np.nan, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64).reshape(-1)
    for val, r in zip(y, row_index):
        M[t_pos[int(r["time_index"])], s_pos[int(r["sensor_index"])]] = float(val)
    # Fill any unobserved cells with the column (sensor) mean, then clip to
    # nonnegative -- PMF/NMF assume nonnegative receptor concentrations.
    col_mean = np.nanmean(np.where(np.isnan(M), np.nan, M), axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_mean, inds[1])
    return np.clip(M, 0.0, None)


def _group_sensor_patterns(H_lag: np.ndarray, c_true: np.ndarray,
                           column_index: Sequence[dict[str, Any]],
                           row_index: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[int]]:
    """True per-group sensor pattern (mean over time of the clean group signal),
    used as the best-case assignment target for PMF factors."""
    sensors = sorted({int(r["sensor_index"]) for r in row_index})
    times = sorted({int(r["time_index"]) for r in row_index})
    s_pos = {s: i for i, s in enumerate(sensors)}
    t_pos = {t: i for i, t in enumerate(times)}
    groups = sorted({int(ci["source_index"]) for ci in column_index})
    H = _as_2d(H_lag)
    c_true = np.asarray(c_true, dtype=np.float64).reshape(-1)
    patterns = np.zeros((len(groups), len(sensors)), dtype=np.float64)
    for gi, g in enumerate(groups):
        cols = [j for j, ci in enumerate(column_index) if int(ci["source_index"]) == g]
        sig = H[:, cols] @ c_true[cols]
        acc = np.zeros((len(times), len(sensors)), dtype=np.float64)
        cnt = np.zeros((len(times), len(sensors)), dtype=np.float64)
        for val, r in zip(sig, row_index):
            acc[t_pos[int(r["time_index"])], s_pos[int(r["sensor_index"])]] += float(val)
            cnt[t_pos[int(r["time_index"])], s_pos[int(r["sensor_index"])]] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
        patterns[gi] = grid.mean(axis=0)
    return patterns, groups


def pmf_nmf(Y, row_index: Sequence[dict[str, Any]], H_lag, c_true,
            column_index: Sequence[dict[str, Any]], *, K: int | None = None,
            seed: int = 0, n_init: int = 10) -> dict[str, Any]:
    """B2: PMF-style NMF on the [T, S] observation matrix, with factors matched to
    inventory groups by best-case (Hungarian) spatial correlation. Returns
    apportionment *shares* per group (fraction of reconstructed sensor signal),
    the units common to receptor methods; coefficient units are not defined here."""
    from sklearn.decomposition import NMF

    M = _observation_matrix(np.asarray(Y), row_index)          # [T, S]
    patterns, groups = _group_sensor_patterns(H_lag, c_true, column_index, row_index)
    k = int(K) if K is not None else len(groups)
    model = NMF(n_components=k, init="nndsvda", random_state=int(seed),
                max_iter=1000, tol=1e-6)
    W = model.fit_transform(M)          # [T, k] temporal factors
    Hf = model.components_              # [k, S] spatial loadings

    # Best-case factor -> group assignment by spatial correlation.
    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else 0.0

    corr = np.zeros((k, len(groups)), dtype=np.float64)
    for fi in range(k):
        for gi in range(len(groups)):
            corr[fi, gi] = _corr(Hf[fi], patterns[gi])
    rows_i, cols_i = linear_sum_assignment(-corr)              # maximize correlation
    factor_of_group = {int(groups[gj]): int(fi) for fi, gj in zip(rows_i, cols_i)}

    # Reconstructed signal mass per assigned factor -> apportionment shares.
    mass = np.array([float(W[:, fi].sum() * Hf[fi].sum()) for fi in range(k)])
    total = float(mass.sum()) if mass.sum() > 0 else 1.0
    shares = {int(g): float(mass[factor_of_group[int(g)]] / total) for g in groups}

    recon = W @ Hf
    recon_err = float(np.linalg.norm(M - recon) / max(np.linalg.norm(M), 1e-12))
    return {
        "method": "pmf_nmf_B2",
        "K": k,
        "shares_hat": shares,
        "factor_of_group": {str(g): factor_of_group[int(g)] for g in groups},
        "assignment_correlation": {str(groups[gj]): float(corr[fi, gj])
                                   for fi, gj in zip(rows_i, cols_i)},
        "reconstruction_relative_error": recon_err,
        "identifiability_flag": None,
    }
