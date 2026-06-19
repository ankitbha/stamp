from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from model.iasa.background import BackgroundBasisResult


@dataclass(frozen=True)
class ProjectionConfig:
    rank_tolerance: float | None = None
    allow_over_rank: bool = False


@dataclass(frozen=True)
class BackgroundProjector:
    Q: np.ndarray
    U_r: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    tolerance: float
    row_index: list[dict[str, Any]]
    column_names: list[str]
    metadata: dict[str, Any]

    def project(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.Q.shape[0]:
            raise ValueError("X must have shape [row_count] or [row_count,n]")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        return values - self.U_r @ (self.U_r.T @ values)


@dataclass(frozen=True)
class ProjectionResult:
    H_tilde: np.ndarray
    Y_tilde: np.ndarray
    H_removed: np.ndarray
    Y_removed: np.ndarray
    Q: np.ndarray
    U_r: np.ndarray
    row_index: list[dict[str, Any]]
    column_index: list[dict[str, Any]]
    metadata: dict[str, Any]


def _rank(values: np.ndarray, tolerance: float) -> int:
    if values.shape[1] == 0:
        return 0
    return int(np.count_nonzero(np.linalg.svd(values, full_matrices=False, compute_uv=False) > tolerance))


def fit_background_projector(
    background_basis: BackgroundBasisResult,
    config: ProjectionConfig | None = None,
) -> BackgroundProjector:
    cfg = config or ProjectionConfig()
    Q = np.asarray(background_basis.Q, dtype=np.float64)
    if Q.ndim != 2 or not np.isfinite(Q).all():
        raise ValueError("background Q must be a finite matrix")
    if Q.shape[0] != len(background_basis.row_index):
        raise ValueError("background Q rows must align with row_index")
    if Q.shape[1] != len(background_basis.column_names):
        raise ValueError("background Q columns must align with column_names")
    if cfg.rank_tolerance is not None and (not np.isfinite(cfg.rank_tolerance) or cfg.rank_tolerance < 0):
        raise ValueError("rank_tolerance must be finite and nonnegative")

    if Q.shape[1] == 0:
        singular_values = np.empty(0, dtype=np.float64)
        U_r = np.empty((Q.shape[0], 0), dtype=np.float64)
        tolerance = 0.0 if cfg.rank_tolerance is None else float(cfg.rank_tolerance)
    else:
        U, singular_values, _ = np.linalg.svd(Q, full_matrices=False)
        default_tolerance = float(max(Q.shape) * np.finfo(np.float64).eps * singular_values[0])
        tolerance = default_tolerance if cfg.rank_tolerance is None else float(cfg.rank_tolerance)
        U_r = U[:, singular_values > tolerance]
    effective_rank = int(U_r.shape[1])
    basis_mode = str(background_basis.metadata.get("basis_mode", "normal"))
    max_rank = int(background_basis.metadata.get("max_background_rank", 8))
    if effective_rank > max_rank and not (cfg.allow_over_rank and basis_mode == "stress"):
        raise ValueError("background effective rank exceeds its cap; only labeled stress bases may opt out")

    independent: list[str] = []
    dependent: list[str] = []
    selected = np.empty((Q.shape[0], 0), dtype=np.float64)
    current_rank = 0
    for index, name in enumerate(background_basis.column_names):
        candidate = np.column_stack([selected, Q[:, index]])
        candidate_rank = _rank(candidate, tolerance)
        if candidate_rank > current_rank:
            independent.append(name)
            selected = candidate
            current_rank = candidate_rank
        else:
            dependent.append(name)
    metadata = {
        "method": "thin_svd_implicit_orthogonal_projection",
        "basis_mode": basis_mode,
        "effective_rank": effective_rank,
        "rank_tolerance": tolerance,
        "singular_values": singular_values.astype(float).tolist(),
        "independent_column_names": independent,
        "dependent_column_names": dependent,
        "input_shape": list(Q.shape),
        "row_ordering": background_basis.metadata.get("row_ordering"),
        "allow_over_rank": bool(cfg.allow_over_rank),
    }
    return BackgroundProjector(
        Q=Q.copy(), U_r=U_r, singular_values=singular_values, effective_rank=effective_rank,
        tolerance=tolerance, row_index=[dict(row) for row in background_basis.row_index],
        column_names=list(background_basis.column_names), metadata=metadata,
    )


def _norm_ratios(projected: np.ndarray, removed: np.ndarray, original: np.ndarray) -> tuple[list[float], list[float]]:
    if original.ndim == 1:
        original = original[:, None]
        projected = projected[:, None]
        removed = removed[:, None]
    denom = np.linalg.norm(original, axis=0)
    visible = np.divide(np.linalg.norm(projected, axis=0), denom, out=np.zeros_like(denom), where=denom > 0)
    absorbed = np.divide(np.linalg.norm(removed, axis=0), denom, out=np.zeros_like(denom), where=denom > 0)
    return visible.astype(float).tolist(), absorbed.astype(float).tolist()


def project_response_and_observations(
    H_lag: np.ndarray,
    Y: np.ndarray,
    background_basis: BackgroundBasisResult,
    row_index: list[dict[str, Any]],
    column_index: list[dict[str, Any]],
    config: ProjectionConfig | None = None,
) -> ProjectionResult:
    H = np.asarray(H_lag, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if H.ndim != 2 or y.ndim != 1:
        raise ValueError("H_lag must have shape [mT,K] and Y must have shape [mT]")
    if not np.isfinite(H).all() or not np.isfinite(y).all():
        raise ValueError("H_lag and Y must contain only finite values")
    if H.shape[0] != y.shape[0] or H.shape[0] != len(row_index):
        raise ValueError("H_lag, Y, and row_index row counts must match")
    if H.shape[1] != len(column_index):
        raise ValueError("column_index length must match H_lag columns")
    if row_index != background_basis.row_index:
        raise ValueError("response and background row_index metadata must match exactly")
    projector = fit_background_projector(background_basis, config)
    H_tilde = projector.project(H)
    Y_tilde = projector.project(y)
    H_removed = H - H_tilde
    Y_removed = y - Y_tilde
    h_visibility, h_absorption = _norm_ratios(H_tilde, H_removed, H)
    y_visibility, y_absorption = _norm_ratios(Y_tilde, Y_removed, y)
    h_scale = max(float(np.linalg.norm(H)), np.finfo(np.float64).eps)
    y_scale = max(float(np.linalg.norm(y)), np.finfo(np.float64).eps)
    metadata = {
        **projector.metadata,
        "input_H_shape": list(H.shape),
        "input_Y_shape": list(y.shape),
        "output_H_shape": list(H_tilde.shape),
        "output_Y_shape": list(Y_tilde.shape),
        "background_column_names": list(background_basis.column_names),
        "background_column_types": [item.get("type") for item in background_basis.metadata.get("column_provenance", [])],
        "H_orthogonality_residual": float(np.linalg.norm(projector.U_r.T @ H_tilde) / h_scale),
        "Y_orthogonality_residual": float(np.linalg.norm(projector.U_r.T @ Y_tilde) / y_scale),
        "idempotence_residual": float(np.linalg.norm(projector.project(H_tilde) - H_tilde) / h_scale),
        "H_visibility_ratio_by_column": h_visibility,
        "H_absorption_ratio_by_column": h_absorption,
        "Y_visibility_ratio": y_visibility[0],
        "Y_absorption_ratio": y_absorption[0],
    }
    return ProjectionResult(
        H_tilde=H_tilde, Y_tilde=Y_tilde, H_removed=H_removed, Y_removed=Y_removed,
        Q=projector.Q.copy(), U_r=projector.U_r.copy(), row_index=[dict(row) for row in row_index],
        column_index=[dict(column) for column in column_index], metadata=metadata,
    )


__all__ = [
    "BackgroundProjector", "ProjectionConfig", "ProjectionResult", "fit_background_projector",
    "project_response_and_observations",
]
