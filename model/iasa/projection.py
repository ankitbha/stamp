from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from model.iasa.background import BackgroundBasisResult
from model.iasa.backend import (
    as_tensor,
    dtype_name,
    resolve_device,
    resolve_dtype,
    runtime_provenance,
    tensor_hash,
)


@dataclass(frozen=True)
class ProjectionConfig:
    rank_tolerance: float | None = None
    allow_over_rank: bool = False
    device: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class BackgroundProjector:
    Q: torch.Tensor
    U_r: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: int
    tolerance: float
    row_index: list[dict[str, Any]]
    column_names: list[str]
    metadata: dict[str, Any]

    def project(self, X: torch.Tensor) -> torch.Tensor:
        values = as_tensor(X, device=self.U_r.device, dtype=self.U_r.dtype)
        if values.ndim not in (1, 2) or values.shape[0] != self.Q.shape[0]:
            raise ValueError("X must have shape [row_count] or [row_count,n]")
        if not torch.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        return values - self.U_r @ (self.U_r.T @ values)


@dataclass(frozen=True)
class ProjectionResult:
    H_tilde: torch.Tensor
    Y_tilde: torch.Tensor
    H_removed: torch.Tensor
    Y_removed: torch.Tensor
    Q: torch.Tensor
    U_r: torch.Tensor
    row_index: list[dict[str, Any]]
    column_index: list[dict[str, Any]]
    metadata: dict[str, Any]


def _rank(values: torch.Tensor, tolerance: float) -> int:
    if min(values.shape) == 0:
        return 0
    return int(torch.count_nonzero(torch.linalg.svdvals(values) > tolerance))


def fit_background_projector(
    background_basis: BackgroundBasisResult,
    config: ProjectionConfig | None = None,
) -> BackgroundProjector:
    cfg = config or ProjectionConfig()
    device = resolve_device(cfg.device) if cfg.device is not None else background_basis.Q.device
    dtype = resolve_dtype(cfg.dtype, default=torch.float64) if cfg.dtype is not None else background_basis.Q.dtype
    Q = as_tensor(background_basis.Q, device=device, dtype=dtype)
    if Q.ndim != 2 or not torch.isfinite(Q).all():
        raise ValueError("background Q must be a finite matrix")
    if Q.shape[0] != len(background_basis.row_index):
        raise ValueError("background Q rows must align with row_index")
    if Q.shape[1] != len(background_basis.column_names):
        raise ValueError("background Q columns must align with column_names")
    if cfg.rank_tolerance is not None and (cfg.rank_tolerance != cfg.rank_tolerance or cfg.rank_tolerance < 0):
        raise ValueError("rank_tolerance must be finite and nonnegative")

    if min(Q.shape) == 0:
        singular_values = torch.empty(0, dtype=dtype, device=device)
        U_r = torch.empty((Q.shape[0], 0), dtype=dtype, device=device)
        tolerance = 0.0
    else:
        U, singular_values, _ = torch.linalg.svd(Q, full_matrices=False)
        default_tolerance = float(max(Q.shape) * torch.finfo(dtype).eps * float(singular_values[0]))
        tolerance = default_tolerance if cfg.rank_tolerance is None else float(cfg.rank_tolerance)
        keep = singular_values > tolerance
        U_r = U[:, keep]
    effective_rank = int(U_r.shape[1])
    basis_mode = str(background_basis.metadata.get("basis_mode", "normal"))
    max_rank = int(background_basis.metadata.get("max_background_rank", 8))
    if effective_rank > max_rank and not (cfg.allow_over_rank and basis_mode == "stress"):
        raise ValueError("background effective rank exceeds its cap; only labeled stress bases may opt out")

    independent: list[str] = []
    dependent: list[str] = []
    current_rank = 0
    for index, name in enumerate(background_basis.column_names):
        prefix_rank = _rank(Q[:, :index + 1], tolerance)
        if prefix_rank > current_rank:
            independent.append(name)
        else:
            dependent.append(name)
        current_rank = prefix_rank
    if len(independent) != effective_rank:
        raise RuntimeError("prefix rank classification is inconsistent with effective rank")
    metadata = {
        "method": "thin_svd_implicit_orthogonal_projection",
        "basis_mode": basis_mode,
        "effective_rank": effective_rank,
        "rank_tolerance": tolerance,
        "singular_values": [float(v) for v in singular_values.detach().cpu().tolist()],
        "independent_column_names": independent,
        "dependent_column_names": dependent,
        "input_shape": list(Q.shape),
        "row_ordering": background_basis.metadata.get("row_ordering"),
        "allow_over_rank": bool(cfg.allow_over_rank),
        "background_basis_hash": tensor_hash(Q),
        **runtime_provenance(device, dtype),
    }
    return BackgroundProjector(
        Q=Q.clone(), U_r=U_r, singular_values=singular_values, effective_rank=effective_rank,
        tolerance=tolerance, row_index=[dict(row) for row in background_basis.row_index],
        column_names=list(background_basis.column_names), metadata=metadata,
    )


def _norm_ratios(
    projected: torch.Tensor, removed: torch.Tensor, original: torch.Tensor
) -> tuple[list[float], list[float]]:
    if original.ndim == 1:
        original = original[:, None]
        projected = projected[:, None]
        removed = removed[:, None]
    denom = torch.linalg.vector_norm(original, dim=0)
    safe = denom > 0
    visible = torch.where(safe, torch.linalg.vector_norm(projected, dim=0) / denom, torch.zeros_like(denom))
    absorbed = torch.where(safe, torch.linalg.vector_norm(removed, dim=0) / denom, torch.zeros_like(denom))
    return [float(v) for v in visible.detach().cpu().tolist()], [float(v) for v in absorbed.detach().cpu().tolist()]


def project_response_and_observations(
    H_lag: torch.Tensor,
    Y: torch.Tensor,
    background_basis: BackgroundBasisResult,
    row_index: list[dict[str, Any]],
    column_index: list[dict[str, Any]],
    config: ProjectionConfig | None = None,
) -> ProjectionResult:
    cfg = config or ProjectionConfig()
    device = resolve_device(cfg.device) if cfg.device is not None else background_basis.Q.device
    dtype = resolve_dtype(cfg.dtype, default=torch.float64) if cfg.dtype is not None else torch.float64
    H = as_tensor(H_lag, device=device, dtype=dtype)
    y = as_tensor(Y, device=device, dtype=dtype)
    if H.ndim != 2 or y.ndim != 1:
        raise ValueError("H_lag must have shape [mT,K] and Y must have shape [mT]")
    if not torch.isfinite(H).all() or not torch.isfinite(y).all():
        raise ValueError("H_lag and Y must contain only finite values")
    if H.shape[0] != y.shape[0] or H.shape[0] != len(row_index):
        raise ValueError("H_lag, Y, and row_index row counts must match")
    if H.shape[1] != len(column_index):
        raise ValueError("column_index length must match H_lag columns")
    if row_index != background_basis.row_index:
        raise ValueError("response and background row_index metadata must match exactly")
    projector = fit_background_projector(background_basis, ProjectionConfig(
        rank_tolerance=cfg.rank_tolerance,
        allow_over_rank=cfg.allow_over_rank,
        device=str(device),
        dtype=dtype_name(dtype),
    ))
    H_tilde = projector.project(H)
    Y_tilde = projector.project(y)
    H_removed = H - H_tilde
    Y_removed = y - Y_tilde
    h_visibility, h_absorption = _norm_ratios(H_tilde, H_removed, H)
    y_visibility, y_absorption = _norm_ratios(Y_tilde, Y_removed, y)
    eps = torch.finfo(dtype).eps
    h_scale = max(float(torch.linalg.matrix_norm(H, ord="fro")), float(eps))
    y_scale = max(float(torch.linalg.vector_norm(y)), float(eps))
    metadata = {
        **projector.metadata,
        "input_H_shape": list(H.shape),
        "input_Y_shape": list(y.shape),
        "output_H_shape": list(H_tilde.shape),
        "output_Y_shape": list(Y_tilde.shape),
        "background_column_names": list(background_basis.column_names),
        "background_column_types": [item.get("type") for item in background_basis.metadata.get("column_provenance", [])],
        "H_orthogonality_residual": float(torch.linalg.matrix_norm(projector.U_r.T @ H_tilde, ord="fro") / h_scale),
        "Y_orthogonality_residual": float(torch.linalg.vector_norm(projector.U_r.T @ Y_tilde) / y_scale),
        "idempotence_residual": float(torch.linalg.matrix_norm(projector.project(H_tilde) - H_tilde, ord="fro") / h_scale),
        "H_visibility_ratio_by_column": h_visibility,
        "H_absorption_ratio_by_column": h_absorption,
        "Y_visibility_ratio": y_visibility[0],
        "Y_absorption_ratio": y_absorption[0],
    }
    return ProjectionResult(
        H_tilde=H_tilde, Y_tilde=Y_tilde, H_removed=H_removed, Y_removed=Y_removed,
        Q=projector.Q.clone(), U_r=projector.U_r.clone(), row_index=[dict(row) for row in row_index],
        column_index=[dict(column) for column in column_index], metadata=metadata,
    )


__all__ = [
    "BackgroundProjector", "ProjectionConfig", "ProjectionResult", "fit_background_projector",
    "project_response_and_observations",
]
