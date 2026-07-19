"""Per-sensor contribution decomposition and spatial footprints (Task 9B).

Resolves each monitor's fitted signal into contributing source groups and spatial
cells of origin (paper 5.theory.tex subsec:per_sensor_footprints). Two products:

1. Exact per-sensor contribution decomposition. Because the response is linear in
   ``c``, ``y_tilde_{s,t} = sum_{k,b} H_tilde_{(s,t),(k,b)} c_hat_{kb}``, so the
   contribution of source ``k`` to sensor ``s`` is ``Y_hat^{(s)}_k = sum_t sum_b
   H_{(s,t),(k,b)} c_hat_{kb}`` (projected on ``H_tilde``, raw on ``H_lag``). This
   is pure linear algebra on the already-built matrices -- no rebuild.

2. Nonnegative sensor footprints ``F_s(i)`` over source cells: the puff response
   read backward from the sensor. Obtained by re-running the Task 5 builder with
   one-hot single-cell sources (no new transport operator), then weighting by the
   inventory maps and fitted coefficients for the fitted per-group footprint.

Per-sensor attribution uses only that sensor's rows of ``H_tilde``, so identifiability
is inherited, not created: ``sigma_J(H_tilde^{(s)}) <= sigma_J(H_tilde)`` and
``rank(H_tilde^{(s)}) <= rank(H_tilde)`` (paper eq. footprint_inheritance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from model.iasa.backend import to_numpy


@dataclass
class FootprintResult:
    sensor_ids: list[Any]
    per_sensor_source_contribution_projected: dict[str, dict[str, float]]
    per_sensor_source_contribution_raw: dict[str, dict[str, float]]
    per_sensor_group_contribution_projected: dict[str, dict[str, float]] | None
    per_sensor_group_contribution_raw: dict[str, dict[str, float]] | None
    geometric_footprint: dict[str, list[list[float]]]
    fitted_footprint: dict[str, dict[str, list[list[float]]]]
    grid_shape: tuple[int, int]
    active_cells: list[tuple[int, int]]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "sensor_ids": [str(s) for s in self.sensor_ids],
            "per_sensor_source_contribution_projected": self.per_sensor_source_contribution_projected,
            "per_sensor_source_contribution_raw": self.per_sensor_source_contribution_raw,
            "per_sensor_group_contribution_projected": self.per_sensor_group_contribution_projected,
            "per_sensor_group_contribution_raw": self.per_sensor_group_contribution_raw,
            "geometric_footprint": self.geometric_footprint,
            "fitted_footprint": self.fitted_footprint,
            "grid_shape": list(self.grid_shape),
            "active_cells": [list(c) for c in self.active_cells],
            "metadata": self.metadata,
        }


def _sensor_rows(row_index: Sequence[dict[str, Any]]) -> dict[int, tuple[Any, list[int]]]:
    """Map sensor_index -> (sensor_id, row positions) over the time-major layout."""
    out: dict[int, tuple[Any, list[int]]] = {}
    for r, row in enumerate(row_index):
        si = int(row["sensor_index"])
        if si not in out:
            out[si] = (row.get("sensor_id", si), [])
        out[si][1].append(r)
    return out


def _group_key(members: Sequence[int], names: dict[int, str]) -> str:
    return "+".join(names[int(m)] for m in sorted(members))


def decompose_per_sensor(
    H_tilde: torch.Tensor,
    H_lag: torch.Tensor,
    c_hat: torch.Tensor,
    row_index: Sequence[dict[str, Any]],
    column_index: Sequence[dict[str, Any]],
    *,
    groups: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Exact per-sensor per-source (and per-group) contribution decomposition."""
    device = H_tilde.device
    dtype = H_tilde.dtype
    c = c_hat.to(device=device, dtype=dtype).reshape(-1)
    Ht = H_tilde.to(device=device, dtype=dtype)
    Hl = H_lag.to(device=device, dtype=dtype)
    if Ht.shape[1] != len(column_index) or c.shape[0] != len(column_index):
        raise ValueError("column_index, H_tilde columns, and c_hat must align")

    # Column contribution vectors: H[:, col] * c[col].
    proj_cols = Ht * c[None, :]
    raw_cols = Hl * c[None, :]

    source_of = [int(col["source_index"]) for col in column_index]
    source_name = {int(col["source_index"]): str(col.get("source_name")) for col in column_index}

    sensor_rows = _sensor_rows(row_index)
    proj_src: dict[str, dict[str, float]] = {}
    raw_src: dict[str, dict[str, float]] = {}
    for si, (sid, rows) in sensor_rows.items():
        idx = torch.tensor(rows, dtype=torch.long, device=device)
        proj_row_sum = proj_cols.index_select(0, idx).sum(dim=0)  # [J]
        raw_row_sum = raw_cols.index_select(0, idx).sum(dim=0)
        p: dict[str, float] = {}
        r: dict[str, float] = {}
        for col_pos, k in enumerate(source_of):
            name = source_name[k]
            p[name] = p.get(name, 0.0) + float(proj_row_sum[col_pos])
            r[name] = r.get(name, 0.0) + float(raw_row_sum[col_pos])
        proj_src[str(sid)] = p
        raw_src[str(sid)] = r

    proj_grp = raw_grp = None
    if groups is not None:
        proj_grp = {}
        raw_grp = {}
        for si, (sid, _) in sensor_rows.items():
            gp: dict[str, float] = {}
            gr: dict[str, float] = {}
            for members in groups:
                key = _group_key(members, source_name)
                gp[key] = float(sum(proj_src[str(sid)][source_name[int(m)]] for m in members))
                gr[key] = float(sum(raw_src[str(sid)][source_name[int(m)]] for m in members))
            proj_grp[str(sid)] = gp
            raw_grp[str(sid)] = gr

    return {
        "per_sensor_source_contribution_projected": proj_src,
        "per_sensor_source_contribution_raw": raw_src,
        "per_sensor_group_contribution_projected": proj_grp,
        "per_sensor_group_contribution_raw": raw_grp,
        "sensor_rows": {str(sid): rows for si, (sid, rows) in sensor_rows.items()},
    }


def per_sensor_identifiability(
    projection: Any,
    sensor_index: int,
    *,
    config: Any = None,
    pooled: Any = None,
) -> dict[str, Any]:
    """Diagnose the per-sensor row submatrix and confirm inherited identifiability."""
    from model.iasa.diagnostics import diagnose_identifiability

    row_index = projection.row_index
    rows = [r for r, row in enumerate(row_index) if int(row["sensor_index"]) == int(sensor_index)]
    if not rows:
        raise ValueError(f"no rows for sensor_index {sensor_index}")
    idx = torch.tensor(rows, dtype=torch.long, device=projection.H_tilde.device)
    sub = projection.H_tilde.index_select(0, idx)
    sub_diag = diagnose_identifiability(sub, projection.column_index, config=config)
    pooled_diag = pooled if pooled is not None else diagnose_identifiability(
        projection.H_tilde, projection.column_index, config=config
    )
    inherited = (sub_diag.sigma_J <= pooled_diag.sigma_J + 1e-9) and (sub_diag.numerical_rank <= pooled_diag.numerical_rank)
    return {
        "sensor_index": int(sensor_index),
        "sigma_J_sensor": sub_diag.sigma_J,
        "sigma_J_pooled": pooled_diag.sigma_J,
        "rank_sensor": sub_diag.numerical_rank,
        "rank_pooled": pooled_diag.numerical_rank,
        "inherited": bool(inherited),
    }


def _active_cells(maps: np.ndarray, threshold: float, max_cells: int) -> list[tuple[int, int]]:
    union = np.any(maps > threshold, axis=0)
    cells = [tuple(int(v) for v in c) for c in np.argwhere(union)]
    if len(cells) > max_cells:
        raise ValueError(f"active cell count {len(cells)} exceeds max_cells {max_cells}; raise cap or threshold")
    return cells


def compute_sensor_footprints(
    source_maps: np.ndarray,
    source_names: Sequence[str],
    basis: Any,
    observer: Any,
    wind: Any,
    *,
    fit: Any,
    projection: Any,
    response_config: Any = None,
    dispersion_config: Any = None,
    groups: Sequence[Sequence[int]] | None = None,
    cell_threshold: float = 1e-6,
    max_cells: int = 4096,
) -> FootprintResult:
    """Per-sensor decomposition + nonnegative footprint fields over source cells.

    Reuses the Task 5 open-boundary response builder with one-hot single-cell
    sources (no new transport operator) to obtain the per-cell pullback.
    """
    from model.iasa.response import build_lagged_response_matrix

    maps = np.asarray(source_maps, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError("source_maps must have shape [K, Nx, Ny]")
    K, nx, ny = maps.shape
    names = [str(s) for s in source_names]

    decomposition = decompose_per_sensor(
        projection.H_tilde, projection.H_tilde + projection.H_removed, fit.c_hat,
        projection.row_index, projection.column_index, groups=groups,
    )

    # One-hot single-cell sources over the union of active cells.
    cells = _active_cells(maps, cell_threshold, max_cells)
    cell_maps = np.zeros((len(cells), nx, ny), dtype=np.float32)
    for j, (ix, iy) in enumerate(cells):
        cell_maps[j, ix, iy] = 1.0
    cell_names = [f"cell_{ix}_{iy}" for (ix, iy) in cells]

    build_kwargs: dict[str, Any] = {}
    if response_config is not None:
        build_kwargs["response_config"] = response_config
    if dispersion_config is not None:
        build_kwargs["dispersion_config"] = dispersion_config
    cell_response = build_lagged_response_matrix(
        cell_maps, cell_names, basis, observer, wind, **build_kwargs
    )
    H_cell = to_numpy(cell_response.H_lag).astype(np.float64)  # [mT, n_cells * B]
    cell_cols = cell_response.column_index  # source_index=cell j, basis_index=b

    # Per-cell, per-basis pullback summed over each sensor's rows.
    sensor_rows = _sensor_rows(cell_response.row_index)
    if not np.isfinite(H_cell).all() or np.any(H_cell < -1e-9):
        raise ValueError("per-cell transport pullback must be finite and nonnegative")

    # Column lookups.
    cell_col_by_jb: dict[tuple[int, int], int] = {}
    for pos, col in enumerate(cell_cols):
        cell_col_by_jb[(int(col["source_index"]), int(col["basis_index"]))] = pos
    n_basis = 1 + max(int(col["basis_index"]) for col in cell_cols)

    # Fitted coefficients keyed by (source_index k, basis_index b).
    c_hat = to_numpy(fit.c_hat).reshape(-1)
    c_by_kb: dict[tuple[int, int], float] = {}
    for pos, col in enumerate(projection.column_index):
        c_by_kb[(int(col["source_index"]), int(col["basis_index"]))] = float(c_hat[pos])

    source_name_by_index = {int(col["source_index"]): str(col.get("source_name")) for col in projection.column_index}
    if groups is not None:
        group_defs = [(list(sorted(int(m) for m in members)),) for members in groups]
        group_keys = [_group_key(members, source_name_by_index) for members in groups]
    else:
        present = sorted(source_name_by_index)
        group_defs = [([k],) for k in present]
        group_keys = [source_name_by_index[k] for k in present]

    geometric: dict[str, list[list[float]]] = {}
    fitted: dict[str, dict[str, list[list[float]]]] = {}
    for si, (sid, rows) in sensor_rows.items():
        rows_arr = np.asarray(rows, dtype=np.int64)
        # geometric F_s(i) = sum_b sum_t H_cell[(s,t),(cell_j,b)] per cell.
        geom = np.zeros((nx, ny), dtype=np.float64)
        # per-cell per-basis pullback t-sum: [n_cells, n_basis]
        pull = np.zeros((len(cells), n_basis), dtype=np.float64)
        for j in range(len(cells)):
            for b in range(n_basis):
                col = cell_col_by_jb.get((j, b))
                if col is None:
                    continue
                pull[j, b] = float(H_cell[rows_arr, col].sum())
        for j, (ix, iy) in enumerate(cells):
            geom[ix, iy] = float(pull[j].sum())
        geometric[str(sid)] = geom.tolist()

        group_fields: dict[str, list[list[float]]] = {}
        for (members,), key in zip(group_defs, group_keys):
            field = np.zeros((nx, ny), dtype=np.float64)
            for k in members:
                mk = maps[k] if k < K else None
                if mk is None:
                    continue
                for j, (ix, iy) in enumerate(cells):
                    coeff_weighted = 0.0
                    for b in range(n_basis):
                        coeff_weighted += c_by_kb.get((k, b), 0.0) * pull[j, b]
                    field[ix, iy] += float(mk[ix, iy]) * coeff_weighted
            group_fields[key] = field.tolist()
        fitted[str(sid)] = group_fields

    metadata = {
        "grid_shape": [nx, ny],
        "n_active_cells": len(cells),
        "n_basis": int(n_basis),
        "cell_threshold": float(cell_threshold),
        "reused_operator": "task5_open_boundary_puff_one_hot_cell_sources",
        "response_boundary_mode": cell_response.metadata.get("boundary_mode"),
    }
    return FootprintResult(
        sensor_ids=[sensor_rows[si][0] for si in sorted(sensor_rows)],
        per_sensor_source_contribution_projected=decomposition["per_sensor_source_contribution_projected"],
        per_sensor_source_contribution_raw=decomposition["per_sensor_source_contribution_raw"],
        per_sensor_group_contribution_projected=decomposition["per_sensor_group_contribution_projected"],
        per_sensor_group_contribution_raw=decomposition["per_sensor_group_contribution_raw"],
        geometric_footprint=geometric,
        fitted_footprint=fitted,
        grid_shape=(nx, ny),
        active_cells=cells,
        metadata=metadata,
    )


__all__ = [
    "FootprintResult",
    "compute_sensor_footprints",
    "decompose_per_sensor",
    "per_sensor_identifiability",
]
