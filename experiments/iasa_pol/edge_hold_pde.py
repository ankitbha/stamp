"""Experiment 5b structural transport generator: the edge-hold advection-diffusion PDE.

This is the *auxiliary structural mismatch* generator for Experiment 5. It builds
observations with a first-order upwind advection + five-point Laplacian +
two-stage Heun integrator on the ``sim/polsim.py`` primitives, with diffusivity
``k=3e-4`` and the edge-hold boundary already implemented by
``polsim._neighbors_lr_tb`` (boundary cells reuse themselves -> mass is held at
the edge rather than exiting). The IASA fit deliberately uses the OPEN-BOUNDARY
puff response, so this operator is a genuine structural mismatch.

The generator is labeled ``edge_hold_pde`` in its returned metadata and is never
silently substituted for the puff response: callers pass its observations into
the same projection/fit pipeline that uses ``build_lagged_response_matrix``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

import sim.polsim as polsim

EDGE_HOLD_DIFFUSIVITY = 3e-4
GENERATOR_LABEL = "edge_hold_pde"


@dataclass(frozen=True)
class EdgeHoldConfig:
    dt: float = 1.0
    substeps_per_step: int = 4  # Heun sub-steps per observation interval (stability)
    diffusivity: float = EDGE_HOLD_DIFFUSIVITY
    dx: float = 1.0
    dy: float = 1.0


def _sensor_cells(sensor_xy: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(sensor_xy, dtype=np.float64)
    ix = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, ny - 1)
    return ix, iy


def simulate_edge_hold_observations(
    source_terms: np.ndarray,  # [T, Nx, Ny] emission field per observation step
    wind_vx: np.ndarray,       # [T]
    wind_vy: np.ndarray,       # [T]
    sensor_xy: np.ndarray,     # [M, 2] grid coordinates
    *,
    config: EdgeHoldConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Integrate the edge-hold advection-diffusion PDE and sample at sensor cells.

    Returns per-(time, sensor) concentrations plus provenance. Two-stage Heun with
    ``substeps_per_step`` sub-steps per observation interval; the emission field
    for observation step ``t`` is applied across that interval.
    """
    cfg = config or EdgeHoldConfig()
    src = np.asarray(source_terms, dtype=np.float64)
    if src.ndim != 3:
        raise ValueError("source_terms must have shape [T, Nx, Ny]")
    T, nx, ny = src.shape
    vx = np.asarray(wind_vx, dtype=np.float64).reshape(-1)
    vy = np.asarray(wind_vy, dtype=np.float64).reshape(-1)
    if vx.shape[0] < T or vy.shape[0] < T:
        raise ValueError("wind arrays must cover every observation step")

    dev = torch.device(device)
    dtype = torch.float64
    k = torch.tensor(float(cfg.diffusivity), dtype=dtype, device=dev)
    U = torch.zeros((nx, ny), dtype=dtype, device=dev)
    ix, iy = _sensor_cells(sensor_xy, nx, ny)
    M = ix.shape[0]
    obs = np.zeros((T, M), dtype=np.float64)

    sub_dt = float(cfg.dt) / int(cfg.substeps_per_step)
    src_t = torch.as_tensor(src, dtype=dtype, device=dev)
    for t in range(T):
        S_t = src_t[t]
        # polsim.advection_upwind uses torch.where(Vx >= 0, ...), which requires a
        # tensor condition; pass 0-dim tensors (not Python floats) so it broadcasts.
        Vx = torch.tensor(float(vx[t]), dtype=dtype, device=dev)
        Vy = torch.tensor(float(vy[t]), dtype=dtype, device=dev)
        for _ in range(int(cfg.substeps_per_step)):
            # Two-stage Heun (Heun's method / improved Euler).
            k1 = polsim.rhs(U, Vx, Vy, k, S_t, cfg.dx, cfg.dy)
            U_pred = U + sub_dt * k1
            k2 = polsim.rhs(U_pred, Vx, Vy, k, S_t, cfg.dx, cfg.dy)
            U = U + 0.5 * sub_dt * (k1 + k2)
            U = torch.clamp(U, min=0.0)  # concentrations are nonnegative
        sampled = U[torch.as_tensor(ix, device=dev), torch.as_tensor(iy, device=dev)]
        obs[t] = sampled.detach().cpu().numpy()

    return {
        "generator": GENERATOR_LABEL,
        "observations": obs,  # [T, M]
        "final_max_concentration": float(U.max().item()),
        "config": {
            "dt": cfg.dt,
            "substeps_per_step": cfg.substeps_per_step,
            "diffusivity": cfg.diffusivity,
            "dx": cfg.dx,
            "dy": cfg.dy,
            "boundary": "edge_hold",
            "advection": "first_order_upwind",
            "laplacian": "five_point",
            "integrator": "two_stage_heun",
        },
        "T": int(T),
        "n_sensors": int(M),
    }


def observations_to_row_vector(
    obs: np.ndarray,  # [T, M]
    row_index: Sequence[dict[str, Any]],
    sensor_ids: Sequence[str],
) -> np.ndarray:
    """Flatten [T, M] sensor-time concentrations into a Y vector aligned to the
    puff response ``row_index`` (each row carries time_index + sensor_index)."""
    id_to_col = {sid: j for j, sid in enumerate(sensor_ids)}
    Y = np.zeros(len(row_index), dtype=np.float64)
    for r, row in enumerate(row_index):
        t = int(row["time_index"])
        si = row.get("sensor_index")
        if si is None:
            si = id_to_col[row["sensor_id"]]
        Y[r] = obs[t, int(si)]
    return Y
