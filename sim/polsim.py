#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
sim/polsim.py

Differentiable (PyTorch) 2D advection-diffusion-source utilities for the
STAMP / FieldFormer pollution case.

This file is adapted from the FieldFormer data-generation script (pollution.py) but refactored
into a reusable simulator module, similar in spirit to heatsim.py / swesim.py.

Core PDE form (in normalized coordinates x,y in [0,1]):

    dU/dt = - Vx(t) dU/dx - Vy(t) dU/dy + k Laplacian(U) + S(x,y,t)

- U(x,y,t): pollutant concentration (normalized)
- Vx(t), Vy(t): time-varying (monsoon-like) wind components
- k: diffusivity
- S(x,y,t): explicit source term supplied by future inventory-activity paths

Numerics:
- Upwind first-order for advection
- 5-point Laplacian for diffusion
- Heun / RK2 time integration
- Boundary handling:
    - By default we keep a simple "edge-hold" boundary (same as neighbors in pollution.py)
    - A more complex Orlanski open BC can be added later if needed.

Notes:
- This module assumes the intensity *.npy files are available in the same folder at runtime.
- Source inventory loading is handled without aggregating source categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np
import torch

from sim.pol_sources import PolSourceInventory, load_pol_source_inventory


# =============================================================================
# Grid / Params
# =============================================================================

DEFAULT_DIFFUSIVITY = 3e-4  # same value you were already using
DEFAULT_DTYPE = torch.float32

@dataclass
class PolGrid:
    Nx: int
    Ny: int
    Lx: float = 1.0
    Ly: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    x: Optional[torch.Tensor] = None  # [Nx]
    y: Optional[torch.Tensor] = None  # [Ny]

    # Base directory for pollution inputs (sources + govdata files)
    src_dir: str = "./"

    # Deprecated placeholder only; source loaders no longer populate aggregate S_known.
    S_known: Optional[torch.Tensor] = None

    # Named source inventories for the IASA path. These are not aggregated.
    source_names: Optional[list[str]] = None
    source_maps: Optional[torch.Tensor] = None  # [K,Nx,Ny]
    source_matrix: Optional[torch.Tensor] = None  # [Nx*Ny,K]
    source_activity_defaults: Optional[Dict[str, Any]] = None
    source_metadata: Optional[Dict[str, Any]] = None

    # Cached initial condition field (constructed from govdata via kriging) [Nx,Ny]
    U0: Optional[torch.Tensor] = None

    # Crop window applied to 80x80 intensity maps to produce 40x40 domain
    crop: Tuple[slice, slice] = (slice(21, 61), slice(16, 56))


@dataclass
class PolParams:
    # diffusivity (scalar tensor, fixed)
    k: torch.Tensor = torch.tensor(DEFAULT_DIFFUSIVITY, dtype=DEFAULT_DTYPE)

    # Base winds (scalars)
    Vx0: float = 1.12
    Vy0: float = 0.984

    # Monsoon variation settings
    sim_seconds_per_day: float = 5.0
    diurnal_amp_frac: float = 0.5  # ± around base magnitude
    ar1_rho: float = 0.90
    ar1_sigma_frac: float = 0.15  # fraction of base magnitude for AR(1) noise

    # Numerical safety
    eps: float = 1e-8


# =============================================================================
# Sources I/O
# =============================================================================

def make_grid(
    Nx: int = 40,
    Ny: int = 40,
    Lx: float = 1.0,
    Ly: float = 1.0,
    src_dir: str = "./",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    load_inventory: bool = False,
) -> PolGrid:
    """Create grid (x,y,dx,dy) and optionally load named source inventories."""
    x = torch.linspace(0.0, float(Lx), int(Nx), device=torch.device(device), dtype=dtype)
    y = torch.linspace(0.0, float(Ly), int(Ny), device=torch.device(device), dtype=dtype)
    dx = float(Lx) / float(Nx - 1)
    dy = float(Ly) / float(Ny - 1)

    grid = PolGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, x=x, y=y, src_dir=src_dir)

    if load_inventory:
        inventory = load_pol_source_inventory(src_dir=src_dir, crop=grid.crop)
        grid.source_names = list(inventory.source_names)
        grid.source_maps = torch.as_tensor(inventory.source_maps, device=torch.device(device), dtype=dtype)
        grid.source_matrix = torch.as_tensor(inventory.source_matrix, device=torch.device(device), dtype=dtype)
        grid.source_activity_defaults = inventory.source_activity_defaults
        grid.source_metadata = inventory.raw_metadata

    return grid



# =============================================================================
# Initial condition (U0) construction from govdata via kriging (matches pollution.py)
# =============================================================================

def build_U0_from_govdata_kriging(
    grid: PolGrid,
    *,
    root: Optional[str] = None,
    res_time: str = "1H",
    sensor: str = "pm25",
    ic_row_idx: int = 745,
    lon_min: float = 77.01,
    lon_max: float = 77.40,
    lat_min: float = 28.39,
    lat_max: float = 28.78,
    tz_offset_minutes: int = 330,
    variogram_model: str = "spherical",
    drop_station: str = "Pusa_IMD",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Construct initial concentration field U0 on the simulator grid using Universal Kriging,
    mirroring the FieldFormer pollution.py logic.

    Files expected in `root` (defaults to grid.src_dir):
      - govdata_locations.csv
      - govdata_{res_time}_current.csv

    Returns:
      U0_norm_torch: [Nx,Ny] tensor (percentile-99 normalized)
      meta: dict with useful debugging metadata (scale factors, selected timestamp, etc.)
    """
    # Local imports: keep sim runtime light if govdata isn't needed
    import pandas as pd
    import pytz
    from pykrige.uk import UniversalKriging

    if root is None:
        root = grid.src_dir
    root = root if root.endswith("/") else (root + "/")

    if device is None:
        device = torch.device("cpu") if grid.x is None else grid.x.device
    if dtype is None:
        dtype = torch.float32 if grid.x is None else grid.x.dtype

    filepath_locs_gov = f"{root}govdata_locations.csv"
    filepath_data_gov = f"{root}govdata_{res_time}_current.csv"

    locs = pd.read_csv(filepath_locs_gov, index_col=0)

    raw = pd.read_csv(filepath_data_gov, index_col=[0, 1], parse_dates=True)[sensor]
    raw.replace(0, np.nan, inplace=True)

    # Normalize to IST (FixedOffset(330)) and clip to data range (as in pollution.py)
    start_dt = raw.index.levels[1][0]
    end_dt = raw.index.levels[1][-1]
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize("UTC")
    start_dt = start_dt.tz_convert(pytz.FixedOffset(int(tz_offset_minutes)))
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize("UTC")
    end_dt = end_dt.tz_convert(pytz.FixedOffset(int(tz_offset_minutes)))

    data = raw.sort_index().loc[(slice(None), slice(start_dt, end_dt))]
    df = data.unstack(level=0)

    # Drop a problematic station (optional, matches pollution.py)
    if drop_station is not None:
        df = df.drop([drop_station], axis=1, errors="ignore")

    if df.shape[0] <= ic_row_idx:
        raise ValueError(f"ic_row_idx={ic_row_idx} out of range for df with {df.shape[0]} rows.")

    row = df.iloc[int(ic_row_idx)]
    xs = locs.loc[row.index]["Longitude"].to_numpy()
    ys = locs.loc[row.index]["Latitude"].to_numpy()
    zs = row.to_numpy()

    # Drop NaNs
    mask = ~np.isnan(zs)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]

    if zs.size == 0:
        raise ValueError("All station values were NaN for selected IC row; cannot krige U0.")

    UK = UniversalKriging(
        xs, ys, zs,
        variogram_model=str(variogram_model),
        verbose=False,
        enable_plotting=False,
        exact_values=True,
    )

    gridx = np.linspace(float(lon_min), float(lon_max), int(grid.Nx))
    gridy = np.linspace(float(lat_min), float(lat_max), int(grid.Ny))
    vals_grid, _ = UK.execute("grid", gridx, gridy)

    U0 = np.asarray(vals_grid.data, dtype=np.float32)

    IC_scale = float(np.percentile(U0, 99))
    U0_norm = (U0 / (IC_scale + 1e-12)).astype(np.float32, copy=False)

    U0_t = torch.as_tensor(U0_norm, device=device, dtype=dtype)

    meta: Dict[str, object] = {
        "filepath_locs_gov": filepath_locs_gov,
        "filepath_data_gov": filepath_data_gov,
        "sensor": sensor,
        "res_time": res_time,
        "ic_row_idx": int(ic_row_idx),
        "timestamp": df.index[int(ic_row_idx)],
        "IC_scale_p99": IC_scale,
        "num_stations_used": int(zs.size),
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "variogram_model": str(variogram_model),
        "drop_station": drop_station,
    }
    return U0_t, meta


# =============================================================================
# Wind model (monsoon-like variations)
# =============================================================================

def _ar1_noise(n: int, rho: float, sigma: float, device: torch.device, dtype: torch.dtype, seed: Optional[int] = None) -> torch.Tensor:
    """
    AR(1) process:
      x_0 ~ N(0, sigma^2 / (1-rho^2))
      x_t = rho x_{t-1} + eps_t, eps_t ~ N(0, sigma^2 (1-rho^2))
    Implemented in torch for reproducibility under given seed.
    """
    if n <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1).")

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))

    eps = torch.randn((n,), generator=g, device=device, dtype=dtype)
    out = torch.empty((n,), device=device, dtype=dtype)

    denom = max(1e-12, 1.0 - rho * rho)
    out[0] = float(sigma) / math.sqrt(denom) * eps[0]
    s = float(sigma) * math.sqrt(max(0.0, 1.0 - rho * rho))
    for i in range(1, n):
        out[i] = float(rho) * out[i - 1] + s * eps[i]
    return out


def monsoon_wind_series(
    t_seconds: torch.Tensor,
    Vx_base: float,
    Vy_base: float,
    sim_seconds_per_day: float = 5.0,
    diurnal_amp_frac: float = 0.5,
    ar1_rho: float = 0.90,
    ar1_sigma_frac: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce time series Vx(t), Vy(t) over t_seconds.

    - Diurnal modulation: a smooth sinusoid over simulated "days"
    - AR(1) stochastic perturbations to mimic weather variability

    Returns:
      Vx_steps, Vy_steps : both [T]
    """
    device, dtype = t_seconds.device, t_seconds.dtype
    T = int(t_seconds.numel())
    if T == 0:
        return t_seconds, t_seconds

    # Diurnal cycle: t in "days" (simulated)
    day_phase = 2.0 * math.pi * (t_seconds / float(sim_seconds_per_day))
    diurnal = 1.0 + float(diurnal_amp_frac) * torch.sin(day_phase)

    base_mag = math.sqrt(Vx_base * Vx_base + Vy_base * Vy_base) + 1e-12
    sigma = float(ar1_sigma_frac) * float(base_mag)

    n1 = _ar1_noise(T, rho=ar1_rho, sigma=sigma, device=device, dtype=dtype, seed=seed)
    n2 = _ar1_noise(T, rho=ar1_rho, sigma=sigma, device=device, dtype=dtype, seed=None if seed is None else (seed + 1))

    Vx = float(Vx_base) * diurnal + n1
    Vy = float(Vy_base) * diurnal + n2
    return Vx, Vy


# =============================================================================
# Numerical operators (edge-hold BC like pollution.py neighbors)
# =============================================================================

def _neighbors_lr_tb(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (L,R,T,B) neighbors with edge-hold boundary:
      left boundary uses itself, right boundary uses itself, etc.
    U: [..., Nx, Ny]
    """
    # x-direction is dim=-2, y-direction dim=-1
    L = torch.empty_like(U)
    R = torch.empty_like(U)
    T = torch.empty_like(U)
    B = torch.empty_like(U)

    # left/right neighbors in y-index (dim=-1) in pollution.py, but note their U indexing was [Ny,Nx].
    # Here we store [Nx,Ny] consistently, so neighbors along x are dim=-2 and along y are dim=-1.
    L[..., 1:, :] = U[..., :-1, :]
    L[..., 0, :] = U[..., 0, :]

    R[..., :-1, :] = U[..., 1:, :]
    R[..., -1, :] = U[..., -1, :]

    T[..., :, 1:] = U[..., :, :-1]
    T[..., :, 0] = U[..., :, 0]

    B[..., :, :-1] = U[..., :, 1:]
    B[..., :, -1] = U[..., :, -1]

    return L, R, T, B


def laplacian(U: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    L, R, T, B = _neighbors_lr_tb(U)
    return (R - 2.0 * U + L) / (dx * dx) + (B - 2.0 * U + T) / (dy * dy)


def advection_upwind(U: torch.Tensor, Vx: float | torch.Tensor, Vy: float | torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    First-order upwind advection: Vx dU/dx + Vy dU/dy using backward/forward differences based on sign.
    """
    L, R, T, B = _neighbors_lr_tb(U)
    dxb = (U - L) / dx
    dxf = (R - U) / dx
    dyb = (U - T) / dy
    dyf = (B - U) / dy

    # Vx, Vy could be tensors of shape [...]; broadcast should work
    Vx_t = Vx
    Vy_t = Vy

    dUx = torch.where(Vx_t >= 0.0, dxb, dxf)
    dUy = torch.where(Vy_t >= 0.0, dyb, dyf)
    return Vx_t * dUx + Vy_t * dUy


def rhs(U: torch.Tensor, Vx: float | torch.Tensor, Vy: float | torch.Tensor, k: torch.Tensor, S: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    RHS of advection-diffusion-source PDE:
      dU/dt = - (Vx dU/dx + Vy dU/dy) + k ΔU + S
    """
    adv = advection_upwind(U, Vx, Vy, dx, dy)
    diff = k * laplacian(U, dx, dy)
    return -adv + diff + S


__all__ = [
    "PolGrid",
    "PolParams",
    "make_grid",
    "PolSourceInventory",
    "load_pol_source_inventory",
    "monsoon_wind_series",
]
