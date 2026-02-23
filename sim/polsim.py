#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
sim/polsim.py

Differentiable (PyTorch) 2D advection–diffusion–source simulator for the STAMP / FieldFormer
pollution case.

This file is adapted from the FieldFormer data-generation script (pollution.py) but refactored
into a reusable simulator module, similar in spirit to heatsim.py / swesim.py.

Core PDE (in normalized coordinates x,y \in [0,1]):

    ∂_t U = - Vx(t) ∂_x U - Vy(t) ∂_y U  +  k ΔU  +  S_known(x,y) + S_unknown(x,y)

- U(x,y,t): pollutant concentration (normalized)
- Vx(t), Vy(t): time-varying (monsoon-like) wind components
- k: diffusivity
- S_known: static known sources loaded from *.npy intensity maps and cropped to 40x40
- S_unknown: a 10x10 coarse matrix (optionally batched) representing unknown sources to infer;
            it is smoothed and upsampled to 40x40 internally before use

Numerics:
- Upwind first-order for advection
- 5-point Laplacian for diffusion
- Heun / RK2 time integration
- Boundary handling:
    - By default we keep a simple "edge-hold" boundary (same as neighbors in pollution.py)
    - A more complex Orlanski open BC can be added later if needed.

Notes:
- This module assumes the intensity *.npy files are available in the same folder at runtime.
- All operations are implemented in PyTorch to allow gradients w.r.t. S_unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import math
import numpy as np
import torch
import torch.nn.functional as F


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

    # Known sources (normalized) on the simulation grid [Nx,Ny]
    S_known: Optional[torch.Tensor] = None

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
# Sources I/O (keep this part aligned with pollution.py)
# =============================================================================

def load_known_sources_40x40(
    src_dir: str = "./",
    dtype: np.dtype = np.float32,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    """
    Load known sources from intensity maps and crop to the 40x40 domain, matching pollution.py.

    Files expected in src_dir:
      - brick_kilns_intensity_80x80.npy
      - industries_intensity_80x80.npy
      - population_density_intensity_80x80.npy
      - traffic_00_intensity_80x80.npy
      - traffic_06_intensity_80x80.npy
      - traffic_12_intensity_80x80.npy
      - traffic_18_intensity_80x80.npy

    Returns:
      S_norm_torch: [40,40] torch tensor, percentile-99 normalized (like pollution.py)
      raw: dict of raw numpy arrays (useful for debugging)
    """
    src_dir = src_dir if src_dir.endswith("/") else (src_dir + "/")

    brick_kilns = np.load(src_dir + "brick_kilns_intensity_80x80.npy").astype(dtype, copy=False)
    industries = np.load(src_dir + "industries_intensity_80x80.npy").astype(dtype, copy=False)
    population_density = np.load(src_dir + "population_density_intensity_80x80.npy").astype(dtype, copy=False)
    traffic_00 = np.load(src_dir + "traffic_00_intensity_80x80.npy").astype(dtype, copy=False)
    traffic_06 = np.load(src_dir + "traffic_06_intensity_80x80.npy").astype(dtype, copy=False)
    traffic_12 = np.load(src_dir + "traffic_12_intensity_80x80.npy").astype(dtype, copy=False)
    traffic_18 = np.load(src_dir + "traffic_18_intensity_80x80.npy").astype(dtype, copy=False)

    traffic = (traffic_00 + traffic_06 + traffic_12 + traffic_18) / 4.0
    known_source_full = brick_kilns + industries + population_density + traffic

    # Match the exact crop in pollution.py -> 40x40
    known_source = known_source_full[21:61, 16:56]
    S_scale = np.percentile(known_source, 99)
    S_norm = (known_source / (S_scale + 1e-12)).astype(dtype, copy=False)

    S_norm_torch = torch.as_tensor(S_norm, device=torch.device(device), dtype=torch.float32)

    raw = {
        "brick_kilns": brick_kilns,
        "industries": industries,
        "population_density": population_density,
        "traffic_00": traffic_00,
        "traffic_06": traffic_06,
        "traffic_12": traffic_12,
        "traffic_18": traffic_18,
        "known_source_full": known_source_full,
        "known_source_cropped": known_source,
        "S_scale_p99": np.array([S_scale], dtype=np.float32),
    }
    return S_norm_torch, raw


def make_grid(
    Nx: int = 40,
    Ny: int = 40,
    Lx: float = 1.0,
    Ly: float = 1.0,
    src_dir: str = "./",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    load_sources: bool = True,
) -> PolGrid:
    """Create grid (x,y,dx,dy) and optionally load known sources."""
    x = torch.linspace(0.0, float(Lx), int(Nx), device=torch.device(device), dtype=dtype)
    y = torch.linspace(0.0, float(Ly), int(Ny), device=torch.device(device), dtype=dtype)
    dx = float(Lx) / float(Nx - 1)
    dy = float(Ly) / float(Ny - 1)

    grid = PolGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, x=x, y=y, src_dir=src_dir)

    if load_sources:
        S_known, _raw = load_known_sources_40x40(src_dir=src_dir, device=device)
        grid.S_known = S_known.to(device=torch.device(device), dtype=dtype)

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


# =============================================================================
# Source mixing: known + unknown
# =============================================================================

def combine_sources(
    S_known: torch.Tensor,
    S_unknown: torch.Tensor,
    mode: Literal["add", "softplus"] = "add",
    softplus_beta: float = 1.0,
) -> torch.Tensor:
    """
    Combine known + unknown sources.

    S_known: [Nx,Ny]
    S_unknown: [Nx,Ny] or [...,Nx,Ny] (batched)
      - if S_unknown is batched, S_known is broadcast.

    mode:
      - "add": S_total = S_known + S_unknown
      - "softplus": S_total = S_known + softplus(S_unknown)

    Note: rollout_pollution expects S_unknown on a 10x10 coarse grid and will smooth + upsample it to the
simulation grid (typically 40x40) before combining; using softplus can enforce nonnegativity.
    """
    if mode == "add":
        return S_known + S_unknown
    if mode == "softplus":
        return S_known + F.softplus(S_unknown, beta=float(softplus_beta))
    raise ValueError("mode must be 'add' or 'softplus'.")



# =============================================================================
# Unknown source parameterization: 10x10 -> smooth -> upsample to [Nx,Ny]
# =============================================================================

def _gaussian_kernel2d(
    sigma: float,
    truncate: float = 3.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel suitable for depthwise conv2d."""
    if sigma <= 0:
        # Degenerate: no smoothing
        k = torch.zeros((1, 1, 1, 1), device=device, dtype=dtype)
        k[..., 0, 0] = 1.0
        return k

    radius = int(math.ceil(float(truncate) * float(sigma)))
    size = 2 * radius + 1
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-0.5 * (xs / float(sigma)) ** 2)
    g1 = g1 / (g1.sum() + 1e-12)
    g2 = torch.outer(g1, g1)
    g2 = g2 / (g2.sum() + 1e-12)
    return g2.view(1, 1, size, size)


def smooth_and_upsample_unknown(
    S_coarse: torch.Tensor,
    out_hw: Tuple[int, int],
    *,
    sigma: float = 1.0,
    truncate: float = 3.0,
    mode: Literal["bilinear", "bicubic"] = "bilinear",
) -> torch.Tensor:
    """
    Convert a 10x10 coarse unknown-source grid into an [H,W] field:

      1) Smooth on the coarse grid using a Gaussian kernel (helps avoid checkerboard artifacts).
      2) Upsample to (H,W) using interpolation.

    Args:
      S_coarse: [..., 10, 10]
      out_hw: (H, W) target resolution (e.g., (40,40))
      sigma: Gaussian std (in coarse-grid pixels). sigma=0 disables smoothing.
      mode: interpolation mode for upsampling.

    Returns:
      S_fine: [..., H, W]
    """
    if S_coarse.ndim < 2:
        raise ValueError("S_coarse must have shape [..., 10, 10].")
    if tuple(S_coarse.shape[-2:]) != (10, 10):
        raise AssertionError(f"S_unknown must be a 10x10 coarse grid; got shape {tuple(S_coarse.shape)}")

    device, dtype = S_coarse.device, S_coarse.dtype

    # Reshape to NCHW for conv/interp: [B,1,10,10]
    lead_shape = S_coarse.shape[:-2]
    B = int(torch.tensor(lead_shape).prod().item()) if len(lead_shape) > 0 else 1
    S = S_coarse.reshape(B, 1, 10, 10)

    # Smooth on coarse grid (depthwise conv)
    if sigma > 0:
        k = _gaussian_kernel2d(float(sigma), truncate=float(truncate), device=device, dtype=dtype)
        pad = (k.shape[-1] // 2, k.shape[-1] // 2, k.shape[-2] // 2, k.shape[-2] // 2)  # (l,r,t,b)
        S = F.pad(S, pad, mode="replicate")
        S = F.conv2d(S, k)

    # Upsample to fine grid
    H, W = int(out_hw[0]), int(out_hw[1])
    align = False if mode in ("bilinear", "bicubic") else None
    S_up = F.interpolate(S, size=(H, W), mode=mode, align_corners=align)

    return S_up.reshape(*lead_shape, H, W)


# =============================================================================
# Rollout
# =============================================================================

def rollout_pollution(
    S_unknown: torch.Tensor,               # [..., 10, 10] (coarse unknown source grid to infer)
    grid: PolGrid,
    params: PolParams,
    dt: float,
    steps: int,
    save_every: int = 1,
    enforce_cfl: bool = False,
    cfl: float = 0.45,
    source_mode: Literal["add", "softplus"] = "add",
    U0: Optional[torch.Tensor] = None,                      # [..., Nx, Ny]
    softplus_beta: Optional[float] = None,
    wind_seed: Optional[int] = None,
    no_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Simulate the PDE for a given unknown source (provided as a 10x10 coarse grid).

    Returns a dict with:
      - "U": concentration snapshots [..., Nx, Ny, Tsave]
      - "t": times [Tsave]
      - "Vx", "Vy": wind series [steps]
    """
    if grid.S_known is None:
        raise ValueError("grid.S_known is None. Create grid with load_sources=True or set S_known explicitly.")


    # Enforce unknown-source parameterization: S_unknown must be a 10x10 coarse grid (optionally batched)
    if tuple(S_unknown.shape[-2:]) != (10, 10):
        raise AssertionError(f"S_unknown must have shape [..., 10, 10]; got {tuple(S_unknown.shape)}")

    if softplus_beta is None:
        softplus_beta = float(params.sim_seconds_per_day) * 0.0 + 1.0  # default 1.0

    # Determine device/dtype from provided U0, otherwise from grid tensors
    if U0 is not None:
        device, dtype = U0.device, U0.dtype
    else:
        device = torch.device("cpu") if grid.x is None else grid.x.device
        dtype = torch.float32 if grid.x is None else grid.x.dtype

    # Build U0 internally from govdata via kriging if not provided
    if U0 is None:
        if grid.U0 is None:
            U0_built, _meta = build_U0_from_govdata_kriging(grid, device=device, dtype=dtype)
            grid.U0 = U0_built
        U0 = grid.U0.to(device=device, dtype=dtype)

    # ---- Batch-broadcast U0 to match S_unknown leading dims ----
    # S_unknown: [...,10,10]
    batch_shape = tuple(S_unknown.shape[:-2])  # () if unbatched, (B,) if batched
    if batch_shape:
        # U0 is expected to be [...,Nx,Ny] or [Nx,Ny]
        if U0.ndim == 2:
            # [Nx,Ny] -> [...,Nx,Ny]
            U0 = U0.expand(*batch_shape, *U0.shape).contiguous()
        elif U0.ndim == 2 + len(batch_shape):
            # already batched correctly
            pass
        else:
            raise RuntimeError(
                f"U0 has shape {tuple(U0.shape)} but expected [Nx,Ny] or {batch_shape}+[Nx,Ny]"
            )
    # ----------------------------------------------------------

    # Time vector for wind model
    t_full = torch.arange(0, steps, device=device, dtype=dtype) * float(dt)
    Vx_steps, Vy_steps = monsoon_wind_series(
        t_seconds=t_full,
        Vx_base=float(params.Vx0),
        Vy_base=float(params.Vy0),
        sim_seconds_per_day=float(params.sim_seconds_per_day),
        diurnal_amp_frac=float(params.diurnal_amp_frac),
        ar1_rho=float(params.ar1_rho),
        ar1_sigma_frac=float(params.ar1_sigma_frac),
        seed=wind_seed,
    )

    # Optional CFL check using a conservative bound:
    # For advection-diffusion, a simple check is dt <= cfl * min(dx/|Vx|, dy/|Vy|).
    if enforce_cfl:
        vmax = float(torch.max(torch.stack([torch.max(torch.abs(Vx_steps)), torch.max(torch.abs(Vy_steps)), torch.tensor(params.eps, device=device, dtype=dtype)])).item())
        dt_max = float(cfl) * min(grid.dx, grid.dy) / vmax
        if dt > dt_max:
            raise ValueError(f"CFL violation (advection): dt={dt:.3e} > dt_max={dt_max:.3e} (vmax={vmax:.3e}).")

    # Smooth + upsample unknown sources: [...,10,10] -> [...,Nx,Ny]
    S_unknown_fine = smooth_and_upsample_unknown(S_unknown.to(device=device, dtype=dtype), out_hw=(grid.Nx, grid.Ny), sigma=1.0)

    # Combine sources (broadcast known to batch if needed)
    S_total = combine_sources(grid.S_known.to(device=device, dtype=dtype), S_unknown_fine, mode=source_mode, softplus_beta=float(softplus_beta))

    # Heun / RK2 time stepping
    def _simulate() -> Dict[str, torch.Tensor]:
        U = U0
        snaps = []
        times = []
        for n in range(steps):
            if (n % save_every) == 0:
                snaps.append(U)
                times.append(t_full[n])

            Vx_t = Vx_steps[n]
            Vy_t = Vy_steps[n]
            k = params.k.to(device=device, dtype=dtype)

            f0 = rhs(U, Vx_t, Vy_t, k, S_total, grid.dx, grid.dy)
            f0 = torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)

            U_star = U + float(dt) * f0
            f1 = rhs(U_star, Vx_t, Vy_t, k, S_total, grid.dx, grid.dy)
            f1 = torch.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

            U = U + float(dt) * 0.5 * (f0 + f1)

        # last snapshot if steps not aligned with save_every
        if ((steps - 1) % save_every) != 0:
            snaps.append(U)
            times.append(t_full[-1])

        U_snap = torch.stack(snaps, dim=-1)  # [...,Nx,Ny,Tsave]
        t_snap = torch.stack(times, dim=0)   # [Tsave]
        return {"U": U_snap, "t": t_snap, "Vx": Vx_steps, "Vy": Vy_steps}

    if no_grad:
        with torch.no_grad():
            return _simulate()
    return _simulate()


__all__ = [
    "PolGrid",
    "PolParams",
    "make_grid",
    "load_known_sources_40x40",
    "monsoon_wind_series",
    "smooth_and_upsample_unknown",
    "rollout_pollution",
]

# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    """
    Lightweight sanity test for the pollution simulator.

    Verifies:
    - rollout runs without error
    - shapes are correct
    - concentration stays finite and non-negative
    """
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    dtype = torch.float32

    # Grid
    Nx = Ny = 40
    Lx = Ly = 1.0
    grid = make_grid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, dtype=dtype)

    # Unknown source field (what STAMP would infer): 10x10 coarse grid
    S_unknown = 0.1 * torch.rand((10, 10), device=device, dtype=dtype)

    # Wind forcing
    steps = 50
    dt = 0.01

    # Parameters
    params = PolParams(
        k=torch.tensor(0.01),
    )

    # Rollout
    out = rollout_pollution(
        grid=grid,
        params=params,
        dt=dt,
        steps=steps,
        S_unknown=S_unknown,
        save_every=1,
        no_grad=True,
    )

    # Basic checks
    assert "U" in out, "Output must contain concentration field 'c'"
    c = out["U"]  # [Nx,Ny,T] or [T,Nx,Ny] depending on your implementation

    assert torch.isfinite(c).all(), "NaNs or Infs detected in concentration"
    assert (c >= 0).all(), "Negative concentration detected"

    print("[polsim self-test] PASSED")
    print(f"  c.shape = {tuple(c.shape)}")
    print(f"  c.max   = {float(c.max()):.4e}")
    print(f"  c.mean  = {float(c.mean()):.4e}")


if __name__ == "__main__":
    _self_test()



# In[ ]:




