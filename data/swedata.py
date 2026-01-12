#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
data/swedata.py

STAMP / FieldFormer-style dataset generator for 2D nonlinear Shallow Water Equations (SWE)
under periodic boundary conditions, using sim/swesim.py (Rusanov flux).

Key design choices (matching our discussion):
- Nonlinear conservative SWE state q = [h, hu, hv]
- Partial observability by default: sensors observe HEIGHT ONLY (h)
- Unknown physics parameter: friction c_f
    * cf_mode="scalar":  c_f = softplus(cf0)
    * cf_mode="basis":   c_f(x,y) = softplus(cf0 + sum_k a_k * phi_k(x,y))
      with K~16 basis coefficients (mid-dimensional inverse problem)

Saved .npz keys (mirrors heatdata.py as closely as possible):
  x, y, t,
  params, param_names,
  bc, rng_seed,
  sensors_idx, sensors_xy,
  sensor_clean, sensor_noisy,
  misalign_eps_idx, misalign_mode,
  noise_mode, noise_div, noise_std,

SWE-specific (explicit + useful for training / evaluation):
  cf_mode, cf0, cf_a, softplus_beta, h_floor,
  zb (if used), cf_basis (if used),
  field_kind ("h"),
  (optional) field (full h field) if SAVE_FIELD=True

Run:
  python data/swedata.py

Notes:
- This script assumes ONLY the nonlinear swesim.py API:
    rollout_swe(q0, grid, params, dt, steps, ...)
  (No legacy linearized API support.)
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Allow "from sim.swesim import ..." when running from data/
sys.path.append(os.path.abspath(".."))

from sim.swesim import SWEParams, make_grid, rollout_swe, check_cfl_explicit


# In[ ]:


# =============================================================================
# Config (edit as needed)
# =============================================================================

# domain / discretization
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64

T_final = 5.0
Nt = 10000  # number of saved time points INCLUDING t=0; steps = Nt-1
SAVE_EVERY = 1

# physical constants
g = 9.81
H0 = 1.0

# bathymetry (known, optional)
USE_BATHYMETRY = True
BATHY_AMP = 0.05
BATHY_KX = 2
BATHY_KY = 3

# friction unknown
CF_MODE = "basis"          # "none" | "scalar" | "basis"
CF_K = 16                  # basis size for cf_mode="basis"
CF0_UNCONSTRAINED = -3.0   # softplus(cf0) -> small positive friction
CF_A_STD = 0.5             # std of cf_a coefficients in unconstrained space
SOFTPLUS_BETA = 1.0
H_FLOOR = 1e-4

# sensors / noise
SENSORS = 20
RNG_SEED = 42

# sensor misalignment (index-space, periodic)
MISALIGN_EPS_IDX = 0.3
MISALIGN_MODE = "uniform"  # "uniform" | "normal" | "none"

# noise
NOISE_MODE = "max"   # "std" | "max"
NOISE_DIV = 10.0

# torch
DEVICE = "cpu"
DTYPE = torch.float32

# saving
SAVE_PATH = "swe_dataset.npz"
SAVE_FIELD = False         # if True, save full h field over time (can be big)
SAVE_BASIS = True          # save cf_basis in npz (handy for debugging / baselines)
SAVE_BATHY = True          # save zb if used


# In[ ]:


# =============================================================================
# Utilities (kept similar to heatdata.py patterns)
# =============================================================================

def sample_sensors_uniform_unique(Nx: int, Ny: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """Unique uniform sensor locations, returned as int32 indices [S,2] (i,j)."""
    if S > Nx * Ny:
        raise ValueError(f"S={S} cannot exceed Nx*Ny={Nx*Ny}.")
    all_ij = np.array([(i, j) for i in range(Nx) for j in range(Ny)], dtype=np.int32)
    sel = rng.choice(all_ij.shape[0], size=S, replace=False)
    return all_ij[sel]


def sample_sensor_misalignment(
    S: int,
    rng: np.random.Generator,
    eps_idx: float = 0.0,
    mode: str = "uniform",
) -> np.ndarray:
    """Misalignment offsets in index units, shape [S,2] float32, periodic sampled later."""
    if eps_idx <= 0.0 or mode == "none":
        return np.zeros((S, 2), dtype=np.float32)

    if mode == "uniform":
        delta = rng.uniform(-eps_idx, eps_idx, size=(S, 2)).astype(np.float32)
    elif mode == "normal":
        delta = rng.normal(0.0, eps_idx / 2.0, size=(S, 2)).astype(np.float32)
        delta = np.clip(delta, -eps_idx, eps_idx).astype(np.float32)
    else:
        raise ValueError("MISALIGN_MODE must be 'uniform', 'normal', or 'none'.")

    return delta


def _bilinear_sample_frame_periodic(frame: np.ndarray, ij_float: np.ndarray) -> np.ndarray:
    """
    Bilinear sample a 2D frame [Nx,Ny] at fractional index coords ij_float [S,2],
    with periodic wrap.
    """
    Nx, Ny = frame.shape
    i = ij_float[:, 0]
    j = ij_float[:, 1]

    i0 = np.floor(i).astype(np.int64)
    j0 = np.floor(j).astype(np.int64)
    di = (i - i0).astype(np.float32)
    dj = (j - j0).astype(np.float32)

    i1 = i0 + 1
    j1 = j0 + 1

    i0 = np.mod(i0, Nx); i1 = np.mod(i1, Nx)
    j0 = np.mod(j0, Ny); j1 = np.mod(j1, Ny)

    f00 = frame[i0, j0]
    f10 = frame[i1, j0]
    f01 = frame[i0, j1]
    f11 = frame[i1, j1]

    w00 = (1 - di) * (1 - dj)
    w10 = di * (1 - dj)
    w01 = (1 - di) * dj
    w11 = di * dj

    return (w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11).astype(np.float32)


def extract_sensor_series(field: np.ndarray, sensors_idx: np.ndarray, delta_ij: Optional[np.ndarray]) -> np.ndarray:
    """
    field: [Nx,Ny,T]
    sensors_idx: [S,2] int
    delta_ij: [S,2] float misalignment in index units (optional)
    returns: [S,T]
    """
    if field.ndim != 3:
        raise ValueError(f"field must be [Nx,Ny,T], got {field.shape}")
    Nx, Ny, T = field.shape
    S = sensors_idx.shape[0]

    ij = sensors_idx.astype(np.float32)
    if delta_ij is not None:
        ij = ij + delta_ij.astype(np.float32)

    out = np.zeros((S, T), dtype=np.float32)
    for t in range(T):
        out[:, t] = _bilinear_sample_frame_periodic(field[:, :, t], ij)
    return out


def add_noise(sensor_clean: np.ndarray, rng: np.random.Generator, mode: str, div: float) -> Tuple[np.ndarray, float]:
    """Gaussian noise with scale determined by mode; returns (noisy, noise_std)."""
    if div <= 0:
        raise ValueError("NOISE_DIV must be > 0.")
    if mode == "std":
        sigma = float(np.std(sensor_clean))
    elif mode == "max":
        sigma = float(np.max(np.abs(sensor_clean)))
    else:
        raise ValueError("NOISE_MODE must be 'std' or 'max'.")
    noise_std = sigma / float(div)
    noisy = sensor_clean + rng.normal(scale=noise_std, size=sensor_clean.shape).astype(np.float32)
    return noisy.astype(np.float32), float(noise_std)


# In[ ]:


# =============================================================================
# SWE-specific: bathymetry + basis + ICs
# =============================================================================

def make_bathymetry(x: np.ndarray, y: np.ndarray, amp: float, kx: int, ky: int) -> np.ndarray:
    """Smooth periodic bathymetry z_b(x,y) as a sinusoidal product, shape [Nx,Ny]."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    # x in [0,L), use L = max(x)+dx
    Lx = float(x[-1] + (x[1] - x[0]))
    Ly = float(y[-1] + (y[1] - y[0]))
    zb = amp * (np.sin(2.0 * np.pi * kx * X / Lx) * np.sin(2.0 * np.pi * ky * Y / Ly))
    return zb.astype(np.float32)


def make_rbf_basis(x: np.ndarray, y: np.ndarray, K: int, lengthscale: float) -> np.ndarray:
    """
    RBF basis phi_k(x,y) with centers on a coarse grid; returns [K,Nx,Ny].
    """
    Nx = x.shape[0]; Ny = y.shape[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    nside = int(np.ceil(np.sqrt(K)))
    cx = np.linspace(0.0, float(x[-1] + (x[1] - x[0])), nside, endpoint=False)
    cy = np.linspace(0.0, float(y[-1] + (y[1] - y[0])), nside, endpoint=False)

    centers = [(cx[i], cy[j]) for i in range(nside) for j in range(nside)]
    centers = centers[:K]

    ls2 = float(lengthscale) ** 2
    basis = np.zeros((K, Nx, Ny), dtype=np.float32)
    for k, (xk, yk) in enumerate(centers):
        phi = np.exp(-((X - xk) ** 2 + (Y - yk) ** 2) / (2.0 * ls2))
        phi = phi / (np.max(phi) + 1e-12)
        basis[k] = phi.astype(np.float32)
    return basis


def gaussian_eta0(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    """Initial surface displacement eta0 (Gaussian bump), shape [Nx,Ny]."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta0 = np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2)))
    return eta0.astype(np.float32)


# In[ ]:


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    device = torch.device(DEVICE)

    # steps + dt
    if Nt < 2:
        raise ValueError("Nt must be >= 2 (includes t=0).")
    steps = int(Nt - 1)
    dt = float(T_final) / float(steps)

    # time stamps for saved frames (every SAVE_EVERY steps)
    t_np = (np.arange(0, steps + 1, SAVE_EVERY, dtype=np.float32) * np.float32(dt))

    # Build known bathymetry (optional) and basis (optional)
    x_np = np.linspace(0.0, float(Lx), int(Nx), endpoint=False, dtype=np.float32)
    y_np = np.linspace(0.0, float(Ly), int(Ny), endpoint=False, dtype=np.float32)

    zb_np = None
    if USE_BATHYMETRY:
        zb_np = make_bathymetry(x_np, y_np, amp=BATHY_AMP, kx=BATHY_KX, ky=BATHY_KY)
        zb_t = torch.as_tensor(zb_np, device=device, dtype=DTYPE)
    else:
        zb_t = None

    cf_basis_np = None
    if CF_MODE == "basis":
        cf_basis_np = make_rbf_basis(x_np, y_np, K=CF_K, lengthscale=0.2 * float(min(Lx, Ly)))
        cf_basis_t = torch.as_tensor(cf_basis_np, device=device, dtype=DTYPE)
    else:
        cf_basis_t = None

    # Construct grid (swesim.make_grid validates shapes)
    grid = make_grid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, dtype=DTYPE, zb=zb_t, cf_basis=cf_basis_t)

    # Overwrite x/y from grid (authoritative)
    x_np = grid.x.detach().cpu().numpy().astype(np.float32)
    y_np = grid.y.detach().cpu().numpy().astype(np.float32)

    # True friction parameters (unconstrained; simulator uses softplus)
    cf0 = None
    cf_a = None
    if CF_MODE == "none":
        pass
    elif CF_MODE == "scalar":
        cf0 = torch.tensor(float(CF0_UNCONSTRAINED), device=device, dtype=DTYPE)
    elif CF_MODE == "basis":
        cf0 = torch.tensor(float(CF0_UNCONSTRAINED), device=device, dtype=DTYPE)
        cf_a_np = rng.normal(loc=0.0, scale=float(CF_A_STD), size=(CF_K,)).astype(np.float32)
        cf_a = torch.as_tensor(cf_a_np, device=device, dtype=DTYPE)
    else:
        raise ValueError("CF_MODE must be 'none', 'scalar', or 'basis'.")

    # Params
    params = SWEParams(
        g=torch.tensor(float(g), device=device, dtype=DTYPE),
        H=torch.tensor(float(H0), device=device, dtype=DTYPE),
        cf_mode=CF_MODE,
        cf0=cf0,
        cf_a=cf_a,
        softplus_beta=float(SOFTPLUS_BETA),
        h_floor=float(H_FLOOR),
    )

    # Initial conditions
    eta0_np = gaussian_eta0(x_np, y_np, x0=0.5 * Lx, y0=0.5 * Ly, sigma=0.08 * float(min(Lx, Ly)))
    eta0 = torch.as_tensor(eta0_np, device=device, dtype=DTYPE)

    h0 = params.H + eta0
    hu0 = torch.zeros_like(h0)
    hv0 = torch.zeros_like(h0)

    # q0 must be [...,3,Nx,Ny]; we use [3,Nx,Ny]
    q0 = torch.stack([h0, hu0, hv0], dim=0)

    # CFL check (optional guard)
    try:
        # Optional safety margin: use a slightly smaller effective CFL
        cfl_eff = 0.95 * 0.45
        check_cfl_explicit(q=q0, grid=grid, params=params, dt=dt, cfl=cfl_eff)
    except Exception as e:
        print(f"[WARN] CFL check skipped/failed: {e}")

    # Rollout
    out = rollout_swe(
        q0=q0,
        grid=grid,
        params=params,
        dt=dt,
        steps=steps,
        save_every=SAVE_EVERY,
        enforce_cfl=True,
        cfl=0.45,
        return_primitive=True,
        no_grad=True,
    )

    # Height-like field for sensors
    if "h" not in out:
        raise RuntimeError("rollout_swe(return_primitive=True) must return key 'h'.")
    h_traj = out["h"]  # [T, ..., Nx, Ny] or [T,Nx,Ny]

    # Move to numpy as [Nx,Ny,T]
    h_np = h_traj.detach().cpu().numpy().astype(np.float32)
    if h_np.ndim == 4:
        # [T, B?, Nx, Ny] not expected; if [T, ..., Nx, Ny], flatten leading dims
        # common is [T, 3, Nx, Ny] for q, but for h should be [T, Nx, Ny]
        # if it is [T,1,Nx,Ny], squeeze channel
        if h_np.shape[1] == 1:
            h_np = h_np[:, 0, :, :]
        else:
            raise ValueError(f"Unexpected h array shape: {h_np.shape}")
    # now [T,Nx,Ny] -> transpose to [Nx,Ny,T]
    if h_np.ndim != 3:
        raise ValueError(f"Unexpected h array shape: {h_np.shape}")
    h_np = np.transpose(h_np, (1, 2, 0))  # [Nx,Ny,T]

    # Sensors
    sensors_idx = sample_sensors_uniform_unique(Nx, Ny, SENSORS, rng=rng)
    sensors_xy = np.stack([x_np[sensors_idx[:, 0]], y_np[sensors_idx[:, 1]]], axis=1).astype(np.float32)

    delta_ij = sample_sensor_misalignment(SENSORS, rng=rng, eps_idx=MISALIGN_EPS_IDX, mode=MISALIGN_MODE)
    sensor_clean = extract_sensor_series(h_np, sensors_idx, delta_ij=delta_ij)
    sensor_noisy, noise_std = add_noise(sensor_clean, rng=rng, mode=NOISE_MODE, div=NOISE_DIV)

    # Params metadata (numeric array, like heatdata)
    # Encode cf_mode as int for compact numeric params: none=0, scalar=1, basis=2
    cf_mode_int = {"none": 0, "scalar": 1, "basis": 2}[CF_MODE]
    dx = float(grid.dx); dy = float(grid.dy)

    param_names = np.array(
        ["g", "H0", "Lx", "Ly", "Nx", "Ny", "steps", "dt", "dx", "dy", "cf_mode_int", "cf_K", "cf0"],
        dtype="<U16",
    )

    cf0_val = float(CF0_UNCONSTRAINED) if CF_MODE != "none" else 0.0
    params_arr = np.array(
        [g, H0, Lx, Ly, Nx, Ny, steps, dt, dx, dy, cf_mode_int, int(CF_K), cf0_val],
        dtype=np.float64,
    )

    # Save
    save_kwargs = dict(
        x=x_np,
        y=y_np,
        t=t_np,
        params=params_arr,
        param_names=param_names,
        bc=np.array(["periodic"]),
        rng_seed=np.array([RNG_SEED], dtype=np.int64),
        sensors_idx=sensors_idx.astype(np.int32),
        sensors_xy=sensors_xy,
        sensor_clean=sensor_clean.astype(np.float32),
        sensor_noisy=sensor_noisy.astype(np.float32),
        misalign_eps_idx=np.array([MISALIGN_EPS_IDX], dtype=np.float32),
        misalign_mode=np.array([MISALIGN_MODE]),
        noise_mode=np.array([NOISE_MODE]),
        noise_div=np.array([NOISE_DIV], dtype=np.float32),
        noise_std=np.array([noise_std], dtype=np.float32),
        field_kind=np.array(["h"]),
        # SWE paramization (explicit)
        cf_mode=np.array([CF_MODE]),
        softplus_beta=np.array([SOFTPLUS_BETA], dtype=np.float32),
        h_floor=np.array([H_FLOOR], dtype=np.float32),
    )

    if CF_MODE != "none":
        save_kwargs["cf0"] = np.array([float(CF0_UNCONSTRAINED)], dtype=np.float32)
    if CF_MODE == "basis":
        save_kwargs["cf_a"] = cf_a.detach().cpu().numpy().astype(np.float32)
        save_kwargs["cf_K"] = np.array([CF_K], dtype=np.int32)

    if SAVE_FIELD:
        save_kwargs["field"] = h_np.astype(np.float32)

    if SAVE_BATHY and zb_np is not None:
        save_kwargs["zb"] = zb_np.astype(np.float32)

    if SAVE_BASIS and cf_basis_np is not None:
        save_kwargs["cf_basis"] = cf_basis_np.astype(np.float32)

    np.savez_compressed(SAVE_PATH, **save_kwargs)

    print(f"[SAVE] {SAVE_PATH}")
    print(f"  grid: Nx={Nx}, Ny={Ny}, dt={dt:.4g}, steps={steps}, save_every={SAVE_EVERY}, T={t_np.shape[0]}")
    print(f"  sensors: {SENSORS}, noise_std={noise_std:.4g}, noise_mode={NOISE_MODE}, noise_div={NOISE_DIV}")
    print(f"  cf_mode={CF_MODE}, K={CF_K}, cf0(unconstrained)={CF0_UNCONSTRAINED}")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




