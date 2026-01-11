#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
data/heatdata.py

Generate synthetic sensor dataset for 2D anisotropic heat equation (periodic BC)
using the canonical PyTorch simulator in sim/heat.py.

Saves an .npz compatible with the FieldFormer-style keys:
  u, x, y, t, params, param_names, bc, rng_seed,
  sensors_idx, sensors_xy, sensor_clean, sensor_noisy,
  noise_mode, noise_div, noise_std
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple
import sys
sys.path.append(os.path.abspath(".."))

from sim.heatsim import make_grid, HeatParams, rollout_heat, forcing_default, check_cfl_explicit


# In[2]:


# =============================================================================
# Config (edit me)
# =============================================================================
# PDE / forcing
alpha_x = 0.01
alpha_y = 0.001
A = 5.0

# domain / discretization
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64
T_final = 20.0
Nt = 10000

# sensors / noise
SENSORS = 20
NOISE_MODE = "max"   # "std" or "max"
NOISE_DIV = 10.0
RNG_SEED = 42
MISALIGN_EPS_IDX = 0.3
MISALIGN_MODE = "uniform"

# saving
SAVE_PATH = "heat_periodic_dataset.npz"

# torch
DEVICE = "cpu"       # "cpu" for determinism; "cuda" ok if you want speed
DTYPE = torch.float32


# In[3]:


# =============================================================================
# Helpers: sensor sampling + noise
# =============================================================================

def sample_sensors_uniform_unique(
    Nx: int,
    Ny: int,
    S: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns sensor indices as int32 array of shape [S, 2] with (i,j) grid indices.
    """
    if S > Nx * Ny:
        raise ValueError(f"S={S} cannot exceed Nx*Ny={Nx*Ny}.")

    all_ij = np.array([(i, j) for i in range(Nx) for j in range(Ny)], dtype=np.int32)
    sel_idx = rng.choice(all_ij.shape[0], size=S, replace=False)
    return all_ij[sel_idx]


def sample_sensor_misalignment(
    S: int,
    rng: np.random.Generator,
    eps_idx: float,
    mode: str = "uniform",
) -> np.ndarray:
    """
    Sample sub-cell sensor misalignment in *index space*.

    Returns:
      delta_ij: float32 array [S, 2] with (delta_i, delta_j), typically in [-eps_idx, eps_idx].

    Notes:
      - eps_idx is in grid-index units. eps_idx=0.5 means up to half a cell.
      - Use eps_idx <= 0.49 to avoid crossing many cells (still fine if it does; we clamp).
    """
    if eps_idx < 0.0:
        raise ValueError(f"eps_idx must be >= 0, got {eps_idx}.")
    if mode == "uniform":
        delta = rng.uniform(low=-eps_idx, high=eps_idx, size=(S, 2)).astype(np.float32)
    elif mode == "normal":
        # Interpret eps_idx as ~2σ so 95% within [-eps, eps]
        sigma = eps_idx / 2.0 if eps_idx > 0 else 0.0
        delta = rng.normal(loc=0.0, scale=sigma, size=(S, 2)).astype(np.float32)
        delta = np.clip(delta, -eps_idx, eps_idx)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
    return delta


def _bilinear_sample_frame(
    frame: np.ndarray,          # [Nx, Ny]
    pos_ij: np.ndarray,         # [S, 2] float (continuous i,j)
) -> np.ndarray:
    """
    Bilinear sample a single frame at continuous index positions.

    Returns:
      vals: [S] float32
    """
    Nx, Ny = frame.shape
    pi = pos_ij[:, 0]
    pj = pos_ij[:, 1]

    # Clamp to valid range so i0+1, j0+1 are in-bounds
    pi = np.clip(pi, 0.0, Nx - 1.001)
    pj = np.clip(pj, 0.0, Ny - 1.001)

    i0 = np.floor(pi).astype(np.int32)
    j0 = np.floor(pj).astype(np.int32)
    i1 = i0 + 1
    j1 = j0 + 1

    # Clamp neighbors
    i1 = np.clip(i1, 0, Nx - 1)
    j1 = np.clip(j1, 0, Ny - 1)

    wi = (pi - i0).astype(np.float32)
    wj = (pj - j0).astype(np.float32)

    v00 = frame[i0, j0].astype(np.float32)
    v10 = frame[i1, j0].astype(np.float32)
    v01 = frame[i0, j1].astype(np.float32)
    v11 = frame[i1, j1].astype(np.float32)

    # Bilinear interpolation
    vals = (1 - wi) * (1 - wj) * v00 + wi * (1 - wj) * v10 + (1 - wi) * wj * v01 + wi * wj * v11
    return vals.astype(np.float32)


def extract_sensor_series(
    u_traj: np.ndarray,                 # [Nt, Nx, Ny]
    sensors_idx: np.ndarray,            # [S, 2] int (i,j)
    delta_ij: Optional[np.ndarray] = None,  # [S, 2] float (delta_i, delta_j)
) -> np.ndarray:
    """
    Extract sensor series from u_traj.

    If delta_ij is None:
      - samples exactly at integer grid points (old behavior)

    If delta_ij is provided:
      - samples at continuous positions (i+delta_i, j+delta_j) with bilinear interpolation
      - models sub-cell sensor misalignment while still storing nominal integer indices

    Returns:
      sensor_clean: [S, Nt]
    """
    if u_traj.ndim != 3:
        raise ValueError(f"u_traj must have shape [Nt,Nx,Ny], got {u_traj.shape}.")
    Nt, Nx, Ny = u_traj.shape
    sensors_idx = sensors_idx.astype(np.int32)

    if delta_ij is None:
        ii = sensors_idx[:, 0]
        jj = sensors_idx[:, 1]
        return u_traj[:, ii, jj].T.astype(np.float32)  # [S, Nt]

    delta_ij = delta_ij.astype(np.float32)
    if delta_ij.shape != sensors_idx.shape:
        raise ValueError(f"delta_ij shape {delta_ij.shape} must match sensors_idx shape {sensors_idx.shape}.")

    pos_ij = sensors_idx.astype(np.float32) + delta_ij  # [S,2] continuous (i,j)

    # Loop over time (S is small; this is fast enough for S~20-100)
    out = np.empty((sensors_idx.shape[0], Nt), dtype=np.float32)
    for t in range(Nt):
        out[:, t] = _bilinear_sample_frame(u_traj[t], pos_ij)
    return out


def add_noise(sensor_clean: np.ndarray, rng: np.random.Generator, mode: str, div: float) -> tuple[np.ndarray, float]:
    """
    sensor_clean: [S, Nt]
    returns (sensor_noisy, noise_std)
    """
    if mode == "std":
        sigma = float(np.std(sensor_clean))
    elif mode == "max":
        sigma = float(np.max(np.abs(sensor_clean)))
    else:
        raise ValueError("NOISE_MODE must be 'std' or 'max'")

    noise_std = sigma / float(div)
    sensor_noisy = sensor_clean + rng.normal(scale=noise_std, size=sensor_clean.shape).astype(sensor_clean.dtype)
    return sensor_noisy, noise_std


# In[4]:


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    # RNG
    rng = np.random.default_rng(RNG_SEED)

    # Torch setup
    device = torch.device(DEVICE)

    # Time vector (match numpy: inclusive linspace)
    t_vec = torch.linspace(0.0, float(T_final), int(Nt), device=device, dtype=DTYPE)
    dt = float(t_vec[1] - t_vec[0])

    # Grid
    grid = make_grid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, dtype=DTYPE)

    # CFL check (same condition as generator)
    check_cfl_explicit(alpha_x=alpha_x, alpha_y=alpha_y, dx=grid.dx, dy=grid.dy, dt=dt)

    # Initial condition (match FieldFormer generator)
    u0 = torch.sin(2.0 * torch.pi * grid.X) * torch.sin(2.0 * torch.pi * grid.Y)  # [Nx,Ny]

    # Params (as torch tensors)
    params = HeatParams(
        alpha_x=torch.tensor(alpha_x, device=device, dtype=DTYPE),
        alpha_y=torch.tensor(alpha_y, device=device, dtype=DTYPE),
        A=torch.tensor(A, device=device, dtype=DTYPE),
    )

    # Rollout using canonical simulator (no gradients)
    out = rollout_heat(
        u0=u0,
        params=params,
        grid=grid,
        t_vec=t_vec,
        forcing_fn=forcing_default,
        forcing_kwargs={"T_final": T_final},
        save_every=1,
        no_grad=True,
    )
    u = out["u"]  # [Nt, Nx, Ny] or [Nsaved, Nx, Ny] depending on save_every
    t_saved = out["t"]

    # Convert to numpy for saving (FieldFormer expects float32)
    u_np = u.detach().cpu().numpy().astype(np.float32)
    x_np = grid.x.detach().cpu().numpy().astype(np.float32)
    y_np = grid.y.detach().cpu().numpy().astype(np.float32)
    t_np = t_saved.detach().cpu().numpy().astype(np.float32)

    # Sensor sampling (grid indices + physical coords)
    sensors_idx = sample_sensors_uniform_unique(Nx, Ny, SENSORS, rng=rng)  # [S,2]
    sensors_xy = np.stack([x_np[sensors_idx[:, 0]], y_np[sensors_idx[:, 1]]], axis=1).astype(np.float32)

    # Sensor time series
    # Note: if save_every != 1, then "Nt" here is Nsaved. Still consistent if we save t accordingly.
    delta_ij = sample_sensor_misalignment(SENSORS, rng, eps_idx=MISALIGN_EPS_IDX, mode=MISALIGN_MODE)
    sensor_clean = extract_sensor_series(u_np, sensors_idx, delta_ij=delta_ij).astype(np.float32)  # [S, Nt_saved]
    sensor_noisy, noise_std = add_noise(sensor_clean, rng=rng, mode=NOISE_MODE, div=NOISE_DIV)
    sensor_noisy = sensor_noisy.astype(np.float32)

    # Params block (FieldFormer-compatible)
    param_names = np.array(
        ["alpha_x", "alpha_y", "A", "T", "Nx", "Ny", "Nt", "Lx", "Ly", "dx", "dy", "dt"],
        dtype="<U16",
    )
    params_arr = np.array(
        [alpha_x, alpha_y, A, T_final, Nx, Ny, int(t_np.shape[0]), Lx, Ly, grid.dx, grid.dy, dt],
        dtype=np.float64,
    )

    # Save
    np.savez_compressed(
        SAVE_PATH,
        x=x_np,
        y=y_np,
        t=t_np,
        params=params_arr,
        param_names=param_names,
        bc=np.array(["periodic"]),
        rng_seed=np.array([RNG_SEED], dtype=np.int64),
        sensors_idx=sensors_idx.astype(np.int32),
        sensors_xy=sensors_xy,
        sensor_clean=sensor_clean,
        sensor_noisy=sensor_noisy,
        misalign_eps_idx=np.array([MISALIGN_EPS_IDX], dtype=np.float32),
        misalign_mode=np.array([MISALIGN_MODE]),
        noise_mode=np.array([NOISE_MODE]),
        noise_div=np.array([NOISE_DIV], dtype=np.float32),
        noise_std=np.array([noise_std], dtype=np.float32),
    )

    print(f"[SAVE] Wrote dataset to: {SAVE_PATH}")
    print(f"      u shape: {u_np.shape}, sensors: {SENSORS}, noise σ ≈ {noise_std:.4g}")


# In[5]:


if __name__ == "__main__":
    main()


# In[ ]:




