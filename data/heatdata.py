"""
data/gen_heat.py

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

from sim.heat import make_grid, HeatParams, rollout_heat, forcing_default, check_cfl_explicit


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

# saving
SAVE_PATH = "heat_periodic_dataset.npz"

# torch
DEVICE = "cpu"       # "cpu" for determinism; "cuda" ok if you want speed
DTYPE = torch.float32


# =============================================================================
# Helpers: sensor sampling + noise
# =============================================================================
def sample_sensors_uniform_unique(Nx: int, Ny: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns sensor indices as int32 array of shape [S, 2] with (i,j) grid indices.
    """
    all_ij = np.array([(i, j) for i in range(Nx) for j in range(Ny)], dtype=np.int32)
    sel_idx = rng.choice(all_ij.shape[0], size=S, replace=False)
    return all_ij[sel_idx]


def extract_sensor_series(u_traj: np.ndarray, sensors_idx: np.ndarray) -> np.ndarray:
    """
    u_traj: [Nt, Nx, Ny] numpy
    sensors_idx: [S,2] (i,j)
    returns sensor_clean: [S, Nt]
    """
    ii = sensors_idx[:, 0]
    jj = sensors_idx[:, 1]
    # gather then transpose to [S, Nt]
    return u_traj[:, ii, jj].T


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
    sensor_clean = extract_sensor_series(u_np, sensors_idx).astype(np.float32)  # [S, Nt_saved]
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
        u=u_np,                # [Nt_saved, Nx, Ny]
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
        noise_mode=np.array([NOISE_MODE]),
        noise_div=np.array([NOISE_DIV], dtype=np.float32),
        noise_std=np.array([noise_std], dtype=np.float32),
    )

    print(f"[SAVE] Wrote dataset to: {SAVE_PATH}")
    print(f"      u shape: {u_np.shape}, sensors: {SENSORS}, noise σ ≈ {noise_std:.4g}")


if __name__ == "__main__":
    main()
