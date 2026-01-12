# data/poldata.py
from __future__ import annotations

import os
import sys

# Required import pattern (poldata in data/, polsim in sim/)
sys.path.append(os.path.abspath(".."))
import sim.polsim as polsim  # noqa: E402

import numpy as np
import torch


# -----------------------------
# Config (edit here)
# -----------------------------
SRC_DIR = os.path.abspath("..")          # project root (adjust if needed)
OUT_PATH = os.path.join(SRC_DIR, "data", "pol_dataset.npz")
SIM_DIR = os.path.join(SRC_DIR, "sim")

SEED = 0
DEVICE = "cpu"
DTYPE = torch.float32

# Time settings (FieldFormer pollution style)
T_FINAL = 20.0
NT = 10000
DT = T_FINAL / (NT - 1)

# Unknown-source specs
UNKNOWN_COARSE_HW = (10, 10)
UNKNOWN_TARGET_FRAC = 0.25
NORTH_SOUTH_RATIO = 1.5   # top row ≈ 1.5x bottom row (coarse grid)
UNKNOWN_SMOOTH_SIGMA = 1.0
UNKNOWN_UPSAMPLE_MODE = "bilinear"  # "bilinear" or "bicubic"

# Noise (FieldFormer style: sigma/max divided by NOISE_DIV)
NOISE_DIV = 10.0
NOISE_RNG_SEED = 42

# Delhi bounding box mapping for sensor locations (match your pollution.py)
LON_MIN, LON_MAX = 77.01, 77.40
LAT_MIN, LAT_MAX = 28.39, 28.78


# -----------------------------
# Helpers
# -----------------------------
def map_gov_locs_to_grid(
    locs_csv: str,
    *,
    Nx: int,
    Ny: int,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Map govdata sensor lon/lat to grid indices.

    Returns:
      sensors_idx: (S,2) int32 as (iy, ix)
      sensors_xy:  (S,2) float32 as (x, y) in [0,1]
    """
    import pandas as pd

    locs = pd.read_csv(locs_csv, index_col=0)

    sensor_indices = []
    for lon, lat in zip(locs["Longitude"].to_numpy(), locs["Latitude"].to_numpy()):
        x_norm = (float(lon) - lon_min) / (lon_max - lon_min + 1e-12)
        y_norm = (float(lat) - lat_min) / (lat_max - lat_min + 1e-12)
        ix = max(0, min(Nx - 1, int(round(x_norm * (Nx - 1)))))
        iy = max(0, min(Ny - 1, int(round(y_norm * (Ny - 1)))))
        sensor_indices.append((iy, ix))

    # de-duplicate, stable ordering
    sensor_locs = sorted(set(sensor_indices))
    sensors_idx = np.array(sensor_locs, dtype=np.int32)

    x = np.linspace(0.0, 1.0, Nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, Ny, dtype=np.float32)
    sensors_xy = np.stack([x[sensors_idx[:, 1]], y[sensors_idx[:, 0]]], axis=1).astype(np.float32)

    return sensors_idx, sensors_xy


def build_unknown_source_coarse(
    *,
    Kx: int = 10,
    Ky: int = 10,
    seed: int = 0,
    north_to_south_ratio: float = 1.5,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Map-like coarse pattern: north-high trend * soft footprint + a few smooth lobes.

    - Large-scale north>south trend (top ≈ ratio * bottom).
    - Soft elliptical mask to mimic a city footprint.
    - 2–4 anisotropic Gaussian lobes to mimic localized hotspot patches.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Coarse grid coords in [0,1]
    xs = torch.linspace(0.0, 1.0, Kx, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, Ky, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")  # X: (Kx,Ky), Y: (Kx,Ky)

    # --- 1) North>South base trend ---
    ratio = float(north_to_south_ratio)
    # Make bottom ~1, top ~ratio using a smooth power curve
    p = 1.4
    base = 1.0 + (ratio - 1.0) * (Y ** p)

    # --- 2) Soft footprint mask (tilted ellipse) ---
    # Center + axes; tweak to match your city footprint look
    xc, yc = 0.52, 0.55
    a, b = 0.55, 0.70

    # Small rotation to avoid axis-aligned symmetry
    theta = 0.12  # radians
    Xc = X - xc
    Yc = Y - yc
    Xr =  torch.cos(torch.tensor(theta, device=device, dtype=dtype)) * Xc + torch.sin(torch.tensor(theta, device=device, dtype=dtype)) * Yc
    Yr = -torch.sin(torch.tensor(theta, device=device, dtype=dtype)) * Xc + torch.cos(torch.tensor(theta, device=device, dtype=dtype)) * Yc

    ellipse = 1.0 - (Xr / a) ** 2 - (Yr / b) ** 2
    tau = 0.15  # softness
    mask = torch.sigmoid(ellipse / tau)

    # --- 3) A few anisotropic Gaussian lobes (localized patches) ---
    def gauss2d(x0, y0, sx, sy, w):
        return w * torch.exp(-0.5 * ((X - x0) / sx) ** 2 - 0.5 * ((Y - y0) / sy) ** 2)

    # Pick a small set of lobes (fixed locations feel “map-like”)
    # You can randomize weights mildly for dataset variety.
    w_jit = 0.15
    w1 = 1.0 + w_jit * torch.randn((), generator=g, device=device, dtype=dtype)
    w2 = 0.8 + w_jit * torch.randn((), generator=g, device=device, dtype=dtype)
    w3 = 0.6 + w_jit * torch.randn((), generator=g, device=device, dtype=dtype)

    lobes = (
        gauss2d(0.55, 0.75, 0.22, 0.18, w1) +   # north-central
        gauss2d(0.75, 0.45, 0.18, 0.22, w2) +   # east-ish patch
        gauss2d(0.35, 0.35, 0.25, 0.20, w3)     # southwest-ish patch
    )

    # Blend base trend and lobes
    alpha = 0.75  # mostly trend, some patches
    S = (alpha * base + (1.0 - alpha) * lobes) * mask

    # Mild x-smoothing to keep east-west variation small (optional)
    # 1D smoothing kernel across x
    kx = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=dtype).view(1, 1, 3, 1)
    S4 = S.view(1, 1, Kx, Ky)
    S4 = torch.nn.functional.pad(S4, (0, 0, 1, 1), mode="replicate")
    S4 = torch.nn.functional.conv2d(S4, kx)
    S = S4.view(Kx, Ky)

    return torch.clamp(S, min=0.0)




def smooth_and_upsample_unknown(
    S_coarse: torch.Tensor,
    *,
    out_hw: tuple[int, int],
    sigma: float,
    mode: str,
) -> torch.Tensor:
    """Call simulator's helper if present; else fall back to local implementation."""
    if hasattr(polsim, "smooth_and_upsample_unknown"):
        return polsim.smooth_and_upsample_unknown(S_coarse, out_hw=out_hw, sigma=sigma, mode=mode)

    # Fallback: Gaussian-like smoothing via separable conv + interpolate
    # (Only used if your polsim doesn't expose the helper.)
    Hc, Wc = S_coarse.shape[-2], S_coarse.shape[-1]
    x = torch.arange(Wc, device=S_coarse.device, dtype=S_coarse.dtype) - (Wc - 1) / 2
    y = torch.arange(Hc, device=S_coarse.device, dtype=S_coarse.dtype) - (Hc - 1) / 2
    X, Y = torch.meshgrid(x, y, indexing="xy")
    r2 = X**2 + Y**2
    k = torch.exp(-0.5 * r2 / (sigma**2 + 1e-12))
    k = k / (k.sum() + 1e-12)

    S4 = S_coarse.view(1, 1, Hc, Wc)
    k4 = k.view(1, 1, Hc, Wc)
    # "same-ish" smoothing: pad then conv with full kernel is expensive; instead do a small kernel if needed.
    # Here, we just skip extra smoothing if no helper exists.
    Su = torch.nn.functional.interpolate(S4, size=out_hw, mode=mode, align_corners=False if mode != "nearest" else None)
    return Su.view(out_hw[0], out_hw[1])


def scale_unknown_to_fraction_of_known(
    S_coarse: torch.Tensor,
    *,
    grid: polsim.PolGrid,
    target_frac: float,
    sigma: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Scale unknown so sum(S_unknown_fine) ~= target_frac * sum(S_known)."""
    if grid.S_known is None:
        raise ValueError("grid.S_known is None; expected known sources loaded inside polsim.")

    S_fine = smooth_and_upsample_unknown(S_coarse, out_hw=(grid.Nx, grid.Ny), sigma=sigma, mode=mode)

    known_sum = float(torch.sum(torch.clamp(grid.S_known, min=0.0)).item())
    unk_sum = float(torch.sum(torch.clamp(S_fine, min=0.0)).item())
    target = float(target_frac) * known_sum
    alpha = 0.0 if unk_sum <= 1e-12 else target / unk_sum

    return S_coarse * alpha, S_fine * alpha, float(alpha)


# -----------------------------
# Main (no argparse)
# -----------------------------
def generate_and_save() -> None:
    device = torch.device(DEVICE)

    # Build grid (expects polsim to handle loading/cropping known sources internally)
    # Adjust call signature if your polsim.make_grid differs.
    grid = polsim.make_grid(Nx=40, Ny=40, src_dir=SIM_DIR, device=device, dtype=DTYPE, load_sources=True)

    # Sensors from govdata
    locs_csv = os.path.join(SIM_DIR, "govdata_locations.csv")
    sensors_idx, sensors_xy = map_gov_locs_to_grid(locs_csv, Nx=grid.Nx, Ny=grid.Ny)

    # Unknown sources: coarse 10x10, enforce assertion
    S_coarse = build_unknown_source_coarse(
        Kx=UNKNOWN_COARSE_HW[0],
        Ky=UNKNOWN_COARSE_HW[1],
        seed=SEED,
        north_to_south_ratio=NORTH_SOUTH_RATIO,
        device=device,
        dtype=DTYPE,
    )
    assert tuple(S_coarse.shape) == UNKNOWN_COARSE_HW, (
        f"S_unknown_coarse must be {UNKNOWN_COARSE_HW}; got {tuple(S_coarse.shape)}"
    )

    # Scale to 25% (measured after smoothing+upsample)
    S_coarse, S_fine, alpha = scale_unknown_to_fraction_of_known(
        S_coarse,
        grid=grid,
        target_frac=UNKNOWN_TARGET_FRAC,
        sigma=UNKNOWN_SMOOTH_SIGMA,
        mode=UNKNOWN_UPSAMPLE_MODE,
    )

    # Diagnostics: coarse north/south ratio (mean over x)
    coarse_bottom = float(torch.mean(S_coarse[:, 0]).item())
    coarse_top = float(torch.mean(S_coarse[:, -1]).item())
    coarse_ratio_emp = coarse_top / (coarse_bottom + 1e-12)

    # Params: use your real-data calibrated base winds
    params = polsim.PolParams(
        k=torch.tensor(3e-4, device=device, dtype=DTYPE),
        Vx0=1.12,   # <-- remember: real-data calibrated
        Vy0=0.984,  # <-- remember: real-data calibrated
        sim_seconds_per_day=5.0,
        diurnal_amp_frac=0.5,
        ar1_rho=0.90,
        ar1_sigma_frac=0.15,
    )

    # Run simulator (expects S_unknown to be 10x10; polsim should smooth+upsample internally)
    out = polsim.rollout_pollution(
        S_unknown=S_coarse,
        grid=grid,
        params=params,
        dt=DT,
        steps=NT,
        save_every=1,
        enforce_cfl=False,
        wind_seed=123,
        no_grad=True,
    )
    U = out["U"]  # [Nx, Ny, Nt] (per your polsim)

    # Sample sensors (sensors_idx is (iy, ix); U is indexed as U[ix, iy, t] if U is [Nx,Ny,Nt]
    # If your U is [Ny,Nx,Nt], flip indexing accordingly.
    ix = torch.as_tensor(sensors_idx[:, 1], device=device, dtype=torch.long)
    iy = torch.as_tensor(sensors_idx[:, 0], device=device, dtype=torch.long)
    sensor_clean = U[ix, iy, :].detach().cpu().numpy().astype(np.float32)  # [S, Nt]

    # Add noise (FieldFormer-style)
    sigma = float(np.max(np.abs(sensor_clean)))
    noise_std = sigma / float(NOISE_DIV)
    rng = np.random.default_rng(NOISE_RNG_SEED)
    sensor_noisy = sensor_clean + rng.normal(scale=noise_std, size=sensor_clean.shape).astype(np.float32)

    # Save arrays
    U_np = U.detach().cpu().numpy().astype(np.float32)
    x = np.linspace(0.0, 1.0, grid.Nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, grid.Ny, dtype=np.float32)
    t = np.linspace(0.0, T_FINAL, NT, dtype=np.float32)

    known_sum = float(torch.sum(torch.clamp(grid.S_known, min=0.0)).item())
    unk_sum = float(torch.sum(torch.clamp(S_fine, min=0.0)).item())
    frac_emp = unk_sum / (known_sum + 1e-12)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        U=U_np,
        x=x, y=y, t=t,
        S_known=grid.S_known.detach().cpu().numpy().astype(np.float32),
        S_unknown_coarse=S_coarse.detach().cpu().numpy().astype(np.float32),
        S_unknown_fine=S_fine.detach().cpu().numpy().astype(np.float32),
        sensors_idx=sensors_idx.astype(np.int32),
        sensors_xy=sensors_xy.astype(np.float32),
        sensor_clean=sensor_clean,
        sensor_noisy=sensor_noisy,
        noise_mode=np.array(["max"]),
        noise_div=np.array([float(NOISE_DIV)], dtype=np.float32),
        noise_std=np.array([float(noise_std)], dtype=np.float32),
        unknown_target_frac=np.array([float(UNKNOWN_TARGET_FRAC)], dtype=np.float32),
        unknown_frac_empirical=np.array([float(frac_emp)], dtype=np.float32),
        unknown_scale_alpha=np.array([float(alpha)], dtype=np.float32),
        coarse_ratio_target=np.array([float(NORTH_SOUTH_RATIO)], dtype=np.float32),
        coarse_ratio_empirical=np.array([float(coarse_ratio_emp)], dtype=np.float32),
    )

    print(f"[SAVE] {OUT_PATH}")
    print(f"  U: {U_np.shape}  sensors: {sensor_clean.shape[0]}  T={T_FINAL} Nt={NT} dt={DT:.3e}")
    print(f"  unknown/known (sum) target={UNKNOWN_TARGET_FRAC:.3f} empirical={frac_emp:.3f} (alpha={alpha:.3g})")
    print(f"  coarse N/S ratio target={NORTH_SOUTH_RATIO:.3f} empirical={coarse_ratio_emp:.3f}")
    print(f"  noise_std={noise_std:.4g}")


# If you run this file directly (python poldata.py), it will generate the dataset.
# If you import it in a notebook, call generate_and_save().
if __name__ == "__main__":
    generate_and_save()
