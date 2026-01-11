#!/usr/bin/env python
# coding: utf-8
"""
sim/swesim.py

Differentiable (PyTorch) simulator for 2D nonlinear shallow-water equations (SWE)
with periodic boundary conditions.

We simulate the depth-averaged, inviscid SWE in conservative form with optional
bathymetry forcing and quadratic bottom friction:

State (conservative):
  q = [h, hu, hv]

Fluxes:
  F_x(q) = [ hu,
             hu*u + 0.5*g*h^2,
             hu*v ]
  F_y(q) = [ hv,
             hv*u,
             hv*v + 0.5*g*h^2 ]

where u = (hu/h), v = (hv/h).

Sources:
  Bathymetry (known z_b(x,y)):
    S_b = [0, -g*h*∂x z_b, -g*h*∂y z_b]

  Quadratic friction (unknown c_f, scalar or low-dim field):
    speed = sqrt(u^2 + v^2)
    S_f = [0, -c_f*h*u*speed, -c_f*h*v*speed]

Time stepping:
  First-order explicit finite-volume update using Rusanov (local Lax-Friedrichs) flux.
  Periodic BC via torch.roll.

Design goals (aligned with heatsim.py):
- Single canonical discretization used by data generation and calibration.
- Batched simulation and optional trajectory saving.
- Hook for moderate-dimensional parameterization of c_f(x,y) via fixed basis.

Notes:
- This is intentionally a "clean & stable" baseline scheme (first order). If you later
  want sharper wave fronts, we can add MUSCL reconstruction / TVD limiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
from contextlib import nullcontext
import torch


# =============================================================================
# Grid / params containers
# =============================================================================

@dataclass
class SWEGrid:
    # spatial grids (1D) and spacings
    x: torch.Tensor          # [Nx]
    y: torch.Tensor          # [Ny]
    dx: float
    dy: float

    # meshgrids for convenience
    X: torch.Tensor          # [Nx, Ny]
    Y: torch.Tensor          # [Nx, Ny]

    # known bathymetry (optional)
    zb: Optional[torch.Tensor] = None   # [Nx, Ny]

    # fixed basis for friction field (optional)
    cf_basis: Optional[torch.Tensor] = None  # [K, Nx, Ny]


@dataclass
class SWEParams:
    # gravity and mean depth (can be scalar tensors, optionally learnable)
    g: torch.Tensor                  # scalar
    H: torch.Tensor                  # scalar (mean depth used for h = H + eta convenience)

    # friction parameterization
    cf_mode: Literal["none", "scalar", "basis"] = "none"

    # scalar friction: c_f = softplus(cf0)
    cf0: Optional[torch.Tensor] = None  # scalar or [B]

    # basis friction: c_f(x,y) = softplus(cf0 + sum_k a_k * phi_k(x,y))
    cf_a: Optional[torch.Tensor] = None  # [B, K] or [K] (broadcast)

    # softplus sharpness for positivity constraint
    softplus_beta: float = 1.0

    # numerical safety for divisions
    h_floor: float = 1e-4


# =============================================================================
# Grid construction
# =============================================================================

def make_grid(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    device: torch.device,
    dtype: torch.dtype,
    zb: Optional[torch.Tensor] = None,
    cf_basis: Optional[torch.Tensor] = None,
) -> SWEGrid:
    """
    Create x,y grids and meshgrids X,Y.
    Uses [0, L) periodic convention with Nx points.
    """
    if Nx < 4 or Ny < 4:
        raise ValueError("Nx and Ny should be >= 4 for stable central differences.")

    x = torch.linspace(0.0, float(Lx), steps=Nx+1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0.0, float(Ly), steps=Ny+1, device=device, dtype=dtype)[:-1]
    dx = float(Lx) / float(Nx)
    dy = float(Ly) / float(Ny)

    # meshgrid with ij indexing: X[i,j] corresponds to x[i], y[j]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    if zb is not None:
        zb = zb.to(device=device, dtype=dtype)
        if zb.shape != (Nx, Ny):
            raise ValueError(f"zb must have shape (Nx,Ny)=({Nx},{Ny}), got {tuple(zb.shape)}")

    if cf_basis is not None:
        cf_basis = cf_basis.to(device=device, dtype=dtype)
        if cf_basis.ndim != 3 or cf_basis.shape[1:] != (Nx, Ny):
            raise ValueError(f"cf_basis must have shape (K,Nx,Ny), got {tuple(cf_basis.shape)}")

    return SWEGrid(x=x, y=y, dx=dx, dy=dy, X=X, Y=Y, zb=zb, cf_basis=cf_basis)


# =============================================================================
# Periodic derivatives (for bathymetry slopes)
# =============================================================================

def ddx_periodic(f: torch.Tensor, dx: float) -> torch.Tensor:
    """Central difference ∂x with periodic wrap. f: [..., Nx, Ny]."""
    return (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (2.0 * dx)

def ddy_periodic(f: torch.Tensor, dy: float) -> torch.Tensor:
    """Central difference ∂y with periodic wrap. f: [..., Nx, Ny]."""
    return (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (2.0 * dy)


# =============================================================================
# Helpers: friction field construction
# =============================================================================

def _softplus(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.softplus(x, beta=beta)

def make_cf_field(params: SWEParams, grid: SWEGrid, batch_shape: Tuple[int, ...], device, dtype) -> Optional[torch.Tensor]:
    """
    Return c_f field of shape [B, Nx, Ny] (or [Nx,Ny] if B=()) depending on batch_shape.
    If cf_mode == "none", returns None.
    """
    if params.cf_mode == "none":
        return None

    beta = float(params.softplus_beta)

    if params.cf_mode == "scalar":
        if params.cf0 is None:
            raise ValueError("cf_mode='scalar' requires params.cf0.")
        cf0 = params.cf0
        if not torch.is_tensor(cf0):
            cf0 = torch.tensor(cf0, device=device, dtype=dtype)
        cf0 = cf0.to(device=device, dtype=dtype)

        # broadcast to batch
        # shape: batch_shape + (1,1)
        while cf0.ndim < len(batch_shape):
            cf0 = cf0.unsqueeze(0)
        cf0 = cf0.reshape(batch_shape + (1, 1))
        return _softplus(cf0, beta=beta)

    if params.cf_mode == "basis":
        if params.cf0 is None or params.cf_a is None:
            raise ValueError("cf_mode='basis' requires params.cf0 and params.cf_a.")
        if grid.cf_basis is None:
            raise ValueError("cf_mode='basis' requires grid.cf_basis (K,Nx,Ny).")

        cf0 = params.cf0
        a = params.cf_a
        if not torch.is_tensor(cf0):
            cf0 = torch.tensor(cf0, device=device, dtype=dtype)
        if not torch.is_tensor(a):
            a = torch.tensor(a, device=device, dtype=dtype)
        cf0 = cf0.to(device=device, dtype=dtype)
        a = a.to(device=device, dtype=dtype)

        K, Nx, Ny = grid.cf_basis.shape

        # a: [B,K] or [K] -> broadcast to batch
        if a.ndim == 1:
            if a.shape[0] != K:
                raise ValueError(f"cf_a has shape {tuple(a.shape)} but basis has K={K}.")
            a = a.unsqueeze(0)  # [1,K]
        if a.shape[-1] != K:
            raise ValueError(f"cf_a last dim must be K={K}, got {a.shape[-1]}")

        # infer batch size from batch_shape (e.g., (B,) or (B1,B2))
        B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1
        a_flat = a.reshape(-1, K)
        if a_flat.shape[0] == 1 and B > 1:
            a_flat = a_flat.expand(B, K)
        if a_flat.shape[0] != B:
            raise ValueError(f"cf_a batch mismatch: expected {B} rows from batch_shape={batch_shape}, got {a_flat.shape[0]}")

        # compute field: [B,Nx,Ny] = cf0 + sum_k a_k * phi_k
        phi = grid.cf_basis.reshape(K, Nx * Ny)  # [K, N]
        field = (a_flat @ phi).reshape(B, Nx, Ny)  # [B,Nx,Ny]

        cf0_flat = cf0
        # broadcast cf0 to [B,1,1]
        while cf0_flat.ndim < 1:
            cf0_flat = cf0_flat.unsqueeze(0)
        if cf0_flat.ndim == 0:
            cf0_flat = cf0_flat.reshape(1)
        if cf0_flat.numel() == 1 and B > 1:
            cf0_flat = cf0_flat.expand(B)
        if cf0_flat.numel() != B:
            # allow cf0 of shape batch_shape as well
            try:
                cf0_flat = cf0.reshape(-1)
            except Exception as e:
                raise ValueError("cf0 could not be broadcast to batch size.") from e
            if cf0_flat.numel() == 1 and B > 1:
                cf0_flat = cf0_flat.expand(B)
            if cf0_flat.numel() != B:
                raise ValueError(f"cf0 batch mismatch: expected {B}, got {cf0_flat.numel()}")

        field = field + cf0_flat.reshape(B, 1, 1)
        field = _softplus(field, beta=beta)

        # reshape back to batch_shape + (Nx,Ny)
        if len(batch_shape) > 0:
            return field.reshape(batch_shape + (Nx, Ny))
        return field.reshape(Nx, Ny)

    raise ValueError(f"Unknown cf_mode: {params.cf_mode}")


# =============================================================================
# CFL check (for explicit Rusanov update)
# =============================================================================

@torch.no_grad()
def estimate_max_wavespeed(q: torch.Tensor, g: torch.Tensor, h_floor: float) -> torch.Tensor:
    """
    Estimate max(|u| + c, |v| + c) where c = sqrt(g*h).
    q: [..., 3, Nx, Ny]
    """
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    hu = q[..., 1, :, :]
    hv = q[..., 2, :, :]
    u = hu / h
    v = hv / h
    c = torch.sqrt(torch.clamp(g, min=0.0) * h)
    sx = torch.abs(u) + c
    sy = torch.abs(v) + c
    return torch.maximum(sx.max(), sy.max())

@torch.no_grad()
def check_cfl_explicit(q: torch.Tensor, grid: SWEGrid, params: SWEParams, dt: float, cfl: float = 0.45) -> None:
    """
    Raise if dt violates a conservative CFL bound for explicit FV update.
    """
    g = params.g.to(device=q.device, dtype=q.dtype)
    amax = float(estimate_max_wavespeed(q, g, params.h_floor).item())
    if amax <= 0.0:
        return
    dt_max = cfl * min(grid.dx, grid.dy) / amax
    if dt > dt_max:
        raise ValueError(f"CFL violation: dt={dt:.3e} > dt_max={dt_max:.3e} (amax={amax:.3e}).")


# =============================================================================
# Rusanov flux + one step update
# =============================================================================

def _flux_x(q: torch.Tensor, g: torch.Tensor, h_floor: float) -> torch.Tensor:
    """Physical flux in x direction. q: [...,3,Nx,Ny] -> same shape."""
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    hu = q[..., 1, :, :]
    hv = q[..., 2, :, :]
    u = hu / h
    v = hv / h
    Fx0 = hu
    Fx1 = hu * u + 0.5 * g * h * h
    Fx2 = hu * v
    return torch.stack([Fx0, Fx1, Fx2], dim=-3)

def _flux_y(q: torch.Tensor, g: torch.Tensor, h_floor: float) -> torch.Tensor:
    """Physical flux in y direction. q: [...,3,Nx,Ny] -> same shape."""
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    hu = q[..., 1, :, :]
    hv = q[..., 2, :, :]
    u = hu / h
    v = hv / h
    Fy0 = hv
    Fy1 = hv * u
    Fy2 = hv * v + 0.5 * g * h * h
    return torch.stack([Fy0, Fy1, Fy2], dim=-3)

def _maxwavespeed_x(q: torch.Tensor, g: torch.Tensor, h_floor: float) -> torch.Tensor:
    """Local max wave speed in x: |u| + sqrt(g h)."""
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    u = q[..., 1, :, :] / h
    c = torch.sqrt(torch.clamp(g, min=0.0) * h)
    return torch.abs(u) + c

def _maxwavespeed_y(q: torch.Tensor, g: torch.Tensor, h_floor: float) -> torch.Tensor:
    """Local max wave speed in y: |v| + sqrt(g h)."""
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    v = q[..., 2, :, :] / h
    c = torch.sqrt(torch.clamp(g, min=0.0) * h)
    return torch.abs(v) + c

def step_swe_rusanov(
    q: torch.Tensor,          # [..., 3, Nx, Ny]
    grid: SWEGrid,
    params: SWEParams,
    dt: float,
    cf_field: Optional[torch.Tensor] = None,  # [..., Nx, Ny] or [Nx,Ny] broadcastable
) -> torch.Tensor:
    """
    One explicit FV step using Rusanov flux + source terms.
    """
    g = params.g.to(device=q.device, dtype=q.dtype)
    h_floor = float(params.h_floor)

    # Precompute physical fluxes at cell centers
    Fx = _flux_x(q, g, h_floor)
    Fy = _flux_y(q, g, h_floor)

    # Interface fluxes in x: i+1/2 between q(i) and q(i+1)
    qR_x = torch.roll(q, shifts=-1, dims=-2)          # i+1
    FxR = torch.roll(Fx, shifts=-1, dims=-2)
    aL = _maxwavespeed_x(q, g, h_floor)
    aR = _maxwavespeed_x(qR_x, g, h_floor)
    a_int = torch.maximum(aL, aR)                     # [...,Nx,Ny]
    # broadcast to vector components
    a_int_v = a_int.unsqueeze(-3)                     # [...,1,Nx,Ny]
    Fint_x = 0.5 * (Fx + FxR) - 0.5 * a_int_v * (qR_x - q)

    # Interface fluxes in y: j+1/2 between q(j) and q(j+1)
    qR_y = torch.roll(q, shifts=-1, dims=-1)
    FyR = torch.roll(Fy, shifts=-1, dims=-1)
    bL = _maxwavespeed_y(q, g, h_floor)
    bR = _maxwavespeed_y(qR_y, g, h_floor)
    b_int = torch.maximum(bL, bR)
    b_int_v = b_int.unsqueeze(-3)
    Fint_y = 0.5 * (Fy + FyR) - 0.5 * b_int_v * (qR_y - q)

    # Divergence of fluxes: (F_{i+1/2}-F_{i-1/2})/dx with periodic wrap
    Fint_x_L = torch.roll(Fint_x, shifts=1, dims=-2)  # i-1/2
    Fint_y_L = torch.roll(Fint_y, shifts=1, dims=-1)

    divF = (Fint_x - Fint_x_L) / grid.dx + (Fint_y - Fint_y_L) / grid.dy
    q_next = q - dt * divF

    # Source terms
    h = torch.clamp(q[..., 0, :, :], min=h_floor)
    hu = q[..., 1, :, :]
    hv = q[..., 2, :, :]
    u = hu / h
    v = hv / h

    # Bathymetry slope source (if provided)
    if grid.zb is not None:
        zb = grid.zb
        # broadcast zb if batch exists
        while zb.ndim < h.ndim:
            zb = zb.unsqueeze(0)
        dzdx = ddx_periodic(zb, grid.dx)
        dzdy = ddy_periodic(zb, grid.dy)
        q_next[..., 1, :, :] = q_next[..., 1, :, :] - dt * (g * h * dzdx)
        q_next[..., 2, :, :] = q_next[..., 2, :, :] - dt * (g * h * dzdy)

    # Friction source (if provided)
    if cf_field is not None:
        cf = cf_field
        # make broadcastable to h
        while cf.ndim < h.ndim:
            cf = cf.unsqueeze(0)
        speed = torch.sqrt(u * u + v * v + 1e-12)
        q_next[..., 1, :, :] = q_next[..., 1, :, :] - dt * (cf * h * u * speed)
        q_next[..., 2, :, :] = q_next[..., 2, :, :] - dt * (cf * h * v * speed)

    return q_next


# =============================================================================
# Convenience: build conservative state from eta,u,v and mean depth H
# =============================================================================

def build_state_from_eta_uv(
    eta: torch.Tensor,   # [..., Nx, Ny]
    u: torch.Tensor,     # [..., Nx, Ny]
    v: torch.Tensor,     # [..., Nx, Ny]
    params: SWEParams,
) -> torch.Tensor:
    """
    Construct q=[h,hu,hv] from eta and velocities using h = H + eta.
    """
    H = params.H.to(device=eta.device, dtype=eta.dtype)
    while H.ndim < eta.ndim - 2:
        H = H.unsqueeze(0)
    h = torch.clamp(H + eta, min=params.h_floor)
    hu = h * u
    hv = h * v
    return torch.stack([h, hu, hv], dim=-3)


# =============================================================================
# Rollout
# =============================================================================

def rollout_swe(
    q0: torch.Tensor,                 # [..., 3, Nx, Ny]
    grid: SWEGrid,
    params: SWEParams,
    dt: float,
    steps: int,
    save_every: int = 1,
    enforce_cfl: bool = False,
    cfl: float = 0.45,
    return_primitive: bool = True,
    no_grad: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Roll out SWE for 'steps' timesteps of size dt.

    Returns a dict with:
      - 'q': [T, ..., 3, Nx, Ny]  (always)
      - if return_primitive:
          'h','u','v': [T, ..., Nx, Ny]
      - 't': [T]
      - 'cf': friction field used (if any): [..., Nx, Ny] (constant over time)
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    device = q0.device
    dtype = q0.dtype

    # Determine batch_shape = q0.shape[:-3]
    batch_shape = q0.shape[:-3]

    # Construct friction field once (time-invariant) if requested and not passed externally.
    cf_field = None
    if params.cf_mode != "none":
        cf_field = make_cf_field(params, grid, batch_shape=batch_shape, device=device, dtype=dtype)

    # Optionally check CFL at start (conservative, but cheap)
    if enforce_cfl:
        check_cfl_explicit(q0, grid, params, dt=dt, cfl=cfl)

    # Saving
    T = (steps // save_every) + 1
    q_traj = []
    t_traj = []

    q = q0
    q_traj.append(q)
    t_traj.append(torch.zeros((), device=device, dtype=dtype))

    with (torch.no_grad() if no_grad else nullcontext()):
        for n in range(1, steps + 1):
            q = step_swe_rusanov(q, grid, params, dt=dt, cf_field=cf_field)
            if (n % save_every) == 0:
                q_traj.append(q)
                t_traj.append(torch.tensor(n * dt, device=device, dtype=dtype))

    q_out = torch.stack(q_traj, dim=0)   # [T, ..., 3, Nx, Ny]
    t_out = torch.stack(t_traj, dim=0)   # [T]

    out: Dict[str, torch.Tensor] = {"q": q_out, "t": t_out}
    if cf_field is not None:
        out["cf"] = cf_field

    if return_primitive:
        h = torch.clamp(q_out[..., 0, :, :], min=params.h_floor)
        hu = q_out[..., 1, :, :]
        hv = q_out[..., 2, :, :]
        u = hu / h
        v = hv / h
        out["h"] = h
        out["u"] = u
        out["v"] = v

    return out


# =============================================================================
# Self-test / example
# =============================================================================

def _make_fourier_basis(grid: SWEGrid, Kx: int, Ky: int) -> torch.Tensor:
    """
    Simple deterministic Fourier cosine basis (excluding constant) for cf field.
    Returns [K, Nx, Ny] with K = Kx*Ky.
    """
    Nx = grid.x.numel()
    Ny = grid.y.numel()
    X = grid.X / (grid.x[-1] + grid.dx) * 2.0 * torch.pi  # scale to [0,2pi)
    Y = grid.Y / (grid.y[-1] + grid.dy) * 2.0 * torch.pi

    phis = []
    for kx in range(1, Kx + 1):
        for ky in range(1, Ky + 1):
            phis.append(torch.cos(kx * X) * torch.cos(ky * Y))
    phi = torch.stack(phis, dim=0)  # [K,Nx,Ny]
    return phi

def _self_test() -> None:
    """
    Quick smoke test:
      - build a small periodic wave packet in eta
      - zero initial velocities
      - run a short rollout
    """
    device = torch.device("cpu")
    dtype = torch.float32

    Nx, Ny = 64, 64
    Lx, Ly = 1.0, 1.0
    grid = make_grid(Nx, Ny, Lx, Ly, device=device, dtype=dtype)

    # Basis for friction (optional)
    phi = _make_fourier_basis(grid, Kx=4, Ky=4)  # K=16
    grid.cf_basis = phi

    params = SWEParams(
        g=torch.tensor(9.81, device=device, dtype=dtype),
        H=torch.tensor(1.0, device=device, dtype=dtype),
        cf_mode="basis",
        cf0=torch.tensor(-4.0, device=device, dtype=dtype),  # softplus(-4) ~ 0.018
        cf_a=torch.zeros((1, phi.shape[0]), device=device, dtype=dtype),
        softplus_beta=1.0,
        h_floor=1e-4,
    )

    # initial condition: small bump in eta
    r2 = (grid.X - 0.5) ** 2 + (grid.Y - 0.5) ** 2
    eta0 = 0.02 * torch.exp(-r2 / (2.0 * 0.05 ** 2))
    u0 = torch.zeros_like(eta0)
    v0 = torch.zeros_like(eta0)
    q0 = build_state_from_eta_uv(eta0, u0, v0, params).unsqueeze(0)  # [B,3,Nx,Ny]

    dt = 1e-3
    out = rollout_swe(q0, grid, params, dt=dt, steps=200, save_every=20, enforce_cfl=True, return_primitive=True)
    h = out["h"]  # [T,B,Nx,Ny]
    print("OK:", h.shape, "h range", float(h.min()), float(h.max()), "cf mean", float(out["cf"].mean()))

if __name__ == "__main__":
    _self_test()
