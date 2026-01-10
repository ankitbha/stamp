#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
sim/heat.py

Differentiable (PyTorch) simulator for the 2D anisotropic heat equation with periodic BC:
  u_t = alpha_x * u_xx + alpha_y * u_yy + f(x,y,t)

Design goals:
- Single canonical discretization used by data generation and later calibration.
- Periodic boundaries via torch.roll.
- Supports batched simulation and optional trajectory saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from contextlib import nullcontext
import torch


# In[ ]:


# =============================================================================
# Grid / params containers
# =============================================================================

@dataclass
class HeatGrid:
    # spatial grids (1D) and spacings
    x: torch.Tensor          # [Nx]
    y: torch.Tensor          # [Ny]
    dx: float
    dy: float

    # meshgrids for forcing / IC convenience (optional)
    X: torch.Tensor          # [Nx, Ny]
    Y: torch.Tensor          # [Nx, Ny]


@dataclass
class HeatParams:
    alpha_x: torch.Tensor    # scalar tensor (possibly learnable)
    alpha_y: torch.Tensor    # scalar tensor (possibly learnable)
    A: torch.Tensor          # forcing amplitude (scalar tensor, optional)


# In[ ]:


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
) -> HeatGrid:
    """
    Create x,y grids and meshgrids X,Y. Uses [0, L] inclusive linspace like numpy version.
    """
    if Nx < 2 or Ny < 2:
        raise ValueError(f"Nx and Ny must be >= 2, got Nx={Nx}, Ny={Ny}.")
    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError(f"Lx and Ly must be > 0, got Lx={Lx}, Ly={Ly}.")

    x = torch.linspace(0.0, float(Lx), int(Nx), device=device, dtype=dtype)
    y = torch.linspace(0.0, float(Ly), int(Ny), device=device, dtype=dtype)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    # Match numpy: X, Y = np.meshgrid(x, y, indexing='ij')
    X = x[:, None].expand(Nx, Ny)
    Y = y[None, :].expand(Nx, Ny)

    return HeatGrid(x=x, y=y, dx=dx, dy=dy, X=X, Y=Y)


# In[ ]:


# =============================================================================
# Forcing
# =============================================================================

def forcing_default(
    X: torch.Tensor,   # [Nx, Ny]
    Y: torch.Tensor,   # [Nx, Ny]
    t: torch.Tensor,   # scalar tensor
    A: torch.Tensor,   # scalar tensor
    T_final: float,
) -> torch.Tensor:
    """
    Matches FieldFormer generator:
      A * cos(pi x) * cos(pi y) * sin(4 pi t / T)
    """
    if T_final <= 0.0:
        raise ValueError(f"T_final must be > 0, got {T_final}.")

    # Ensure t and A are tensors on the right device/dtype; keep gradients if present.
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=X.device, dtype=X.dtype)
    else:
        t = t.to(device=X.device, dtype=X.dtype)

    if not torch.is_tensor(A):
        A = torch.tensor(A, device=X.device, dtype=X.dtype)
    else:
        A = A.to(device=X.device, dtype=X.dtype)

    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, device=X.device, dtype=X.dtype)

    return A * torch.cos(pi * X) * torch.cos(pi * Y) * torch.sin((4.0 * pi * t) / float(T_final))


# In[ ]:


# =============================================================================
# Discrete operators (periodic BC)
# =============================================================================

def laplacian_periodic_anisotropic(
    u: torch.Tensor,   # [B, Nx, Ny] or [Nx, Ny]
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (u_xx, u_yy) using second-order central differences with periodic wrap.
    Uses torch.roll.

    Supports u shaped [Nx, Ny] or [B, Nx, Ny].
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must be > 0, got dx={dx}, dy={dy}.")
    if u.ndim not in (2, 3):
        raise ValueError(f"u must have shape [Nx,Ny] or [B,Nx,Ny], got {tuple(u.shape)}.")

    # Identify spatial dims
    if u.ndim == 2:
        dim_x, dim_y = 0, 1
    else:
        dim_x, dim_y = 1, 2

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    u_xx = (torch.roll(u, shifts=-1, dims=dim_x) - 2.0 * u + torch.roll(u, shifts=1, dims=dim_x)) * inv_dx2
    u_yy = (torch.roll(u, shifts=-1, dims=dim_y) - 2.0 * u + torch.roll(u, shifts=1, dims=dim_y)) * inv_dy2

    return u_xx, u_yy


# In[ ]:


# =============================================================================
# Single step
# =============================================================================

def step_heat_euler(
    u: torch.Tensor,                 # [B,Nx,Ny] or [Nx,Ny]
    params: HeatParams,
    grid: HeatGrid,
    t: torch.Tensor,                 # scalar
    dt: float,
    forcing_fn: Callable[..., torch.Tensor],
    forcing_kwargs: Dict,
) -> torch.Tensor:
    """
    One explicit Euler step:
      u_{n+1} = u_n + dt * (alpha_x u_xx + alpha_y u_yy + f)
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got dt={dt}.")
    if u.ndim not in (2, 3):
        raise ValueError(f"u must have shape [Nx,Ny] or [B,Nx,Ny], got {tuple(u.shape)}.")
    if forcing_kwargs is None:
        forcing_kwargs = {}

    # Laplacian terms (periodic)
    u_xx, u_yy = laplacian_periodic_anisotropic(u, grid.dx, grid.dy)

    # Forcing: grid.X/Y are [Nx,Ny]; broadcast to batch if needed
    f = forcing_fn(X=grid.X, Y=grid.Y, t=t, A=params.A, **forcing_kwargs)  # [Nx,Ny]
    if u.ndim == 3:
        f = f.unsqueeze(0)  # [1,Nx,Ny] -> broadcasts over B

    rhs = params.alpha_x * u_xx + params.alpha_y * u_yy + f
    return u + float(dt) * rhs


# In[ ]:


# =============================================================================
# Rollout / simulation
# =============================================================================

def rollout_heat(
    u0: torch.Tensor,                # [Nx,Ny] or [B,Nx,Ny]
    params: HeatParams,
    grid: HeatGrid,
    t_vec: torch.Tensor,             # [Nt]
    forcing_fn: Callable[..., torch.Tensor] = forcing_default,
    forcing_kwargs: Optional[Dict] = None,
    save_every: int = 1,
    no_grad = False,
) -> Dict[str, torch.Tensor]:
    """
    Rollout heat dynamics over time vector t_vec.

    Returns a dict with:
      - "u": saved trajectory [Nsaved, B, Nx, Ny] (or [Nsaved, Nx, Ny] if unbatched)
      - "t": saved times [Nsaved]
    """
    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        if u0.ndim not in (2, 3):
            raise ValueError(f"u0 must have shape [Nx,Ny] or [B,Nx,Ny], got {tuple(u0.shape)}.")
        if t_vec.ndim != 1:
            raise ValueError(f"t_vec must be 1D [Nt], got {tuple(t_vec.shape)}.")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}.")
        if t_vec.numel() < 2:
            raise ValueError("t_vec must have at least 2 entries to define dt.")

        forcing_kwargs = {} if forcing_kwargs is None else forcing_kwargs

        # Ensure u is on the same device/dtype as the grid
        u = u0.to(device=grid.X.device, dtype=grid.X.dtype)

        dt = float(t_vec[1] - t_vec[0])
        if dt <= 0.0:
            raise ValueError(f"t_vec must be increasing; got dt={dt} from t_vec[1]-t_vec[0].")

        saved_u = []
        saved_t = []

        Nt = int(t_vec.numel())
        for n in range(Nt):
            if (n % save_every) == 0:
                saved_u.append(u.clone())
                saved_t.append(t_vec[n].clone())

            if n == Nt - 1:
                break  # no step past the final time

            u = step_heat_euler(
                u=u,
                params=params,
                grid=grid,
                t=t_vec[n],
                dt=dt,
                forcing_fn=forcing_fn,
                forcing_kwargs=forcing_kwargs,
            )

        u_out = torch.stack(saved_u, dim=0)  # [Nsaved, ...]
        t_out = torch.stack(saved_t, dim=0)  # [Nsaved]

    return {"u": u_out, "t": t_out}


# In[ ]:


# =============================================================================
# Utilities: CFL check (optional but nice)
# =============================================================================

def check_cfl_explicit(
    alpha_x: float,
    alpha_y: float,
    dx: float,
    dy: float,
    dt: float,
) -> None:
    """
    Sufficient stability condition for explicit anisotropic diffusion:
      r_x + r_y <= 1/2 where r_x=alpha_x*dt/dx^2, r_y=alpha_y*dt/dy^2
    """
    if dx <= 0.0 or dy <= 0.0 or dt <= 0.0:
        raise ValueError(f"dx, dy, dt must be > 0; got dx={dx}, dy={dy}, dt={dt}.")
    if alpha_x < 0.0 or alpha_y < 0.0:
        raise ValueError(f"alpha_x and alpha_y must be >= 0; got alpha_x={alpha_x}, alpha_y={alpha_y}.")

    r_x = alpha_x * dt / (dx * dx)
    r_y = alpha_y * dt / (dy * dy)
    cfl_sum = r_x + r_y

    if cfl_sum > 0.5 + 1e-12:
        raise RuntimeError(
            f"Unstable explicit step: r_x + r_y = {cfl_sum:.6f} (> 0.5). "
            f"(r_x={r_x:.6f}, r_y={r_y:.6f})  Try smaller dt or larger grid spacing."
        )


# In[ ]:


# =============================================================================
# Optional: small self-test
# =============================================================================

def _self_test() -> None:
    """
    Quick smoke test:
    - make grid
    - create sinusoidal IC
    - do a short rollout (no_grad)
    - do a short rollout (grad)
    - check grads for alpha_x/alpha_y/A (and optionally u0)
    - print min/max, ensure no NaNs/Infs
    """
    device = torch.device("cpu")
    dtype = torch.float32

    # Grid / time
    Nx, Ny = 32, 32
    Lx, Ly = 1.0, 1.0
    T_final = 0.2
    Nt = 50
    t_vec = torch.linspace(0.0, T_final, Nt, device=device, dtype=dtype)

    # Make grid
    grid = make_grid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, dtype=dtype)

    # Parameters
    alpha_x = torch.tensor(0.01, device=device, dtype=dtype, requires_grad=True)
    alpha_y = torch.tensor(0.001, device=device, dtype=dtype, requires_grad=True)
    A = torch.tensor(5.0, device=device, dtype=dtype, requires_grad=True)
    params = HeatParams(alpha_x=alpha_x, alpha_y=alpha_y, A=A)

    # CFL check (use .item() since these are tensors)
    dt = float(t_vec[1] - t_vec[0])
    check_cfl_explicit(
        alpha_x=float(alpha_x.detach().item()),
        alpha_y=float(alpha_y.detach().item()),
        dx=grid.dx,
        dy=grid.dy,
        dt=dt,
    )

    # Initial condition
    u0 = (torch.sin(2.0 * torch.pi * grid.X) * torch.sin(2.0 * torch.pi * grid.Y)).to(dtype)
    # Optionally test gradients through u0 as well
    u0 = u0.clone().detach().requires_grad_(True)

    # --- No-grad rollout ---
    out_ng = rollout_heat(
        u0=u0.detach(),   # detach for no-grad rollout
        params=HeatParams(alpha_x=alpha_x.detach(), alpha_y=alpha_y.detach(), A=A.detach()),
        grid=grid,
        t_vec=t_vec,
        forcing_fn=forcing_default,
        forcing_kwargs={"T_final": T_final},
        save_every=10,
        no_grad=True
    )
    u_ng = out_ng["u"]
    if torch.isnan(u_ng).any() or torch.isinf(u_ng).any():
        raise RuntimeError("NaNs/Infs detected in no-grad rollout.")

    # --- Grad rollout ---
    out_g = rollout_heat(
        u0=u0,
        params=params,
        grid=grid,
        t_vec=t_vec,
        forcing_fn=forcing_default,
        forcing_kwargs={"T_final": T_final},
        save_every=10,
    )
    u_g = out_g["u"]
    if torch.isnan(u_g).any() or torch.isinf(u_g).any():
        raise RuntimeError("NaNs/Infs detected in grad rollout.")

    # Define a simple scalar objective from the trajectory to backprop through.
    # Use final saved frame mean-squared value (nontrivial, smooth).
    loss = (u_g[-1] ** 2).mean()
    loss.backward()

    # Gradient checks
    def _check_grad(name: str, x: torch.Tensor) -> None:
        if x.grad is None:
            raise RuntimeError(f"Gradient for {name} is None.")
        if torch.isnan(x.grad).any() or torch.isinf(x.grad).any():
            raise RuntimeError(f"Gradient for {name} has NaNs/Infs.")
        # It is possible (though unlikely) that a gradient is exactly zero; warn not fail.
        gnorm = float(x.grad.abs().mean().item())
        if gnorm == 0.0:
            print(f"[self_test][warn] mean(|grad|) for {name} is 0.0 (unexpected but possible).")
        else:
            print(f"[self_test] mean(|grad|) {name}: {gnorm:.3e}")

    _check_grad("alpha_x", alpha_x)
    _check_grad("alpha_y", alpha_y)
    _check_grad("A", A)
    _check_grad("u0", u0)

    # Summary stats
    print("[self_test] Heat simulator OK (no-grad + grad)")
    print(f"  no-grad: saved frames={u_ng.shape[0]}, u min/max=({u_ng.min().item():.4f}, {u_ng.max().item():.4f})")
    print(f"  grad:    saved frames={u_g.shape[0]}, u min/max=({u_g.min().item():.4f}, {u_g.max().item():.4f})")
    print(f"  loss: {float(loss.item()):.6f}")


# In[ ]:


if __name__ == "__main__":
    _self_test()


# In[ ]:




