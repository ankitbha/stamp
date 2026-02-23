# model/calibrator/calibrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# =============================================================================
# Learnable parameter containers
# =============================================================================

class LearnableScalar(nn.Module):
    """
    A learnable scalar with optional bounds via sigmoid transform.

    If bounds=(lo,hi): value = lo + (hi-lo)*sigmoid(raw)
    Else: value = raw
    """
    def __init__(
        self,
        name: str,
        init: float,
        bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        self.bounds = bounds

        if bounds is None:
            self.raw = nn.Parameter(torch.tensor(float(init), device=device, dtype=dtype))
        else:
            lo, hi = bounds
            init01 = (float(init) - lo) / max(hi - lo, 1e-12)
            init01 = min(max(init01, 1e-6), 1.0 - 1e-6)
            raw0 = torch.log(torch.tensor(init01 / (1.0 - init01), device=device, dtype=dtype))
            self.raw = nn.Parameter(raw0)

    def value(self) -> Tensor:
        if self.bounds is None:
            return self.raw
        lo, hi = self.bounds
        return lo + (hi - lo) * torch.sigmoid(self.raw)

    def extra_repr(self) -> str:
        return f"name={self.name}, bounds={self.bounds}"


class LearnableField2D(nn.Module):
    """
    A learnable 2D field (H,W) with optional nonnegativity, output clamp, and output scaling.

    - raw is always an nn.Parameter
    - If nonneg=True: value = softplus(raw, beta)
    - value is then multiplied by `scale`
    - If clamp=(lo,hi): value = clamp(value, lo, hi)

    Note: scaling is applied BEFORE clamp so clamp bounds are in the physical units.
    """
    def __init__(
        self,
        name: str,
        shape_hw: Tuple[int, int],
        init: Union[float, Tensor] = 0.0,
        nonneg: bool = False,
        clamp: Optional[Tuple[float, float]] = None,
        softplus_beta: float = 1.0,
        scale: float = 1.0,                      # <-- NEW
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        self.shape_hw = (int(shape_hw[0]), int(shape_hw[1]))
        self.nonneg = bool(nonneg)
        self.clamp = clamp
        self.softplus_beta = float(softplus_beta)

        # Store scale as a buffer so it moves with .to(device) and matches dtype
        self.register_buffer("scale", torch.tensor(float(scale), device=device, dtype=dtype))

        if isinstance(init, torch.Tensor):
            init_t = init.to(device=device, dtype=dtype)
            if tuple(init_t.shape) != self.shape_hw:
                raise ValueError(f"{name}: init shape {tuple(init_t.shape)} != {self.shape_hw}")
        else:
            init_t = torch.full(self.shape_hw, float(init), device=device, dtype=dtype)

        self.raw = nn.Parameter(init_t)

    def value(self) -> Tensor:
        x = self.raw
        if self.nonneg:
            x = F.softplus(x, beta=self.softplus_beta)

        # Apply physical scale (main stabilization knob)
        x = x * self.scale

        if self.clamp is not None:
            lo, hi = self.clamp
            x = torch.clamp(x, lo, hi)
        return x

    def forward(self) -> Tensor:
        return self.value()

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, shape={self.shape_hw}, nonneg={self.nonneg}, "
            f"scale={float(self.scale.item()):g}, clamp={self.clamp}"
        )

class BatchedLearnableField2D(torch.nn.Module):
    """
    Holds B independent LearnableField2D modules and stacks them into [B,H,W].
    """
    def __init__(self, B: int, **lf_kwargs):
        super().__init__()
        self.B = B
        self.fields = torch.nn.ModuleList()
        for b in range(B):
            self.fields.append(LearnableField2D(**lf_kwargs))

    def forward(self):
        xs = [f() for f in self.fields]  # each [H,W]
        return torch.stack(xs, dim=0)    # [B,H,W

    def value(self):
        return self.forward()

    @property
    def raws(self):
        # list of raw parameters (for clamping etc.)
        return [f.raw for f in self.fields]


# =============================================================================
# Observation operators: field[...,H,W,T] -> sensor[...,S,T]
# =============================================================================

class IndexObserver(nn.Module):
    def __init__(self, sensors_idx: Tensor, channel_index: int = 0):
        super().__init__()
        if sensors_idx.ndim != 2 or sensors_idx.shape[1] != 2:
            raise ValueError(f"sensors_idx must be [S,2], got {tuple(sensors_idx.shape)}")
        self.register_buffer("sensors_idx", sensors_idx.long())
        self.channel_index = int(channel_index)

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim >= 5 and field.shape[-1] > 1:
            field = field[..., self.channel_index]  # [...,H,W,T]
        if field.ndim < 3:
            raise ValueError(f"Expected field [...,H,W,T], got {tuple(field.shape)}")

        H, W, T = field.shape[-3], field.shape[-2], field.shape[-1]
        idx = self.sensors_idx
        iy = idx[:, 0].clamp(0, W - 1)  # careful: iy indexes SECOND spatial dim
        ix = idx[:, 1].clamp(0, H - 1)  # ix indexes FIRST spatial dim

        # Match dataset: U[ix, iy, :]
        pred = field[..., ix, iy, :]  # [...,S,T]
        return pred.contiguous()

class BilinearObserver(nn.Module):
    """
    Bilinear sampling at continuous coordinates sensors_xy in [0,1]x[0,1],
    consistent with dataset convention where U is indexed as U[ix, iy, t]
    for U shaped [Nx, Ny, T].

    Inputs:
      field: [..., Nx, Ny, T]  (time-last)
        Optionally: [..., Nx, Ny, T, C] with channel_index selecting last dim.
    Returns:
      pred_sensor: [..., S, T]
    """
    def __init__(self, sensors_xy: Tensor, channel_index: int = 0, align_corners: bool = True):
        super().__init__()
        if sensors_xy.ndim != 2 or sensors_xy.shape[1] != 2:
            raise ValueError(f"sensors_xy must be [S,2], got {tuple(sensors_xy.shape)}")
        self.register_buffer("sensors_xy", sensors_xy.float())
        self.channel_index = int(channel_index)
        self.align_corners = bool(align_corners)

    def forward(self, field: Tensor) -> Tensor:
        # Optional channel-last: [...,Nx,Ny,T,C] -> [...,Nx,Ny,T]
        if field.ndim >= 5 and field.shape[-1] > 1:
            field = field[..., self.channel_index]  # [...,Nx,Ny,T]

        if field.ndim < 3:
            raise ValueError(f"Expected field [...,Nx,Ny,T], got {tuple(field.shape)}")

        *batch, Nx, Ny, T = field.shape
        device = field.device
        dtype = field.dtype

        # grid_sample expects input [N, C, H, W] where H=height (y axis), W=width (x axis).
        # Our field is [Nx, Ny, T] with first axis = x, second axis = y.
        # So we must transpose spatial dims to make H=Ny, W=Nx.
        #
        # For each time t:
        #   slice = field[..., :, :, t] -> [...,Nx,Ny]
        #   transpose -> [...,Ny,Nx] to be (H,W)
        #
        # Then sample at (x,y) using normalized grid in (W,H) order.

        # Flatten batch -> N
        N = 1
        for d in batch:
            N *= int(d)

        # Reshape to [N, Nx, Ny, T]
        f = field.reshape(N, Nx, Ny, T)

        # Build normalized sampling grid for grid_sample:
        # grid_sample uses grid[...,0]=x in [-1,1] over width (W=Nx),
        # and grid[...,1]=y in [-1,1] over height (H=Ny).
        xyc = self.sensors_xy.clamp(0.0, 1.0)  # [S,2]
        gx = 2.0 * xyc[:, 0] - 1.0  # x -> width axis (Nx)
        gy = 2.0 * xyc[:, 1] - 1.0  # y -> height axis (Ny)

        # Want output shape [N, S, 1, 2] so grid_sample returns [N,1,S,1]
        grid = torch.stack([gx, gy], dim=-1).to(device=device, dtype=dtype)  # [S,2]
        grid = grid[None, :, None, :]                                       # [1,S,1,2]
        grid = grid.expand(N, -1, -1, -1).contiguous()                      # [N,S,1,2]

        preds = []
        for t in range(T):
            # slice: [N, Nx, Ny] -> transpose to [N, Ny, Nx]
            ft = f[..., t].transpose(1, 2).contiguous()  # [N, Ny, Nx]
            ft = ft[:, None, :, :]                       # [N,1,Ny,Nx]
            out = F.grid_sample(
                ft, grid, mode="bilinear", align_corners=self.align_corners
            )                                            # [N,1,S,1]
            preds.append(out[:, 0, :, 0])                # [N,S]

        pred = torch.stack(preds, dim=-1)                # [N,S,T]
        pred = pred.reshape(*batch, pred.shape[-2], pred.shape[-1])  # [...,S,T]
        return pred.contiguous()

# =============================================================================
# Simulator adapter + Calibrator core
# =============================================================================

@dataclass
class CalibratorOutputs:
    pred_sensor: Tensor                  # [S,T]
    sim_out: Dict[str, Any]              # raw simulator output dict
    field: Optional[Tensor] = None       # extracted field series (if requested)
    theta: Optional[Dict[str, Tensor]] = None


class SimulatorAdapter:
    """
    Wrap a simulator rollout function that returns a Dict[str,Tensor],
    and provide a standard way to extract a field series for observation.

    Example keys by your sims:
      heat: out["u"] -> [T,Nx,Ny]
      swe : out["h"] or out["q"] or out["eta"] (depending what you choose to observe)
      pol : out["U"] -> [T,Nx,Ny]  (snapshots)
    """
    def __init__(
        self,
        rollout_fn: Callable[..., Dict[str, Any]],
        field_key: str,
        field_subkey: Optional[Union[int, str]] = None,
    ):
        """
        field_subkey:
          - if field is a tensor with channels, you may pass int channel index (handled by observer too)
          - if simulator stores nested dict (unlikely here), you could pass a str (kept for flexibility)
        """
        self.rollout_fn = rollout_fn
        self.field_key = field_key
        self.field_subkey = field_subkey

    def __call__(self, **kwargs) -> Dict[str, Any]:
        return self.rollout_fn(**kwargs)

    def extract_field(self, sim_out: Dict[str, Any]) -> Tensor:
        if self.field_key not in sim_out:
            raise KeyError(f"Simulator output missing key '{self.field_key}'. Keys: {list(sim_out.keys())}")
        field = sim_out[self.field_key]
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"sim_out['{self.field_key}'] is not a Tensor (got {type(field)})")

        # Optional subkey behavior (kept minimal; most cases won't use this)
        if isinstance(self.field_subkey, int) and field.ndim == 4:
            # If field is [T,C,H,W], keep all channels; observer will pick channel_index.
            return field
        return field


class Calibrator(nn.Module):
    """
    Generic calibrator: owns learnable params; runs sim; observes sensors.

    The tuner_{heat/swe/pol}.py will:
      - construct SimulatorAdapter(rollout_fn=..., field_key=...)
      - construct observer (IndexObserver or BilinearObserver)
      - add learnable params into `params` ModuleDict
      - call forward with fixed sim arguments (grid, params structs, dt, steps, etc.)

    This module does NOT:
      - split train/val
      - run optimization loop
      - early stop
    """
    def __init__(
        self,
        sim: SimulatorAdapter,
        observer: nn.Module,
        params: Optional[nn.ModuleDict] = None,
    ):
        super().__init__()
        self.sim = sim
        self.observer = observer
        self.params = params if params is not None else nn.ModuleDict()

    def theta(self) -> Dict[str, Tensor]:
        """
        Get constrained learnable parameter values.
        """
        th: Dict[str, Tensor] = {}
        for k, mod in self.params.items():
            if hasattr(mod, "value") and callable(getattr(mod, "value")):
                th[k] = mod.value()
            else:
                raise TypeError(f"Param '{k}' does not implement .value()")
        return th

    def forward(
        self,
        return_field: bool = False,
        **fixed_sim_kwargs: Any,
    ) -> CalibratorOutputs:
        """
        fixed_sim_kwargs are non-learned simulator inputs supplied by tuner:
          heat: u0, params(HeatParams), grid(HeatGrid), t_vec, forcing_kwargs, save_every...
          swe : q0, grid(SWEGrid), params(SWEParams), dt, steps, ...
          pol : grid(PolGrid), params(PolParams), dt, steps, U0, save_every, ...

        Learnable parameters (theta) are injected alongside these kwargs.
        """
        th = self.theta()
        sim_out = self.sim(**th, **fixed_sim_kwargs)
        field = self.sim.extract_field(sim_out)
        pred_sensor = self.observer(field)

        return CalibratorOutputs(
            pred_sensor=pred_sensor,
            sim_out=sim_out,
            field=field if return_field else None,
            theta=th,
        )
