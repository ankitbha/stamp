# model/calibrator/tuner_pol.py
"""
SimGrad tuner for the pollution simulator (polsim).

- Optimizes unknown sources (S_unknown) by minimizing sensor data loss through the differentiable simulator.
- Keeps a flag + placeholders for STAMP (dynamics-consistency loss via a frozen prior), to be filled later.

Expected NPZ keys (robustly handled):
  - sensor_noisy or sensor_clean: sensor time series
      shapes supported:
        [S, T] or [T, S] or [N, S, T] or [N, T, S]
  - sensors_idx: [S,2] integer indices into grid (preferred)
  - sensors_xy:  [S,2] float coords in [0,1]^2 (fallback)
  - x, y (optional): grid coordinates (used to infer H,W)
  - dt (optional), steps (optional), save_every (optional): rollout control
  - params / param_names (optional): pollution params (k, etc.) if you saved them
  - rng_seed (optional)

This file is intentionally self-contained and slightly defensive, since dataset formats can differ.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import sys
sys.path.append(os.path.abspath("../../"))

# Your project utilities
from model.utils.io import load_npz
from model.utils.logging import setup_logger

# Calibrator core + objectives
from model.calibrator.calibrator import (
    Calibrator,
    SimulatorAdapter,
    LearnableField2D,
    IndexObserver,
    BilinearObserver,
)
from model.calibrator.objectives import ObjectiveConfig, compute_total_objective, compute_total_objective_batched, softmin_aggregate

# Simulator
import sim.polsim as polsim


# =============================================================================
# Strategy modules (optional)
# =============================================================================

class ShiftedSoftplusField2D(torch.nn.Module):
    """
    Nonnegative field with shifted softplus so raw=0 maps to 0 (not softplus(0)).
    Keeps gradients smooth while avoiding the 0.693 offset.
    """
    def __init__(
        self,
        name: str,
        shape_hw: Tuple[int, int],
        init: float,
        softplus_beta: float,
        scale: float,
        clamp: Optional[Tuple[float, float]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        self.shape_hw = tuple(shape_hw)
        self.softplus_beta = float(softplus_beta)
        self.scale = float(scale)
        self.clamp = clamp
        self.raw = torch.nn.Parameter(torch.full(self.shape_hw, float(init), device=device, dtype=dtype))

    def value(self) -> torch.Tensor:
        x = torch.nn.functional.softplus(self.raw, beta=self.softplus_beta)
        x = x - torch.nn.functional.softplus(torch.zeros_like(self.raw), beta=self.softplus_beta)
        x = torch.clamp(x, min=0.0)
        x = x * self.scale
        if self.clamp is not None:
            lo, hi = self.clamp
            x = torch.clamp(x, lo, hi)
        return x


class FactorizedUnknownField2D(torch.nn.Module):
    """
    S_unknown = amp * base, with both nonnegative.
    Stage A: optimize amp only (freeze base)
    Stage B: unfreeze base to learn spatial detail
    """
    def __init__(
        self,
        name: str,
        shape_hw: Tuple[int, int],
        amp_init: float,
        base_init: float,
        amp_softplus_beta: float,
        base_softplus_beta: float,
        amp_scale: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_shifted_softplus_base: bool = False,
    ):
        super().__init__()
        self.name = name
        self.shape_hw = tuple(shape_hw)
        self.amp = torch.nn.Parameter(torch.full((), float(amp_init), device=device, dtype=dtype))
        self.base_raw = torch.nn.Parameter(torch.full(self.shape_hw, float(base_init), device=device, dtype=dtype))
        self.amp_softplus_beta = float(amp_softplus_beta)
        self.base_softplus_beta = float(base_softplus_beta)
        self.amp_scale = float(amp_scale)
        self.use_shifted_softplus_base = bool(use_shifted_softplus_base)

    def set_base_trainable(self, trainable: bool) -> None:
        self.base_raw.requires_grad_(trainable)

    def value(self) -> torch.Tensor:
        amp = torch.nn.functional.softplus(self.amp, beta=self.amp_softplus_beta)
        if self.use_shifted_softplus_base:
            amp = amp - torch.nn.functional.softplus(
                torch.zeros_like(self.amp), beta=self.amp_softplus_beta
            )
            amp = torch.clamp(amp, min=0.0)
        amp = amp * self.amp_scale
    
        base = torch.nn.functional.softplus(self.base_raw, beta=self.base_softplus_beta)
        if self.use_shifted_softplus_base:
            base = base - torch.nn.functional.softplus(
                torch.zeros_like(self.base_raw), beta=self.base_softplus_beta
            )
            base = torch.clamp(base, min=0.0)
    
        return amp * base

class BatchedUnknownField2D(torch.nn.Module):
    """Wrap multiple independent unknown-source fields for batched rollouts."""
    def __init__(self, fields: List[torch.nn.Module]):
        super().__init__()
        self.fields = torch.nn.ModuleList(list(fields))

    def set_base_trainable(self, trainable: bool) -> None:
        # Propagate to factorized fields if present
        for f in self.fields:
            if hasattr(f, "set_base_trainable"):
                f.set_base_trainable(trainable)

    def value(self) -> torch.Tensor:
        vals = [f.value() if hasattr(f, "value") else f() for f in self.fields]
        return torch.stack(vals, dim=0)

# =============================================================================
# Helpers: data parsing
# =============================================================================

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # np.load returns numpy scalars for 0-d arrays sometimes
    return np.array(x)


def _extract_sensor_series(npz: Dict[str, Any], traj: int = 0, use_noisy: bool = True) -> np.ndarray:
    """
    Return series as [S, T] float32.
    """
    key = "sensor_noisy" if use_noisy and "sensor_noisy" in npz else "sensor_clean"
    if key not in npz:
        raise KeyError(f"NPZ missing sensor series. Expected 'sensor_noisy' or 'sensor_clean'. Found: {list(npz.keys())}")

    arr = _to_numpy(npz[key]).astype(np.float32)
    if arr.ndim == 2:
        # [S,T] or [T,S]
        if arr.shape[0] <= arr.shape[1]:
            # assume [S,T]
            return arr
        # assume [T,S]
        return arr.T
    if arr.ndim == 3:
        # [N,S,T] or [N,T,S]
        arr = arr[traj]
        if arr.shape[0] <= arr.shape[1]:
            # [S,T]
            return arr
        # [T,S]
        return arr.T

    raise ValueError(f"Unsupported sensor series shape {arr.shape} for key '{key}'")


def _extract_sensors(npz: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    sensors_idx = _to_numpy(npz["sensors_idx"]).astype(np.int64) if "sensors_idx" in npz else None
    sensors_xy = _to_numpy(npz["sensors_xy"]).astype(np.float32) if "sensors_xy" in npz else None
    return sensors_idx, sensors_xy


def _infer_grid_hw(npz: Dict[str, Any], sensors_idx: Optional[np.ndarray], default_hw: int = 40) -> Tuple[int, int]:
    # Prefer explicit x,y if present
    if "x" in npz and "y" in npz:
        x = _to_numpy(npz["x"])
        y = _to_numpy(npz["y"])
        return int(len(x)), int(len(y))

    # Otherwise infer from sensors_idx
    if sensors_idx is not None:
        H = int(sensors_idx[:, 0].max() + 1)
        W = int(sensors_idx[:, 1].max() + 1)
        # In case sensors don't cover full domain, fall back if weirdly small
        if H >= 8 and W >= 8:
            return H, W

    return default_hw, default_hw


def _time_split(series_st: np.ndarray, val_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_st: [S,T]
    returns train_st, val_st (both [S,T_split])
    """
    S, T = series_st.shape
    t0 = int(round((1.0 - val_frac) * T))
    t0 = max(2, min(T - 2, t0))
    return series_st[:, :t0], series_st[:, t0:]

# =============================================================================
# Placeholders: STAMP (to fill later)
# =============================================================================

def stamp_dyn_loss_placeholder(*args, **kwargs) -> torch.Tensor:
    # Later: compute consistency between sim-predicted sensor series and MPRNN prior predictions
    return torch.tensor(0.0, device=kwargs.get("device", None) or torch.device("cpu"))

# =============================================================================
# Tuning configuration
# =============================================================================

@dataclass
class TunerConfig:
    npz_path: str
    out_dir: str
    traj: int = 0

    # SimGrad (calibration) knobs
    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 2000
    grad_clip: float = 1.0

    # Data usage
    use_noisy: bool = True
    val_frac: float = 0.3

    # Unknown source parameterization (matches polsim default coarse source)
    unknown_hw: int = 10
    unknown_nonneg: bool = True
    unknown_init: float = 0.0
    unknown_softplus_beta: float = 1.0
    unknown_scale: float = 1.0

    # Optional optimization strategies (all toggled by one flag)
    enable_strategies: bool = True

    # (3) Batch softmin/mean across candidates (only used if batch_size > 1)
    batch_size: int = 64
    softmin_tau: float = 10.0  # 0 => mean across batch; >0 => softmin weighting

    # (2) Time-horizon curriculum
    curriculum_epochs: int = 50
    curriculum_min_frac: float = 0.1  # start using this fraction of the train horizon

    # (4) Adaptive LR schedule (ReduceLROnPlateau on val)
    lr_plateau_patience: int = 10
    lr_plateau_factor: float = 0.5
    lr_plateau_min: float = 1e-6

    # (5) Block optimization: amplitude warmup then spatial detail
    amp_warmup_epochs: int = 50
    base_init: float = 0.5413248546129181  # inv_softplus(1.0)

    # (6) Mean-drift penalty (strong early, decays to 0)
    lambda_drift_i: float = 1e3
    lambda_drift_f: float = 10.0
    drift_decay_epochs: int = 200

    # (1) Reparameterization choice for the base field (shifted softplus to remove 0.693 offset)
    use_shifted_softplus: bool = True



    # Regularization
    obj: ObjectiveConfig = field(default_factory=lambda: ObjectiveConfig(
        data_kind="mse",
        lambda_data=1.0,
        lambda_tv=1e-3,
        lambda_lap=1e-2,
        lambda_l2=1e-4,
        lambda_box=1.0,
        box_lo=None,
        box_hi=None,
    ))
    reg_tv_on_unknown: bool = False  # if True, reg_field = S_unknown

    # STAMP flag (placeholder only for now)
    use_stamp: bool = False
    lambda_stamp: float = 0.0

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # Rollout controls (fallbacks if not stored in npz)
    dt: float = 1.0
    steps: int = 200
    save_every: int = 1

# =============================================================================
# Config
# =============================================================================

CFG = TunerConfig(
    npz_path="/scratch/ab9738/stamp/data/pol_dataset.npz",   # <-- change
    out_dir="/scratch/ab9738/stamp/logs/calib_pol_run1",            # <-- change
    traj=0,

    epochs=300,
    lr=3e-4,
    wd=1e-4,
    grad_clip=1.0,

    use_noisy=True,
    val_frac=0.3,

    unknown_hw=10,
    unknown_nonneg=True,
    unknown_init=0.0,

    # regularizers (edit these as needed)
    reg_tv_on_unknown=True,
)
# You can also set objective weights here:
# CFG.obj.lambda_tv = 1e-4
# CFG.obj.lambda_lap = 0.0
# CFG.obj.lambda_l2 = 0.0

# STAMP placeholder controls
CFG.use_stamp = False
CFG.lambda_stamp = 0.0
CFG.unknown_scale = 1

# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)
    log_path = os.path.join(CFG.out_dir, "tuner_pol.log")
    logger = setup_logger(log_path, name="tuner_pol")
    logger.info(f"Loading NPZ: {CFG.npz_path}")

    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    device = torch.device(CFG.device)

    npz = load_npz(CFG.npz_path)

    # Load series + sensors
    series_st = _extract_sensor_series(npz, traj=CFG.traj, use_noisy=CFG.use_noisy)  # [S,T]
    sensors_idx, sensors_xy = _extract_sensors(npz)
    H, W = _infer_grid_hw(npz, sensors_idx, default_hw=40)

    logger.info(f"series shape [S,T]={series_st.shape}, grid inferred [H,W]=({H},{W})")
    if sensors_idx is not None:
        logger.info(f"sensors_idx shape={sensors_idx.shape}")
    if sensors_xy is not None:
        logger.info(f"sensors_xy shape={sensors_xy.shape}")

    # Train/val time split (for calibration and monitoring)
    train_st, val_st = _time_split(series_st, val_frac=CFG.val_frac)
    train = torch.from_numpy(train_st).to(device)
    val = torch.from_numpy(val_st).to(device)

    # ----------------------------
    # Infer rollout controls from dataset if not stored
    # ----------------------------
    t_np = _to_numpy(npz["t"]).astype(float)  # shape [T] or [T_full]
    if t_np.ndim != 1 or t_np.size < 2:
        raise ValueError("Expected npz['t'] to be 1D with length >= 2")
    
    # robust dt from time grid
    dt_data = float(np.median(np.diff(t_np)))
    
    # observed horizon length (train+val after your split)
    T_obs = int(train.shape[1] + val.shape[1])
    
    # Use dataset-derived defaults unless user explicitly overrides
    dt_cfg = CFG.dt
    steps_cfg = CFG.steps
    save_every_cfg = CFG.save_every
    
    # If CFG left at defaults (dt=1.0 etc.), override from data
    if ("dt" not in npz) and (dt_cfg == 1.0):
        CFG.dt = dt_data
    if ("steps" not in npz) and (steps_cfg == 200):
        CFG.steps = T_obs - 1
    if ("save_every" not in npz) and (save_every_cfg == 1):
        CFG.save_every = 1
    
    logger.info(f"Using rollout controls: dt={CFG.dt:.3e}, steps={CFG.steps}, save_every={CFG.save_every}")

    

    # Sigma weighting (optional). For now: None (you can pass your learned/estimated sigma later).
    sigma = None
    mask = None

    # Observer
    if sensors_idx is not None:
        observer = IndexObserver(torch.from_numpy(sensors_idx).to(device))
    elif sensors_xy is not None:
        observer = BilinearObserver(torch.from_numpy(sensors_xy).to(device))
    else:
        raise ValueError("Need either sensors_idx or sensors_xy in NPZ for observation.")

    # Build simulator grid/params
    # NOTE: PolGrid/PolParams construction details may differ; we keep it minimal and rely on defaults.
    # Build simulator grid/params (pollution)
    # PolGrid needs Nx, Ny and (optionally) src_dir for loading known sources / U0 internally.
        # Build simulator grid/params (match data/poldata.py behavior)
    # Infer SIM_DIR from the NPZ location: <root>/data/*.npz -> <root>/sim
    root_dir = os.path.abspath(os.path.join(os.path.dirname(CFG.npz_path), ".."))
    sim_dir = os.path.join(root_dir, "sim")

    grid = polsim.make_grid(
        Nx=H,
        Ny=W,
        src_dir=sim_dir,
        device=device,
        dtype=torch.float32,
        load_sources=True,
    )

    # If dataset stored S_known explicitly, prefer it (guarantees exact consistency)
    if "S_known" in npz:
        grid.S_known = torch.from_numpy(_to_numpy(npz["S_known"]).astype(np.float32)).to(device)

    # Params (use defaults, but move k to correct device if you implemented PolParams.to())
    params = polsim.PolParams()
    if hasattr(params, "to"):
        params = params.to(device)

    # If dataset stored U0 explicitly, prefer it; otherwise let polsim build internally if needed
    if "U0" in npz:
        grid.U0 = torch.from_numpy(_to_numpy(npz["U0"]).astype(np.float32)).to(device)


    # Try to load params if saved
    if "param_names" in npz and "params" in npz:
        try:
            names = [str(x) for x in _to_numpy(npz["param_names"]).tolist()]
            vals = _to_numpy(npz["params"]).astype(np.float32)
            if vals.ndim == 1:
                pmap = {names[i]: float(vals[i]) for i in range(min(len(names), len(vals)))}
            else:
                # If multiple trajectories, pick traj
                pmap = {names[i]: float(vals[CFG.traj, i]) for i in range(min(len(names), vals.shape[1]))}
            logger.info(f"Found saved params: {pmap}")
            # If your PolParams supports setting attributes, do it
            if params is not None:
                for k, v in pmap.items():
                    if hasattr(params, k):
                        setattr(params, k, v)
        except Exception as e:
            logger.info(f"Could not apply saved params (non-fatal): {e}")

    # Learnable unknown source on coarse grid (polsim internally upsamples/smooths)
    params_md = torch.nn.ModuleDict()

    if CFG.enable_strategies:
        # (5) Factorized parameterization: amp first, then spatial detail
        if CFG.batch_size > 1:
            fields = [
                FactorizedUnknownField2D(
                    name="S_unknown",
                    shape_hw=(CFG.unknown_hw, CFG.unknown_hw),
                    amp_init=CFG.unknown_init,
                    base_init=CFG.base_init,
                    amp_softplus_beta=CFG.unknown_softplus_beta,
                    base_softplus_beta=CFG.unknown_softplus_beta,
                    amp_scale=CFG.unknown_scale,
                    device=device,
                    dtype=torch.float32,
                    use_shifted_softplus_base=CFG.use_shifted_softplus,
                )
                for _ in range(int(CFG.batch_size))
            ]
            su = BatchedUnknownField2D(fields)
            # Start with base frozen (amp-only warmup). We'll unfreeze later.
            su.set_base_trainable(False)
        else:
            su = FactorizedUnknownField2D(
                name="S_unknown",
                shape_hw=(CFG.unknown_hw, CFG.unknown_hw),
                amp_init=CFG.unknown_init,
                base_init=CFG.base_init,
                amp_softplus_beta=CFG.unknown_softplus_beta,
                base_softplus_beta=CFG.unknown_softplus_beta,
                amp_scale=CFG.unknown_scale,
                device=device,
                dtype=torch.float32,
                use_shifted_softplus_base=CFG.use_shifted_softplus,
            )
            # Start with base frozen (amp-only warmup). We'll unfreeze later.
            su.set_base_trainable(False)
        params_md["S_unknown"] = su
    else:
        if CFG.batch_size > 1:
            fields = [
                LearnableField2D(
                    name="S_unknown",
                    shape_hw=(CFG.unknown_hw, CFG.unknown_hw),
                    init=CFG.unknown_init,
                    nonneg=CFG.unknown_nonneg,
                    clamp=None,
                    softplus_beta=CFG.unknown_softplus_beta,
                    scale=CFG.unknown_scale,
                    device=device,
                    dtype=torch.float32,
                )
                for _ in range(int(CFG.batch_size))
            ]
            params_md["S_unknown"] = BatchedUnknownField2D(fields)
        else:
            params_md["S_unknown"] = LearnableField2D(
                name="S_unknown",
                shape_hw=(CFG.unknown_hw, CFG.unknown_hw),
                init=CFG.unknown_init,
                nonneg=CFG.unknown_nonneg,
                clamp=None,
                softplus_beta=CFG.unknown_softplus_beta,
                scale=CFG.unknown_scale,
                device=device,
                dtype=torch.float32,
            )

    # Simulator adapter: field key in polsim output is assumed "U" (per your prior context).
    sim = SimulatorAdapter(
        rollout_fn=polsim.rollout_pollution,  # expects S_unknown among args (we will pass below)
        field_key="U",
    )

    calib = Calibrator(sim=sim, observer=observer, params=params_md).to(device)

    opt = torch.optim.AdamW(calib.parameters(), lr=CFG.lr, weight_decay=CFG.wd)


    scheduler = None
    if CFG.enable_strategies:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=CFG.lr_plateau_factor,
            patience=CFG.lr_plateau_patience,
            min_lr=CFG.lr_plateau_min,
            verbose=False,
        )
    n_params = sum(p.numel() for p in calib.parameters())
    n_trainable = sum(p.numel() for p in calib.parameters() if p.requires_grad)
    # logger.info(f"params: total={n_params} trainable={n_trainable}")

    # for name, p in calib.named_parameters():
    #     logger.info(f"param {name}: shape={tuple(p.shape)} requires_grad={p.requires_grad}")


    best_val = float("inf")
    best_state = None

    logger.info(
        "Begin SimGrad tuning | "
        f"epochs={CFG.epochs} lr={CFG.lr} wd={CFG.wd} clip={CFG.grad_clip} "
        f"val_frac={CFG.val_frac} unknown_hw={CFG.unknown_hw} nonneg={CFG.unknown_nonneg} "
        f"stamp={CFG.use_stamp} lambda_stamp={CFG.lambda_stamp}"
    )

    # Fixed kwargs for polsim.rollout_pollution
    fixed_sim_kwargs: Dict[str, Any] = {
        "grid": grid,
        "params": params,
        "dt": float(_to_numpy(npz["dt"])) if "dt" in npz else CFG.dt,
        "steps": int(_to_numpy(npz["steps"])) if "steps" in npz else CFG.steps,
        "save_every": int(_to_numpy(npz["save_every"])) if "save_every" in npz else CFG.save_every,
        "enforce_cfl": True,
    }

    # Some datasets may store U0; polsim may or may not accept it (defensive)
    if "U0" in npz:
        fixed_sim_kwargs["U0"] = torch.from_numpy(_to_numpy(npz["U0"]).astype(np.float32)).to(device)

    # Training loop
    patience = 30
    bad = 0


    # ------------------------------------------------------------------
    # Regularization schedules (fast early, stable later)
    # We keep CFG.obj as the "target" (final) weights and create an
    # epoch-specific ObjectiveConfig that we pass into compute_total_objective.
    #   - box: strong early, decays to 0 over drift_decay_epochs
    #   - tv/lap: off during early warmup, then ramp up to target by drift_decay_epochs
    #   - l2: constant (as in CFG.obj)
    # ------------------------------------------------------------------
    obj_target = CFG.obj
    tv_target = float(obj_target.lambda_tv)
    lap_target = float(obj_target.lambda_lap)
    l2_target = float(obj_target.lambda_l2)
    box_target = float(obj_target.lambda_box)

    # Use a small warmup before turning on spatial smoothness; tie to amp warmup if available.
    reg_warmup_epochs = int(min(10, max(0, CFG.amp_warmup_epochs)))

    def _clone_objective_cfg(cfg: ObjectiveConfig) -> ObjectiveConfig:
        return ObjectiveConfig(
            data_kind=cfg.data_kind,
            huber_delta=cfg.huber_delta,
            lambda_data=cfg.lambda_data,
            lambda_tv=cfg.lambda_tv,
            lambda_lap=cfg.lambda_lap,
            lambda_l2=cfg.lambda_l2,
            lambda_box=cfg.lambda_box,
            box_lo=cfg.box_lo,
            box_hi=cfg.box_hi,
        )

    def _lin_01(epoch_i: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return 1.0
        return min(1.0, max(0.0, float(epoch_i) / float(total_epochs - 1)))

    for epoch in range(1, CFG.epochs + 1):
        calib.train()
        opt.zero_grad(set_to_none=True)

        # (5) Block optimization: amp warmup then spatial detail
        if CFG.enable_strategies:
            su_mod = None
            if hasattr(calib, "params") and isinstance(calib.params, torch.nn.ModuleDict):
                if "S_unknown" in calib.params:
                    su_mod = calib.params["S_unknown"]
            if isinstance(su_mod, FactorizedUnknownField2D):
                su_mod.set_base_trainable(epoch > CFG.amp_warmup_epochs)
            elif hasattr(su_mod, "fields"):
                # Batched factorized fields
                su_mod.set_base_trainable(epoch > CFG.amp_warmup_epochs)

        # Forward sim on full horizon; then slice to train/val portions
        out = calib(return_field=False, **fixed_sim_kwargs)
        pred_full = out.pred_sensor  # [S,T_full] or [B,S,T_full]
        is_batched = (pred_full.dim() == 3)

        # Align pred to observed train/val lengths
        # We assume the dataset's sensor series length equals simulator snapshot count.
        T_train = train.shape[1]
        T_val = val.shape[1]

        if (pred_full.shape[2] if is_batched else pred_full.shape[1]) < (T_train + T_val):
            # If sim produces fewer steps than observation, trim obs
            T_avail = pred_full.shape[2] if is_batched else pred_full.shape[1]
            T_train = min(T_train, max(2, int((1.0 - CFG.val_frac) * T_avail)))
            T_val = min(T_val, T_avail - T_train)
            train_use = train[:, :T_train]
            val_use = val[:, :T_val]
            pred_train = pred_full[:, :, :T_train] if is_batched else pred_full[:, :T_train]
            pred_val = pred_full[:, :, T_train:T_train + T_val] if is_batched else pred_full[:, T_train:T_train + T_val]
        else:
            train_use = train
            val_use = val
            pred_train = pred_full[:, :, :T_train] if is_batched else pred_full[:, :T_train]
            pred_val = pred_full[:, :, T_train:T_train + T_val] if is_batched else pred_full[:, T_train:T_train + T_val]


        # (2) Time-horizon curriculum: use only an early prefix of the train window initially
        train_cur = train_use
        pred_train_cur = pred_train
        if CFG.enable_strategies:
            frac = CFG.curriculum_min_frac
            if CFG.curriculum_epochs > 1:
                t = min(1.0, float(epoch - 1) / float(CFG.curriculum_epochs - 1))
                frac = CFG.curriculum_min_frac + (1.0 - CFG.curriculum_min_frac) * t
            T_eff = max(2, int(round(frac * train_use.shape[1])))
            train_cur = train_use[:, :T_eff]
            pred_train_cur = pred_train[..., :T_eff]

        # print("train_use:", train_use.shape)
        # print("pred_train:", pred_train.shape)
        # print("train_cur:", train_cur.shape)
        # print("pred_train_cur:", pred_train_cur.shape)

        # Build epoch-specific objective (scheduled regularization weights)
        obj_epoch = _clone_objective_cfg(obj_target)

        # Schedule: box decays to 0 over drift_decay_epochs (guardrail early)
        if box_target > 0.0:
            t_box = _lin_01(epoch - 1, int(CFG.drift_decay_epochs))
            obj_epoch.lambda_box = box_target * (1.0 - t_box)

        # Schedule: TV/Laplacian ramp up after warmup to their target by drift_decay_epochs
        if tv_target > 0.0:
            if epoch <= reg_warmup_epochs:
                obj_epoch.lambda_tv = 0.0
            else:
                denom = max(2, int(CFG.drift_decay_epochs) - reg_warmup_epochs)
                t_tv = _lin_01(epoch - reg_warmup_epochs - 1, denom)
                obj_epoch.lambda_tv = tv_target * t_tv

        if lap_target > 0.0:
            if epoch <= reg_warmup_epochs:
                obj_epoch.lambda_lap = 0.0
            else:
                denom = max(2, int(CFG.drift_decay_epochs) - reg_warmup_epochs)
                t_lap = _lin_01(epoch - reg_warmup_epochs - 1, denom)
                obj_epoch.lambda_lap = lap_target * t_lap

        # L2 stays constant (tie-breaker / stability)
        obj_epoch.lambda_l2 = l2_target

        # Decide what to regularize this epoch. For field regularizers (TV/Lap) we use reg_field.
        # For tensor regularizers (L2/box) we use reg_tensor (defaults to reg_field if not provided in objectives.py).
        any_field_reg = (obj_epoch.lambda_tv > 0.0) or (obj_epoch.lambda_lap > 0.0)
        any_tensor_reg = (obj_epoch.lambda_l2 > 0.0) or ((obj_epoch.lambda_box > 0.0) and (obj_epoch.box_lo is not None or obj_epoch.box_hi is not None))

        reg_field = out.theta["S_unknown"] if (CFG.reg_tv_on_unknown or any_field_reg) else None
        reg_tensor = out.theta["S_unknown"] if (any_tensor_reg) else None

        Su = out.theta["S_unknown"]
        logger.info(
            f"S_unknown finite={torch.isfinite(Su).all().item()} "
            f"min={Su.min().item():.3e} max={Su.max().item():.3e}"
        )

        logger.info(f"obs finite={torch.isfinite(train_use).all().item()} pred finite={torch.isfinite(pred_train).all().item()}")
        logger.info(f"obs train mean/std = {train_use.mean().item():.3e} / {train_use.std().item():.3e}")
        logger.info(f"pred train mean/std = {pred_train.mean().item():.3e} / {pred_train.std().item():.3e}")

        # ---- sanity checks right before objective ----
        def _stats(tag, x):
            return (f"{tag}: finite={torch.isfinite(x).all().item()} "
                    f"min={x.min().item():.3e} max={x.max().item():.3e}")

        if is_batched:
            # Vectorized objective over candidate batch (no Python loop)
            base_loss_b, logs = compute_total_objective_batched(
                pred_sensor=pred_train_cur,
                obs_sensor=train_cur,
                cfg=obj_epoch,
                sigma=sigma,
                mask=mask,
                reg_field=reg_field,
                reg_tensor=reg_tensor,
            )

            total_b = base_loss_b

            # (6) Mean-drift penalty (early), applied per candidate before aggregation
            if CFG.enable_strategies and CFG.lambda_drift_i > 0.0:
                t = 0.0
                if CFG.drift_decay_epochs > 1:
                    t = min(1.0, float(epoch - 1) / float(CFG.drift_decay_epochs - 1))
                lam = max(float(CFG.lambda_drift_i) * (1.0 - t), CFG.lambda_drift_f)
                if lam > 0.0:
                    drift_err_b = pred_train_cur.mean(dim=(1, 2)) - train_cur.mean()
                    drift_b = drift_err_b * drift_err_b
                    total_b = total_b + lam * drift_b
                    logs["loss_drift"] = float(drift_b.mean().detach().cpu().item())
                    logs["lambda_drift"] = float(lam)

            # Aggregate across candidates: mean (tau<=0) or softmin (tau>0)
            loss_train, w = softmin_aggregate(total_b, tau=CFG.softmin_tau)
            logs["batch_tau"] = float(CFG.softmin_tau)
            # Effective number of candidates (1 / sum w^2); useful to monitor collapse
            logs["batch_eff"] = float((1.0 / (w.pow(2).sum().clamp_min(1e-12))).detach().cpu().item())

        else:
            loss_train, logs = compute_total_objective(
                pred_sensor=pred_train_cur,
                obs_sensor=train_cur,
                cfg=obj_epoch,
                sigma=sigma,
                mask=mask,
                reg_field=reg_field,
                reg_tensor=reg_tensor,
            )

            # (6) Mean-drift penalty (early), helps prevent long-horizon upward drift
            if CFG.enable_strategies and CFG.lambda_drift_i > 0.0:
                t = 0.0
                if CFG.drift_decay_epochs > 1:
                    t = min(1.0, float(epoch - 1) / float(CFG.drift_decay_epochs - 1))
                lam = max(float(CFG.lambda_drift_i) * (1.0 - t), CFG.lambda_drift_f)
                if lam > 0.0:
                    drift_err = (pred_train_cur.mean() - train_cur.mean())
                    loss_train = loss_train + lam * (drift_err * drift_err)
                    logs["loss_drift"] = float((drift_err * drift_err).detach().cpu().item())
                    logs["lambda_drift"] = float(lam)
        if not torch.isfinite(loss_train):
            raise RuntimeError("Non-finite loss_total")

        # Optional STAMP term (placeholder)
        if CFG.use_stamp and CFG.lambda_stamp > 0.0:
            L_stamp = stamp_dyn_loss_placeholder(device=device)
            loss_train = loss_train + CFG.lambda_stamp * L_stamp
            logs["loss_stamp"] = float(L_stamp.detach().cpu().item())
            logs["lambda_stamp"] = float(CFG.lambda_stamp)

        loss_train.backward()

        # for name, p in calib.named_parameters():
        #     if p.grad is None:
        #         logger.info(f"grad {name}: None")
        #     else:
        #         logger.info(f"grad {name}: norm={p.grad.norm().item():.3e}")
                
        if CFG.grad_clip is not None and CFG.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(calib.parameters(), CFG.grad_clip)
        opt.step()

        with torch.no_grad():
            su_mod = calib.params["S_unknown"] if (hasattr(calib, "params") and isinstance(calib.params, torch.nn.ModuleDict) and "S_unknown" in calib.params) else None
            if hasattr(su_mod, "fields"):
                for f in su_mod.fields:
                    if hasattr(f, "raw"):
                        f.raw.clamp_(-1.0, 1.0)
                    if isinstance(f, FactorizedUnknownField2D):
                        f.amp.clamp_(-1.0, 1.0)
                        f.base_raw.clamp_(-1.0, 1.0)
            else:
                if hasattr(su_mod, "raw"):
                    su_mod.raw.clamp_(-1.0, 1.0)
                if isinstance(su_mod, FactorizedUnknownField2D):
                    su_mod.amp.clamp_(-1.0, 1.0)
                    su_mod.base_raw.clamp_(-1.0, 1.0)

        # Validation
        calib.eval()
        with torch.no_grad():
            out2 = calib(return_field=False, **fixed_sim_kwargs)
            pred_full2 = out2.pred_sensor
            is_batched2 = (pred_full2.dim() == 3)
            if is_batched2:
                pred_val2 = pred_full2[:, :, T_train:T_train + T_val] if pred_full2.shape[2] >= (T_train + T_val) else pred_full2[:, :, -T_val:]
            else:
                pred_val2 = pred_full2[:, T_train:T_train + T_val] if pred_full2.shape[1] >= (T_train + T_val) else pred_full2[:, -T_val:]
            val_loss = (pred_val2 - (val_use.unsqueeze(0) if is_batched2 else val_use)).pow(2).mean().item()

        if scheduler is not None:
            # (4) Adaptive LR schedule on validation loss
            scheduler.step(val_loss)

        logs["train_total"] = float(loss_train.detach().cpu().item())
        logs["val_mse"] = float(val_loss)

        # Logging
        # if epoch == 1 or epoch % 5 == 0:
        logger.info(
            f"epoch={epoch:04d} train={logs['train_total']:.6f} val={logs['val_mse']:.6f} "
            + " ".join([f"{k}={v:.4g}" for k, v in logs.items() if k.startswith("loss_") and k not in ("loss_total",)])
        )

        # Early stopping on val
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in out2.theta.items()}
            logger.info(f"  BEST val={best_val:.6f}")
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping at epoch={epoch} (patience={patience}). Best val={best_val:.6f}")
                break

    # Save outputs
    save_path = os.path.join(CFG.out_dir, "calib_pol_best.npz")
    to_save: Dict[str, Any] = {
        "best_val_mse": np.array([best_val], dtype=np.float32),
        "traj": np.array([CFG.traj], dtype=np.int32),
    }
    if best_state is not None:
        for k, v in best_state.items():
            to_save[k] = v.numpy()

    np.savez_compressed(save_path, **to_save)
    logger.info(f"Saved best calibration to: {save_path}")


if __name__ == "__main__":
    main()