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
    BatchedLearnableField2D,
    IndexObserver,
    BilinearObserver,
)
from model.calibrator.objectives import ObjectiveConfig, compute_total_objective
from model.calibrator.objectives import tv_loss_2d, laplacian_smoothness_2d, l2_loss

# Simulator
import sim.polsim as polsim

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
    epochs: int = 300
    grad_clip: float = 1.0

    # Data usage
    use_noisy: bool = True
    val_frac: float = 0.3

    # Unknown source parameterization (matches polsim default coarse source)
    unknown_hw: int = 10
    unknown_nonneg: bool = True
    unknown_init: float = 0.0
    unknown_softplus_beta: float = 1.0

    # Regularization
    obj: ObjectiveConfig = field(default_factory=lambda: ObjectiveConfig(
        data_kind="mse",
        lambda_data=1.0,
        lambda_tv=0.0,
        lambda_lap=0.0,
        lambda_l2=0.0,
        lambda_box=0.0,
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
    batch: int = 8
    batch_seed_offset: int = 1000

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
    unknown_init= -4.0,

    # regularizers (edit these as needed)
    reg_tv_on_unknown=True,
)
# You can also set objective weights here:
CFG.obj.lambda_tv = 1e-4
CFG.obj.lambda_lap = 0.0
CFG.obj.lambda_l2 = 0.0

# STAMP placeholder controls
CFG.use_stamp = False
CFG.lambda_stamp = 0.0
CFG.unknown_scale = 0.2

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

    S_known = grid.S_known
    rms_known = torch.sqrt(torch.mean(S_known**2))
    logger.info(f"RMS(S_known) = {rms_known.item():.6f}")

    # Params (use defaults, but move k to correct device if you implemented PolParams.to())
    params = polsim.PolParams()
    if hasattr(params, "to"):
        params = params.to(device)

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
    # Batched multi-start: B independent S_unknown candidates optimized in parallel.
    params_md["S_unknown"] = BatchedLearnableField2D(
        B=CFG.batch,
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

    # Optional: diversify initializations across batch (multi-start)
    # (Safe even if BatchedLearnableField2D already randomizes internally.)
    with torch.no_grad():
        su = calib.params.S_unknown
        if hasattr(su, "raws") and isinstance(getattr(su, "raws"), (list, tuple)):
            for b, r in enumerate(su.raws):
                cuda_devs = [torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=cuda_devs):
                    torch.manual_seed(CFG.seed + CFG.batch_seed_offset + b)
                    r.add_(0.01 * torch.randn_like(r))
        elif hasattr(su, "raw"):
            su.raw.add_(0.01 * torch.randn_like(su.raw))

    opt = torch.optim.AdamW(calib.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

    n_params = sum(p.numel() for p in calib.parameters())
    n_trainable = sum(p.numel() for p in calib.parameters() if p.requires_grad)
    logger.info(f"params: total={n_params} trainable={n_trainable}")

    for name, p in calib.named_parameters():
        logger.info(f"param {name}: shape={tuple(p.shape)} requires_grad={p.requires_grad}")

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

    # with torch.no_grad():
    #     Su0 = torch.zeros((CFG.batch, CFG.unknown_hw, CFG.unknown_hw), device=device)
    #     out0 = polsim.rollout_pollution(S_unknown=Su0, **fixed_sim_kwargs)  # dict with "U"
    #     U0snap = out0["U"]  # [B,Nx,Ny,T]
    #     pred0 = observer(U0snap)  # should be [B,S,T]
    #     # compare to obs (train+val concatenated)
    #     obs = torch.cat([train, val], dim=1).unsqueeze(0).expand(CFG.batch, -1, -1)  # [B,S,T]
    #     print("obs mean/std", obs.mean().item(), obs.std().item())
    #     print("pred0 mean/std", pred0.mean().item(), pred0.std().item())
    #     print("ratio mean", (pred0.mean()/obs.mean()).item())

    # Training loop
    patience = 30
    bad = 0

    for epoch in range(1, CFG.epochs + 1):
        calib.train()
        opt.zero_grad(set_to_none=True)

        # Forward sim on full horizon; then slice to train/val portions
        out = calib(return_field=False, **fixed_sim_kwargs)
        pred_full = out.pred_sensor  # [B,S,T_full]

        # Align pred to observed train/val lengths
        # We assume the dataset's sensor series length equals simulator snapshot count.
        T_train = train.shape[1]
        T_val = val.shape[1]
        
        if pred_full.shape[-1] < (T_train + T_val):
            # If sim produces fewer steps than observation, trim obs
            T_avail = pred_full.shape[-1]
            T_train = min(T_train, max(2, int((1.0 - CFG.val_frac) * T_avail)))
            T_val = min(T_val, T_avail - T_train)
            B = pred_full.shape[0]
            train_use = train[:, :T_train].unsqueeze(0).expand(B, -1, -1)  # [B,S,T_train]
            val_use = val[:, :T_val].unsqueeze(0).expand(B, -1, -1)        # [B,S,T_val]
            pred_train = pred_full[:, :, :T_train]                         # [B,S,T_train]
            pred_val = pred_full[:, :, T_train:T_train + T_val]
        else:
            B = pred_full.shape[0]
            train_use = train.unsqueeze(0).expand(B, -1, -1)               # [B,S,T_train]
            val_use = val.unsqueeze(0).expand(B, -1, -1)                   # [B,S,T_val]
            pred_train = pred_full[:, :, :T_train]                         # [B,S,T_train]
            pred_val = pred_full[:, :, T_train:T_train + T_val]

        reg_field = out.theta["S_unknown"] if (CFG.reg_tv_on_unknown or CFG.obj.lambda_tv > 0.0 or CFG.obj.lambda_lap > 0.0) else None
        reg_tensor = out.theta["S_unknown"] if (CFG.obj.lambda_l2 > 0.0) else None

        Su = out.theta["S_unknown"]
        logger.info(
            f"S_unknown finite={torch.isfinite(Su).all().item()} "
            f"min={Su.min().item():.3e} max={Su.max().item():.3e}"
            f"S_unknown RMS = {Su.pow(2).mean().sqrt().item():.6f}"
        )

        # logger.info(f"obs finite={torch.isfinite(train_use).all().item()} pred finite={torch.isfinite(pred_train).all().item()}")
        # logger.info(f"pred_full shape = {tuple(pred_full.shape)}")
        # logger.info(f"train_use shape {train_use.shape} val_use shape {val_use.shape}")
        # logger.info(f"pred_train shape {pred_train.shape} pred_val shape {pred_val.shape}")
        # logger.info(f"T_train={T_train} T_val={T_val}")
        logger.info(f"obs train mean/std = {train_use.mean().item():.3e} / {train_use.std().item():.3e}")
        logger.info(f"pred train mean/std = {pred_train.mean().item():.3e} / {pred_train.std().item():.3e}")


        diff = pred_train - train_use                       # [B,S,T]
        loss_data_b = diff.pow(2).mean(dim=(1, 2))          # [B]
        b = loss_data_b.argmin().item()
        loss_train, logs = compute_total_objective(
            pred_sensor=pred_train,
            obs_sensor=train_use,
            cfg=CFG.obj,
            sigma=sigma,
            mask=mask,
            reg_field=reg_field,
            reg_tensor=reg_tensor,
        )

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
            su = calib.params.S_unknown
            if hasattr(su, "raws") and isinstance(getattr(su, "raws"), (list, tuple)):
                for r in su.raws:
                    r.clamp_(-10.0, 10.0)
            elif hasattr(su, "raw"):
                su.raw.clamp_(-10.0, 10.0)

        # Validation
        calib.eval()
        with torch.no_grad():
            out2 = calib(return_field=False, **fixed_sim_kwargs)
            pred_full2 = out2.pred_sensor

            if pred_full2.shape[-1] >= (T_train + T_val):
                pred_val2 = pred_full2[:, :, T_train:T_train + T_val]  # [B,S,T_val]
            else:
                pred_val2 = pred_full2[:, :, -T_val:]                   # [B,S,T_val]

            # Per-candidate validation MSE, then select the best candidate.
            val_loss_b = (pred_val2 - val_use).pow(2).mean(dim=(1, 2))  # [B]
            best_b = int(val_loss_b.argmin().item())
            val_loss = float(val_loss_b.mean().item())

        logs["train_total"] = float(loss_train.detach().cpu().item())
        logs["val_mse"] = float(val_loss)
        logs["val_mse_mean"] = float(val_loss_b.mean().item())
        logs["val_mse_min"] = float(val_loss_b.min().item())
        logs["best_b"] = float(best_b)

        logger.info(
            f"epoch={epoch:04d} "
            f"train={logs['train_total']:.6f} "
            f"val={logs['val_mse']:.6f} "
            f"val_mean={logs['val_mse_mean']:.6f} "
            f"val_min={logs['val_mse_min']:.6f} "
            f"best_b={int(logs['best_b'])} "
            f"loss_data={logs.get('loss_data', float('nan')):.6g} "
            f"loss_tv={logs.get('loss_tv', float('nan')):.6g} "
            f"loss_lap={logs.get('loss_lap', float('nan')):.6g} "
            f"loss_l2={logs.get('loss_l2', float('nan')):.6g} "
            f"loss_box={logs.get('loss_box', float('nan')):.6g} "
            f"loss_total={logs.get('loss_total', float('nan')):.6g}"
        )

        # Early stopping on val
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            # Save ONLY the best candidate's parameters (no batch dim)
            best_state = {}
            for k, v in out2.theta.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == val_loss_b.shape[0]:
                    best_state[k] = v[best_b].detach().cpu().clone()
                else:
                    best_state[k] = v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
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
            if isinstance(v, torch.Tensor):
                to_save[k] = v.numpy()

    np.savez_compressed(save_path, **to_save)
    logger.info(f"Saved best calibration to: {save_path}")


if __name__ == "__main__":
    main()