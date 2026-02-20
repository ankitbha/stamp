# model/calibrator/objectives.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# =============================================================================
# Core losses
# =============================================================================

def weighted_mse(
    pred: Tensor,                  # [S,T] or [B,S,T] or [N,T,S]
    target: Tensor,                # same shape as pred
    sigma: Optional[Tensor] = None,  # [S] or broadcastable to pred
    mask: Optional[Tensor] = None,   # same shape as pred; 1=valid
    eps: float = 1e-6,
) -> Tensor:
    """
    Weighted mean-squared error.

    If sigma is provided, we weight by 1/(sigma^2 + eps).
      - Typical: sigma is per-sensor [S]
      - Works if sigma can broadcast to pred.

    If mask provided, ignores entries with mask==0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")

    err2 = (pred - target) ** 2

    if sigma is not None:
        # weight = 1/sigma^2
        w = 1.0 / (sigma ** 2 + eps)
        err2 = err2 * w

    if mask is not None:
        err2 = err2 * mask
        denom = mask.sum().clamp_min(1.0)
        return err2.sum() / denom

    return err2.mean()


def huber_loss(
    pred: Tensor,
    target: Tensor,
    delta: float = 1.0,
    sigma: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Robust alternative to MSE. Weighted by 1/sigma^2 if provided.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")

    # elementwise huber
    diff = pred - target
    absd = diff.abs()
    quad = torch.minimum(absd, torch.tensor(delta, device=absd.device, dtype=absd.dtype))
    lin = absd - quad
    loss = 0.5 * quad**2 + delta * lin  # same shape as pred

    if sigma is not None:
        w = 1.0 / (sigma ** 2 + eps)
        loss = loss * w

    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    return loss.mean()


# =============================================================================
# Regularizers for 2D fields
# =============================================================================

def tv_loss_2d(field_hw: Tensor, eps: float = 1e-6, reduction: str = "mean") -> Tensor:
    """
    Isotropic total variation on a 2D field.

    field_hw: [H,W] or [B,H,W]
    """
    if field_hw.ndim == 2:
        x = field_hw[None, :, :]
    elif field_hw.ndim == 3:
        x = field_hw
    else:
        raise ValueError(f"tv_loss_2d expects [H,W] or [B,H,W], got {tuple(field_hw.shape)}")

    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    tv = torch.sqrt(dx[:, :-1, :] ** 2 + dy[:, :, :-1] ** 2 + eps)  # align shapes

    if reduction == "mean":
        return tv.mean()
    if reduction == "sum":
        return tv.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def laplacian_smoothness_2d(field_hw: Tensor, reduction: str = "mean") -> Tensor:
    """
    L2 penalty on the discrete Laplacian (encourages smoothness).

    field_hw: [H,W] or [B,H,W]
    """
    if field_hw.ndim == 2:
        x = field_hw[None, None, :, :]  # [1,1,H,W]
    elif field_hw.ndim == 3:
        x = field_hw[:, None, :, :]     # [B,1,H,W]
    else:
        raise ValueError(f"laplacian_smoothness_2d expects [H,W] or [B,H,W], got {tuple(field_hw.shape)}")

    # 5-point Laplacian kernel
    k = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    )[None, None, :, :]  # [1,1,3,3]

    lap = F.conv2d(x, k, padding=1)  # [B,1,H,W]
    val = (lap ** 2)

    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def l2_loss(x: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "mean":
        return (x ** 2).mean()
    if reduction == "sum":
        return (x ** 2).sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def box_penalty(
    x: Tensor,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Soft penalty for violating [lo, hi] bounds.
    Uses squared hinge: max(0, lo-x)^2 + max(0, x-hi)^2
    """
    pen = 0.0
    if lo is not None:
        pen = pen + F.relu(lo - x) ** 2
    if hi is not None:
        pen = pen + F.relu(x - hi) ** 2

    if isinstance(pen, float):
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


# =============================================================================
# STAMP-style dynamics-consistency hook
# =============================================================================

def dyn_consistency_loss(
    prior_fn: Callable[..., Tensor],
    series: Tensor,
    edge_index: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
    loss_kind: str = "mse",
) -> Tensor:
    """
    Generic dynamics consistency loss:
      - prior_fn predicts next-step given a window/series
      - we measure mismatch between simulated series and prior predictions

    This is intentionally abstract. Tuners decide how to prepare inputs.

    Expected usage pattern (in tuner):
      pred_next = prior_fn(x_in, edge_index, edge_attr)  # shape same as y_true
      L = mse(pred_next, y_true)

    Here we provide a placeholder to keep objectives.py complete; you can ignore
    this until we wire the prior interface in tuners.
    """
    raise NotImplementedError(
        "dyn_consistency_loss is a hook; implement it in tuner once prior_fn interface is finalized."
    )


# =============================================================================
# Objective composition
# =============================================================================

@dataclass
class ObjectiveConfig:
    # data term
    data_kind: str = "mse"          # "mse" or "huber"
    huber_delta: float = 1.0
    lambda_data: float = 1.0

    # regularizers (optional)
    lambda_tv: float = 0.0
    lambda_lap: float = 0.0
    lambda_l2: float = 0.0
    lambda_box: float = 0.0

    # box penalty bounds (optional)
    box_lo: Optional[float] = None
    box_hi: Optional[float] = None


def compute_total_objective(
    pred_sensor: Tensor,                # [S,T]
    obs_sensor: Tensor,                 # [S,T]
    cfg: ObjectiveConfig,
    sigma: Optional[Tensor] = None,     # [S] broadcastable
    mask: Optional[Tensor] = None,      # [S,T]
    reg_field: Optional[Tensor] = None, # [H,W] or [B,H,W] to regularize (e.g., S_unknown)
    reg_tensor: Optional[Tensor] = None,# any tensor for L2/box if needed
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute total loss and a dict of scalars for logging.

    reg_field: a 2D field regularized with TV/Laplacian (e.g., unknown sources)
    reg_tensor: used for L2 and/or box penalties (can be same as reg_field)
    """
    logs: Dict[str, float] = {}

    # data term
    if cfg.data_kind == "mse":
        L_data = weighted_mse(pred_sensor, obs_sensor, sigma=sigma, mask=mask)
    elif cfg.data_kind == "huber":
        L_data = huber_loss(pred_sensor, obs_sensor, delta=cfg.huber_delta, sigma=sigma, mask=mask)
    else:
        raise ValueError(f"Unknown data_kind: {cfg.data_kind}")

    if not torch.isfinite(L_data):
        raise RuntimeError("Non-finite loss_data")
    loss = cfg.lambda_data * L_data
    
    logs["loss_data"] = float(L_data.detach().cpu().item())
    logs["lambda_data"] = float(cfg.lambda_data)

    # TV / Laplacian regularizers
    if reg_field is not None and cfg.lambda_tv > 0.0:
        L_tv = tv_loss_2d(reg_field)
        if not torch.isfinite(L_tv):
            raise RuntimeError("Non-finite loss_tv")
        loss = loss + cfg.lambda_tv * L_tv
        logs["loss_tv"] = float(L_tv.detach().cpu().item())
        logs["lambda_tv"] = float(cfg.lambda_tv)

    if reg_field is not None and cfg.lambda_lap > 0.0:
        L_lap = laplacian_smoothness_2d(reg_field)
        if not torch.isfinite(L_lap):
            raise RuntimeError("Non-finite loss_laplacian")
        loss = loss + cfg.lambda_lap * L_lap
        logs["loss_lap"] = float(L_lap.detach().cpu().item())
        logs["lambda_lap"] = float(cfg.lambda_lap)

    # L2 / box penalties
    rt = reg_tensor if reg_tensor is not None else reg_field
    if rt is not None and cfg.lambda_l2 > 0.0:
        L_l2 = l2_loss(rt)
        if not torch.isfinite(L_l2):
            raise RuntimeError("Non-finite loss_l2")
        loss = loss + cfg.lambda_l2 * L_l2
        logs["loss_l2"] = float(L_l2.detach().cpu().item())
        logs["lambda_l2"] = float(cfg.lambda_l2)

    if rt is not None and cfg.lambda_box > 0.0 and (cfg.box_lo is not None or cfg.box_hi is not None):
        L_box = box_penalty(rt, lo=cfg.box_lo, hi=cfg.box_hi)
        loss = loss + cfg.lambda_box * L_box
        logs["loss_box"] = float(L_box.detach().cpu().item())
        logs["lambda_box"] = float(cfg.lambda_box)

    logs["loss_total"] = float(loss.detach().cpu().item())
    return loss, logs
