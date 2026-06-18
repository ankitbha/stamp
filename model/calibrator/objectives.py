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




# =============================================================================
# Batch reducers / softmin
# =============================================================================

def softmin_weights(losses: Tensor, tau: float = 0.0) -> Tensor:
    """Return softmin weights over a 1D tensor of losses.

    tau <= 0: uniform weights (equivalent to mean aggregation)
    tau  > 0: w = softmax(-loss/tau)
    """
    if losses.ndim != 1:
        raise ValueError(f"softmin_weights expects 1D losses [B], got {tuple(losses.shape)}")
    B = losses.numel()
    if B == 0:
        raise ValueError("softmin_weights: empty losses")
    if tau is None or float(tau) <= 0.0:
        return torch.full((B,), 1.0 / float(B), device=losses.device, dtype=losses.dtype)
    tau_t = torch.tensor(float(tau), device=losses.device, dtype=losses.dtype)
    # subtract min for numerical stability (softmin = softmax(-loss/tau))
    z = -(losses - losses.min()) / tau_t
    return torch.softmax(z, dim=0)


def softmin_aggregate(losses: Tensor, tau: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Aggregate per-batch losses into a scalar via mean (tau<=0) or softmin (tau>0).

    Returns:
      agg: scalar Tensor
      w:   weights [B] used in aggregation (uniform if tau<=0)
    """
    w = softmin_weights(losses, tau=tau)
    return (w * losses).sum(), w


# =============================================================================
# Batched objectives (no Python loops)
# =============================================================================

def _ensure_batched_obs(pred: Tensor, obs: Tensor) -> Tensor:
    """Broadcast obs to match pred's batch shape when pred is [B,...]."""
    if pred.ndim == obs.ndim:
        return obs
    if pred.ndim == obs.ndim + 1:
        return obs.unsqueeze(0).expand(pred.shape[0], *obs.shape)
    raise ValueError(f"Cannot broadcast obs {tuple(obs.shape)} to pred {tuple(pred.shape)}")


def weighted_mse_batch(
    pred: Tensor,                  # [B,S,T]
    target: Tensor,                # [S,T] or [B,S,T]
    sigma: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Per-batch weighted MSE. Returns [B]."""
    if pred.ndim != 3:
        raise ValueError(f"weighted_mse_batch expects pred [B,S,T], got {tuple(pred.shape)}")
    target_b = _ensure_batched_obs(pred, target)
    if pred.shape != target_b.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target_b.shape)}")

    err2 = (pred - target_b) ** 2

    if sigma is not None:
        w = 1.0 / (sigma ** 2 + eps)
        err2 = err2 * w

    if mask is not None:
        m = _ensure_batched_obs(pred, mask)
        err2 = err2 * m
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        return err2.sum(dim=(1, 2)) / denom

    return err2.mean(dim=(1, 2))


def huber_loss_batch(
    pred: Tensor,                  # [B,S,T]
    target: Tensor,                # [S,T] or [B,S,T]
    delta: float = 1.0,
    sigma: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Per-batch Huber loss. Returns [B]."""
    if pred.ndim != 3:
        raise ValueError(f"huber_loss_batch expects pred [B,S,T], got {tuple(pred.shape)}")
    target_b = _ensure_batched_obs(pred, target)
    if pred.shape != target_b.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target_b.shape)}")

    diff = pred - target_b
    absd = diff.abs()
    delta_t = torch.tensor(delta, device=absd.device, dtype=absd.dtype)
    quad = torch.minimum(absd, delta_t)
    lin = absd - quad
    loss = 0.5 * quad**2 + delta_t * lin

    if sigma is not None:
        w = 1.0 / (sigma ** 2 + eps)
        loss = loss * w

    if mask is not None:
        m = _ensure_batched_obs(pred, mask)
        loss = loss * m
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        return loss.sum(dim=(1, 2)) / denom

    return loss.mean(dim=(1, 2))


def tv_loss_2d_batch(field_hw: Tensor, eps: float = 1e-6) -> Tensor:
    """Isotropic TV per batch. field_hw: [B,H,W] -> [B]."""
    if field_hw.ndim != 3:
        raise ValueError(f"tv_loss_2d_batch expects [B,H,W], got {tuple(field_hw.shape)}")
    x = field_hw
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    tv = torch.sqrt(dx[:, :-1, :] ** 2 + dy[:, :, :-1] ** 2 + eps)
    return tv.mean(dim=(1, 2))


def laplacian_smoothness_2d_batch(field_hw: Tensor) -> Tensor:
    """L2 Laplacian penalty per batch. field_hw: [B,H,W] -> [B]."""
    if field_hw.ndim != 3:
        raise ValueError(f"laplacian_smoothness_2d_batch expects [B,H,W], got {tuple(field_hw.shape)}")
    x = field_hw[:, None, :, :]  # [B,1,H,W]
    k = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    )[None, None, :, :]
    lap = F.conv2d(x, k, padding=1)  # [B,1,H,W]
    return (lap ** 2).mean(dim=(1, 2, 3))


def l2_loss_batch(x: Tensor) -> Tensor:
    """Per-batch L2. x can be [B,...]. Returns [B]."""
    if x.ndim < 1:
        raise ValueError("l2_loss_batch expects at least 1D tensor")
    if x.ndim == 1:
        return x ** 2
    return (x ** 2).mean(dim=tuple(range(1, x.ndim)))


def box_penalty_batch(
    x: Tensor,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> Tensor:
    """Per-batch box penalty (squared hinge). x: [B,...] -> [B]."""
    if x.ndim < 1:
        raise ValueError("box_penalty_batch expects at least 1D tensor")
    pen = 0.0
    if lo is not None:
        pen = pen + F.relu(lo - x) ** 2
    if hi is not None:
        pen = pen + F.relu(x - hi) ** 2
    if isinstance(pen, float):
        return torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    if pen.ndim == 1:
        return pen
    return pen.mean(dim=tuple(range(1, pen.ndim)))


def compute_total_objective_batched(
    pred_sensor: Tensor,                # [B,S,T]
    obs_sensor: Tensor,                 # [S,T] or [B,S,T]
    cfg: ObjectiveConfig,
    sigma: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    reg_field: Optional[Tensor] = None,  # [B,H,W]
    reg_tensor: Optional[Tensor] = None, # [B,...]
) -> Tuple[Tensor, Dict[str, float]]:
    """Vectorized total objective over a candidate batch. Returns (losses[B], logs_mean)."""
    if pred_sensor.ndim != 3:
        raise ValueError(f"compute_total_objective_batched expects pred_sensor [B,S,T], got {tuple(pred_sensor.shape)}")

    logs: Dict[str, float] = {}

    if cfg.data_kind == "mse":
        L_data_b = weighted_mse_batch(pred_sensor, obs_sensor, sigma=sigma, mask=mask)  # [B]
    elif cfg.data_kind == "huber":
        L_data_b = huber_loss_batch(pred_sensor, obs_sensor, delta=cfg.huber_delta, sigma=sigma, mask=mask)
    else:
        raise ValueError(f"Unknown data_kind: {cfg.data_kind}")

    if not torch.isfinite(L_data_b).all():
        raise RuntimeError("Non-finite loss_data (batched)")

    loss_b = cfg.lambda_data * L_data_b
    logs["loss_data"] = float(L_data_b.mean().detach().cpu().item())
    logs["lambda_data"] = float(cfg.lambda_data)

    if reg_field is not None and cfg.lambda_tv > 0.0:
        L_tv_b = tv_loss_2d_batch(reg_field)
        if not torch.isfinite(L_tv_b).all():
            raise RuntimeError("Non-finite loss_tv (batched)")
        loss_b = loss_b + cfg.lambda_tv * L_tv_b
        logs["loss_tv"] = float(L_tv_b.mean().detach().cpu().item())
        logs["lambda_tv"] = float(cfg.lambda_tv)

    if reg_field is not None and cfg.lambda_lap > 0.0:
        L_lap_b = laplacian_smoothness_2d_batch(reg_field)
        if not torch.isfinite(L_lap_b).all():
            raise RuntimeError("Non-finite loss_laplacian (batched)")
        loss_b = loss_b + cfg.lambda_lap * L_lap_b
        logs["loss_lap"] = float(L_lap_b.mean().detach().cpu().item())
        logs["lambda_lap"] = float(cfg.lambda_lap)

    rt = reg_tensor if reg_tensor is not None else reg_field
    if rt is not None and cfg.lambda_l2 > 0.0:
        L_l2_b = l2_loss_batch(rt)
        if not torch.isfinite(L_l2_b).all():
            raise RuntimeError("Non-finite loss_l2 (batched)")
        loss_b = loss_b + cfg.lambda_l2 * L_l2_b
        logs["loss_l2"] = float(L_l2_b.mean().detach().cpu().item())
        logs["lambda_l2"] = float(cfg.lambda_l2)

    if rt is not None and cfg.lambda_box > 0.0 and (cfg.box_lo is not None or cfg.box_hi is not None):
        L_box_b = box_penalty_batch(rt, lo=cfg.box_lo, hi=cfg.box_hi)
        if not torch.isfinite(L_box_b).all():
            raise RuntimeError("Non-finite loss_box (batched)")
        loss_b = loss_b + cfg.lambda_box * L_box_b
        logs["loss_box"] = float(L_box_b.mean().detach().cpu().item())
        logs["lambda_box"] = float(cfg.lambda_box)

    logs["loss_total"] = float(loss_b.mean().detach().cpu().item())
    return loss_b, logs

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
    reg_field: Optional[Tensor] = None, # [H,W] or [B,H,W] to regularize
    reg_tensor: Optional[Tensor] = None,# any tensor for L2/box if needed
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute total loss and a dict of scalars for logging.

    reg_field: a 2D field regularized with TV/Laplacian
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
