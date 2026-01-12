import numpy as np
import torch
import torch.nn as nn
from model.prior.mprnn import MPRNN

# =============================================================================
# Data-driven sigma (heteroscedastic weights)
# =============================================================================

def estimate_sigma_from_series(train_series: np.ndarray, eps: float) -> np.ndarray:
    """
    Data-only sigma_i estimate using robust high-frequency variation.

    Use MAD of first differences:
      diff_i(t) = d_i(t+1) - d_i(t)
      sigma_i ~= MAD(diff_i) / 0.6745   (Gaussian-consistent)

    Returns sigma: [S] float32
    """
    diff = np.diff(train_series, axis=1)  # [S, T-1]
    med = np.median(diff, axis=1, keepdims=True)
    mad = np.median(np.abs(diff - med), axis=1)  # [S]
    sigma = (mad / 0.6745).astype(np.float32)
    sigma = np.maximum(sigma, float(eps)).astype(np.float32)
    return sigma


# =============================================================================
# Losses
# =============================================================================

def weighted_mse(yhat: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE with per-sensor sigma.

    yhat, y: [B,S,K] or [B,S]
    sigma:   [S]
    """
    if sigma.dim() != 1:
        raise ValueError(f"sigma must be 1D [S], got shape {tuple(sigma.shape)}")

    if yhat.dim() == 3:
        # [B,S,K] / [1,S,1]
        sig = sigma.view(1, -1, 1)
    elif yhat.dim() == 2:
        # [B,S] / [1,S]
        sig = sigma.view(1, -1)
    else:
        raise ValueError(f"Unexpected yhat dim={yhat.dim()} (expected 2 or 3).")

    err = (yhat - y) / (sig + 1e-12)
    return (err * err).mean()



def multi_step_rollout_loss(
    model: MPRNN,
    x: torch.Tensor,                 # [B,S,K] inputs
    y: torch.Tensor,                 # [B,S,K] next-step targets
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    sigma: torch.Tensor,             # [S]
    rollout_H: int,
) -> torch.Tensor:
    """
    Roll out from x[:,:,0] for H steps using model predictions as inputs,
    compare to y[:,:,0:H].
    """
    B, S, K = x.shape
    H = int(min(rollout_H, K))
    if H <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    device = x.device
    h = torch.zeros((B, S, model.hidden_dim), device=device, dtype=x.dtype)
    c = torch.zeros((B, S, model.hidden_dim), device=device, dtype=x.dtype)

    obs = x[:, :, 0]
    preds = []
    for t in range(H):
        pred, h, c = model.forward_one_step(obs, h, c, edge_index, edge_attr)
        preds.append(pred)
        obs = pred  # always feed prediction
    yhat = torch.stack(preds, dim=-1)       # [B,S,H]
    ytrue = y[:, :, :H]                      # [B,S,H]
    return weighted_mse(yhat, ytrue, sigma)
