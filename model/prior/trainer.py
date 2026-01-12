#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mprnn_heat.py

Train an MPRNN (message-passing RNN) dynamics prior on the synthetic periodic heat
sensor dataset saved by data/heatdata.py.

Specs implemented:
- Data-only training on *noisy* sensor series (no physics-equation features).
- Geometry graph over sensors (default: kNN; optional fully connected).
- Edge features include Δx, Δy, r, 1/r (+ optional RBF encoding of r).
- Per-sensor normalization using *train split* stats.
- Data-driven heteroscedastic weighting: sigma estimated from the sensor series
  (do NOT use saved generation noise_std).
- Truncated BPTT via sliding windows (length K+1) with *event-biased* window start sampling.
- Scheduled sampling (teacher forcing -> mix in predictions).
- Multi-step rollout loss added to objective.
- Early stopping on validation loss.
- Save best model to: /scratch/ab9738/stamp/checkpoints/
- Log metrics to: /scratch/ab9738/stamp/logs/mprnn_heat.log
- No argparse; edit CONFIG below if needed.

Dataset keys expected (FieldFormer-style, from heatdata.py):
  sensors_xy, sensor_noisy (and optionally sensor_clean, etc.)
"""

from __future__ import annotations

import os
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append(os.path.abspath("../../"))

from model.utils.seed import set_seed
from model.utils.io import load_npz
from model.utils.logging import setup_logger

from model.prior.normalize import time_split_series, compute_normalization_stats, apply_normalization
from model.prior.graph import build_graph_from_xy, build_edge_features
from model.prior.windows import compute_activity_scores, sample_window_starts_event_biased, make_windows_from_starts
from model.prior.losses import estimate_sigma_from_series, weighted_mse, multi_step_rollout_loss
from model.prior.mprnn import MPRNN
from model.prior.schedule import scheduled_sampling_prob


# =============================================================================
# Config
# =============================================================================

SRC_DIR = "/scratch/ab9738/stamp"
DEFAULT_DATA_PATH = os.path.join(SRC_DIR, "data", "heat_periodic_dataset.npz")

CHECKPOINTS_DIR = os.path.join(SRC_DIR, "checkpoints")
LOGS_DIR = os.path.join(SRC_DIR, "logs")
LOG_PATH = os.path.join(LOGS_DIR, "mprnn_heat.log")
CKPT_BEST_PATH = os.path.join(CHECKPOINTS_DIR, "mprnn_heat_best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 0

# Train/val split (time-contiguous)
TRAIN_FRAC = 0.8
SPLIT_GAP = 100  # gap steps between train and val to reduce leakage

# Windows (truncated BPTT)
WINDOW_K = 200                 # window length K (we use K+1 samples internally)
NUM_TRAIN_WINDOWS_PER_EPOCH = 4096
NUM_VAL_WINDOWS = 2048
BATCH_SIZE = 32

# Event-biased sampling
EVENT_BIAS_GAMMA = 1.5
EVENT_BIAS_EPS = 1e-6
UNIFORM_MIX_ETA = 0.10  # 10% uniform mixing

# Graph
GRAPH_MODE = "knn"  # "knn" or "fully_connected"
KNN_K = 6
EDGE_EPS = 1e-3
USE_RBF = True
RBF_NUM = 8
RBF_RMAX = 1.0

# MPRNN architecture
HIDDEN_DIM = 128
MSG_DIM = 128
EDGE_MLP_DIM = 64
MP_ROUNDS = 1        # message passing rounds per time step
DROPOUT = 0.0

# Scheduled sampling
SS_START_P = 0.0
SS_END_P = 0.7
SS_WARMUP_EPOCHS = 50

# Multi-step rollout loss
ROLLOUT_H = 20
ROLLOUT_LAMBDA = 0.2

# Optimization
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 200
PATIENCE = 15

# Misc
NORM_EPS = 1e-6
SIGMA_EPS = 1e-6

# =============================================================================
# Training / evaluation
# =============================================================================

@torch.no_grad()
def eval_on_fixed_windows(
    model: MPRNN,
    series: np.ndarray,                     # normalized val series [S,T]
    starts: np.ndarray,                     # [N]
    window_K: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    sigma: torch.Tensor,
    device: torch.device,
    scheduled_sampling_p: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    X_np, Y_np = make_windows_from_starts(series, starts, window_K=window_K)
    X = torch.from_numpy(X_np).to(device=device, dtype=DTYPE)
    Y = torch.from_numpy(Y_np).to(device=device, dtype=DTYPE)

    losses = []
    for b0 in range(0, X.shape[0], BATCH_SIZE):
        xb = X[b0 : b0 + BATCH_SIZE]
        yb = Y[b0 : b0 + BATCH_SIZE]
        yhat = model(xb, edge_index=edge_index, edge_attr=edge_attr, scheduled_sampling_p=scheduled_sampling_p)
        loss1 = weighted_mse(yhat, yb, sigma)
        lossH = multi_step_rollout_loss(model, xb, yb, edge_index, edge_attr, sigma, rollout_H=ROLLOUT_H)
        loss = loss1 + ROLLOUT_LAMBDA * lossH
        losses.append(float(loss.item()))

    return {"val_loss": float(np.mean(losses))}


def train_epoch(
    model: MPRNN,
    optimizer: torch.optim.Optimizer,
    train_series: np.ndarray,               # normalized train series [S,T]
    activity_scores: np.ndarray,            # [num_starts]
    window_K: int,
    num_windows: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    sigma: torch.Tensor,
    device: torch.device,
    epoch: int,
    rng: np.random.Generator,
    scheduled_sampling_p: float,
) -> Dict[str, float]:
    model.train()

    starts = sample_window_starts_event_biased(
        activity_scores=activity_scores,
        num_windows=num_windows,
        gamma=EVENT_BIAS_GAMMA,
        eps=EVENT_BIAS_EPS,
        uniform_mix_eta=UNIFORM_MIX_ETA,
        rng=rng,
    )

    X_np, Y_np = make_windows_from_starts(train_series, starts, window_K=window_K)
    X = torch.from_numpy(X_np).to(device=device, dtype=DTYPE)
    Y = torch.from_numpy(Y_np).to(device=device, dtype=DTYPE)

    # Shuffle windows
    perm = torch.randperm(X.shape[0], device=device)
    X = X[perm]
    Y = Y[perm]

    losses = []
    loss1s = []
    lossHs = []
    t0 = time.time()

    for b0 in range(0, X.shape[0], BATCH_SIZE):
        xb = X[b0 : b0 + BATCH_SIZE]
        yb = Y[b0 : b0 + BATCH_SIZE]

        yhat = model(xb, edge_index=edge_index, edge_attr=edge_attr, scheduled_sampling_p=scheduled_sampling_p)
        loss1 = weighted_mse(yhat, yb, sigma)
        lossH = multi_step_rollout_loss(model, xb, yb, edge_index, edge_attr, sigma, rollout_H=ROLLOUT_H)
        loss = loss1 + ROLLOUT_LAMBDA * lossH

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        losses.append(float(loss.item()))
        loss1s.append(float(loss1.item()))
        lossHs.append(float(lossH.item()))

    dt = time.time() - t0
    return {
        "train_loss": float(np.mean(losses)),
        "train_loss_1step": float(np.mean(loss1s)),
        "train_loss_rollout": float(np.mean(lossHs)),
        "train_sec": float(dt),
    }


def maybe_save_best(model: MPRNN, val_loss: float, best_val: float, ckpt_path: str) -> float:
    if val_loss < best_val:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "msg_dim": MSG_DIM,
                    "edge_mlp_dim": EDGE_MLP_DIM,
                    "mp_rounds": MP_ROUNDS,
                    "graph_mode": GRAPH_MODE,
                    "knn_k": KNN_K,
                    "use_rbf": USE_RBF,
                    "rbf_num": RBF_NUM,
                },
            },
            ckpt_path,
        )
        return val_loss
    return best_val


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = setup_logger(LOG_PATH)
    set_seed(SEED)

    logger.info(f"SRC_DIR={SRC_DIR}")
    logger.info(f"DATA={DEFAULT_DATA_PATH}")
    logger.info(f"DEVICE={DEVICE}")
    logger.info(f"LOG={LOG_PATH}")
    logger.info(f"CKPT_BEST={CKPT_BEST_PATH}")

    # Load dataset
    if not os.path.exists(DEFAULT_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DEFAULT_DATA_PATH}")

    data = load_npz(DEFAULT_DATA_PATH)

    if "sensor_noisy" not in data or "sensors_xy" not in data:
        raise KeyError(f"Expected keys 'sensor_noisy' and 'sensors_xy' in {DEFAULT_DATA_PATH}. Found: {list(data.keys())}")

    sensor_noisy = data["sensor_noisy"].astype(np.float32)  # [S,T]
    sensors_xy = data["sensors_xy"].astype(np.float32)      # [S,2]
    S, T = sensor_noisy.shape

    logger.info(f"Loaded sensor_noisy shape={sensor_noisy.shape}, sensors_xy shape={sensors_xy.shape}")

    # Train/val split (time-contiguous)
    train_raw, val_raw = time_split_series(sensor_noisy, TRAIN_FRAC, SPLIT_GAP)
    logger.info(f"Split: train T={train_raw.shape[1]}, val T={val_raw.shape[1]} (gap={SPLIT_GAP})")

    # Normalization (train stats only)
    mu, sd = compute_normalization_stats(train_raw, eps=NORM_EPS)
    train = apply_normalization(train_raw, mu, sd)
    val = apply_normalization(val_raw, mu, sd)

    # Data-driven sigma estimate for heteroscedastic weights (on normalized train series)
    sigma_np = estimate_sigma_from_series(train, eps=SIGMA_EPS)  # [S]
    sigma = torch.from_numpy(sigma_np).to(device=torch.device(DEVICE), dtype=DTYPE)
    logger.info(f"Sigma stats (normalized units): min={sigma_np.min():.4g}, median={np.median(sigma_np):.4g}, max={sigma_np.max():.4g}")

    # Build graph + edge attributes
    edge_index_np = build_graph_from_xy(sensors_xy, mode=GRAPH_MODE, knn_k=KNN_K)
    edge_attr_np = build_edge_features(
        sensors_xy,
        edge_index_np,
        edge_eps=EDGE_EPS,
        use_rbf=USE_RBF,
        rbf_num=RBF_NUM,
        rbf_rmax=RBF_RMAX,
    )
    E = edge_index_np.shape[1]
    F = edge_attr_np.shape[1]
    logger.info(f"Graph: mode={GRAPH_MODE} S={S} E={E} edge_attr_dim={F}")

    edge_index = torch.from_numpy(edge_index_np).to(device=torch.device(DEVICE), dtype=torch.long)
    edge_attr = torch.from_numpy(edge_attr_np).to(device=torch.device(DEVICE), dtype=DTYPE)

    # Event activity scores (train and val)
    train_scores = compute_activity_scores(train, window_K=WINDOW_K)
    val_scores = compute_activity_scores(val, window_K=WINDOW_K)

    # Fixed validation window starts (event-biased but deterministic)
    rng_val = np.random.default_rng(SEED + 12345)
    val_starts = sample_window_starts_event_biased(
        activity_scores=val_scores,
        num_windows=NUM_VAL_WINDOWS,
        gamma=EVENT_BIAS_GAMMA,
        eps=EVENT_BIAS_EPS,
        uniform_mix_eta=UNIFORM_MIX_ETA,
        rng=rng_val,
    )

    # Build model
    model = MPRNN(
        num_sensors=S,
        edge_attr_dim=F,
        hidden_dim=HIDDEN_DIM,
        msg_dim=MSG_DIM,
        edge_mlp_dim=EDGE_MLP_DIM,
        mp_rounds=MP_ROUNDS,
        dropout=DROPOUT,
    ).to(device=torch.device(DEVICE), dtype=DTYPE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop with early stopping
    best_val = float("inf")
    bad = 0
    rng_train = np.random.default_rng(SEED + 999)

    logger.info(
        "Begin training | "
        f"K={WINDOW_K} train_windows/epoch={NUM_TRAIN_WINDOWS_PER_EPOCH} val_windows={NUM_VAL_WINDOWS} "
        f"batch={BATCH_SIZE} H={ROLLOUT_H} lambda={ROLLOUT_LAMBDA} "
        f"ss={SS_START_P}->{SS_END_P} warmup={SS_WARMUP_EPOCHS} "
        f"lr={LR} wd={WEIGHT_DECAY} clip={GRAD_CLIP}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        ss_p = scheduled_sampling_prob(epoch=epoch, ss_start_p=SS_START_P, ss_end_p=SS_END_P, warmup_epochs=SS_WARMUP_EPOCHS)

        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            train_series=train,
            activity_scores=train_scores,
            window_K=WINDOW_K,
            num_windows=NUM_TRAIN_WINDOWS_PER_EPOCH,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sigma=sigma,
            device=torch.device(DEVICE),
            epoch=epoch,
            rng=rng_train,
            scheduled_sampling_p=ss_p,
        )

        val_metrics = eval_on_fixed_windows(
            model=model,
            series=val,
            starts=val_starts,
            window_K=WINDOW_K,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sigma=sigma,
            device=torch.device(DEVICE),
            scheduled_sampling_p=0.0,  # eval with pure teacher forcing by default
        )

        val_loss = float(val_metrics["val_loss"])
        best_prev = best_val
        best_val = maybe_save_best(model, val_loss=val_loss, best_val=best_val, ckpt_path=CKPT_BEST_PATH)

        improved = best_val < best_prev
        if improved:
            bad = 0
        else:
            bad += 1

        logger.info(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['train_loss']:.6f} "
            f"(1step={train_metrics['train_loss_1step']:.6f}, roll={train_metrics['train_loss_rollout']:.6f}) "
            f"val_loss={val_loss:.6f} "
            f"ss_p={ss_p:.3f} "
            f"sec={train_metrics['train_sec']:.2f} "
            f"{'BEST' if improved else ''} "
            f"bad={bad}/{PATIENCE}"
        )

        if bad >= PATIENCE:
            logger.info(f"Early stopping triggered at epoch={epoch} (patience={PATIENCE}). Best val={best_val:.6f}")
            break

    logger.info("Training done.")
    logger.info(f"Best checkpoint: {CKPT_BEST_PATH}")

if __name__ == "__main__":
    main()