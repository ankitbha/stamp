#!/usr/bin/env python3
"""Train the coordinate-query FieldFormer to impute the 2-vector transport wind
field (Ux, Vy) for New Delhi (Task 9D).

Produces the ``out_dim=2`` checkpoint that activates
``FieldFormerCoordinateQueryImputer`` (Task 9A). Supervision is the sparse
observed station-time tuples ``(x, y, t) -> (Ux, Vy)`` from the government WD/WS
record (converted to transport vectors inside ``load_new_delhi_wind_data`` via
paper eq. wind_direction_conversion); there is no dense city-wide wind truth.

Conventions mirror the query-time adapter EXACTLY (or the learned attention is
queried off-distribution): min-max ``xy`` normalized to ``[0, 1]`` from the
station extent, time ``t = k/(T-1)``, s-major observation tuples ``lin = s*T + k``,
and matching ``k_neighbors`` / ``time_radius``. Training follows the upstream
``ffag_polsparse_train.py`` recipe (AdamW two-group with a separate ``log_gammas``
group, ReduceLROnPlateau, EMA weights, grad clipping, ``log_gammas`` frozen for the
first few epochs then clamped, early stopping) but DROPS the scalar-field-specific
sponge/radiation physics regularizers -- those are not assumed to transfer to a
2-vector wind field.

Checkpoints are persistent and resumable under ``checkpoints/`` (git-ignored):
each save holds ``model_state_dict``, ``ema_model_state_dict`` (preferred by the
loader), optimizer/scheduler state, epoch, best held-out RMSE, config, and full
provenance. The trained checkpoint is loadable by
``load_fieldformer_checkpoint(out_dim=2)`` / ``build_fieldformer_wind_imputer``.

The kernel imputer remains the IASA default; adopting the FieldFormer default is a
separate, validation-gated step (it must beat the kernel and city-mean baselines
on the identical held-out split, reported here at the end of training).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.fieldformer.model import (  # noqa: E402
    FieldFormerCoordinateQuery,
    SplitAwareSparseNeighborIndexer,
)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class WindTrainConfig:
    # Data.
    data_csv: str = "sim/govdata_1H_current.csv"
    locations_csv: str = "sim/govdata_locations.csv"
    start: str | None = None
    end: str | None = None
    grid_shape: tuple[int, int] = (40, 40)
    max_timesteps: int | None = None  # truncate T for smoke/fast runs
    holdout_frac: float = 0.2
    # Model (defaults match load_fieldformer_checkpoint so the checkpoint loads).
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    d_ff: int = 256
    k_neighbors: int = 32
    time_radius: int = 3
    # Optimization (upstream recipe, physics losses removed).
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    gamma_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    freeze_gamma_epochs: int = 6
    patience: int = 20
    min_delta: float = 1e-8
    seed: int = 0
    # IO.
    checkpoint_dir: str = "checkpoints"
    run_name: str = "fieldformer_wind_new_delhi"
    resume: bool = False
    smoke: bool = False
    device: str | None = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# EarlyStopping (upstream contract)                                           #
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    patience: int = 20
    min_delta: float = 1e-8
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, metric: float) -> None:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True


# --------------------------------------------------------------------------- #
# Supervision assembly                                                        #
# --------------------------------------------------------------------------- #
def _array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()[:16]


def stations_to_grid(sensors_lonlat: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Map station lon/lat to grid coords EXACTLY as gridded_new_delhi_wind_field."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    lon = np.asarray(sensors_lonlat[:, 0], dtype=np.float64)
    lat = np.asarray(sensors_lonlat[:, 1], dtype=np.float64)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-12)
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-12)
    ix = (lon - np.nanmin(lon)) / lon_scale * (nx - 1)
    iy = (lat - np.nanmin(lat)) / lat_scale * (ny - 1)
    return np.stack([ix, iy], axis=1).astype(np.float64)


def choose_holdout(station_mask: np.ndarray, holdout_frac: float, seed: int) -> list[int]:
    """Pick held-out stations only among those with >=1 observed timestep, and
    keep at least one observed station for training (else validation or training
    would have no tuples). ``station_mask`` is the [S, T] observation mask.
    """
    mask = np.asarray(station_mask, dtype=bool)
    observed = np.flatnonzero(mask.any(axis=1))
    if observed.size < 2:
        raise ValueError("need at least 2 observed stations to form a held-out split")
    n_hold = max(1, int(round(float(holdout_frac) * observed.size)))
    n_hold = min(n_hold, observed.size - 1)  # keep >=1 observed train station
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(observed)
    return sorted(int(i) for i in perm[:n_hold])


def build_supervision(
    station_grid_xy: np.ndarray,
    observed_vectors: np.ndarray,
    station_mask: np.ndarray,
    *,
    holdout_indices: list[int],
    k_neighbors: int,
    time_radius: int,
    device: str,
) -> dict[str, Any]:
    """Assemble normalized s-major tuples, the neighbor indexer, and train/val
    linear-index splits. Held-out (validation) tuples are excluded from
    ``allowed_indices`` so they are never read as neighbors.
    """
    stations = np.asarray(station_grid_xy, dtype=np.float64)
    vectors = np.asarray(observed_vectors, dtype=np.float32)
    mask = np.asarray(station_mask, dtype=bool)
    S, T = vectors.shape[0], vectors.shape[1]
    if stations.shape != (S, 2):
        raise ValueError("station_grid_xy must be [S, 2]")
    if vectors.shape != (S, T, 2):
        raise ValueError("observed_vectors must be [S, T, 2]")
    if mask.shape != (S, T):
        raise ValueError("station_mask must be [S, T]")

    # Min-max normalize xy to [0, 1] over the station extent (adapter convention).
    xmin, ymin = float(stations[:, 0].min()), float(stations[:, 1].min())
    xmax, ymax = float(stations[:, 0].max()), float(stations[:, 1].max())
    sx = max(xmax - xmin, 1e-9)
    sy = max(ymax - ymin, 1e-9)
    stations_norm = np.empty_like(stations)
    stations_norm[:, 0] = (stations[:, 0] - xmin) / sx
    stations_norm[:, 1] = (stations[:, 1] - ymin) / sy
    domain_bounds = (xmin, xmax, ymin, ymax)

    t_grid_np = np.arange(T, dtype=np.float64) / max(T - 1, 1)
    # s-major tuples: lin = s*T + k.
    sxr = np.repeat(stations_norm[:, 0], T)
    syr = np.repeat(stations_norm[:, 1], T)
    st = np.tile(t_grid_np, S)
    obs_coords_np = np.stack([sxr, syr, st], axis=1).astype(np.float32)
    obs_vals_np = vectors.reshape(S * T, 2).astype(np.float32)

    hold = set(int(i) for i in holdout_indices)
    train_lins: list[int] = []
    val_lins: list[int] = []
    for s in range(S):
        for k in range(T):
            if not mask[s, k]:
                continue
            lin = s * T + k
            (val_lins if s in hold else train_lins).append(lin)
    if not train_lins:
        raise ValueError("no observed training tuples")
    if not val_lins:
        raise ValueError("no observed held-out tuples for validation")

    obs_coords = torch.as_tensor(obs_coords_np, dtype=torch.float32, device=device)
    obs_vals = torch.as_tensor(obs_vals_np, dtype=torch.float32, device=device)
    sensors_xy = torch.as_tensor(stations_norm, dtype=torch.float32, device=device)
    t_grid = torch.as_tensor(t_grid_np, dtype=torch.float32, device=device)
    train_lin_t = torch.as_tensor(train_lins, dtype=torch.long, device=device)

    indexer = SplitAwareSparseNeighborIndexer(
        sensors_xy, t_grid, time_radius=int(time_radius), k_neighbors=int(k_neighbors),
        allowed_indices=train_lin_t,
    )
    return {
        "obs_coords": obs_coords,
        "obs_vals": obs_vals,
        "indexer": indexer,
        "train_lins": train_lin_t,
        "val_lins": torch.as_tensor(val_lins, dtype=torch.long, device=device),
        "S": S,
        "T": T,
        "domain_bounds": domain_bounds,
        "holdout_indices": sorted(hold),
        "n_train_tuples": len(train_lins),
        "n_val_tuples": len(val_lins),
    }


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def _ema_update(ema: torch.nn.Module, online: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), online.parameters()):
            ep.copy_(decay * ep + (1.0 - decay) * p)
        for eb, b in zip(ema.buffers(), online.buffers()):
            eb.copy_(b)


def _vector_rmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    d = (pred - tgt).to(torch.float64)
    return float(torch.sqrt(torch.mean(torch.sum(d * d, dim=1))))


def train_from_supervision(supervision: dict[str, Any], cfg: WindTrainConfig, *, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    device = cfg.resolved_device()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    obs_coords = supervision["obs_coords"]
    obs_vals = supervision["obs_vals"]
    indexer = supervision["indexer"]
    train_lins = supervision["train_lins"]
    val_lins = supervision["val_lins"]

    model = FieldFormerCoordinateQuery(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff, out_dim=2).to(device)
    # Initialize the relative-position scales (x, y, t) to the upstream defaults
    # log([1, 1, 0.5]) so the distance kernel starts sensible instead of uniform
    # (exp(0)=1 on every axis); the model still learns them after freeze_gamma_epochs.
    with torch.no_grad():
        model.log_gammas.copy_(torch.log(torch.tensor([1.0, 1.0, 0.5], device=device)))
    ema_model = FieldFormerCoordinateQuery(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff, out_dim=2).to(device)
    ema_model.load_state_dict(model.state_dict())

    base_params = [p for n, p in model.named_parameters() if n != "log_gammas"]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": [model.log_gammas], "lr": cfg.gamma_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{cfg.run_name}.pt"
    last_path = ckpt_dir / f"{cfg.run_name}.last.pt"

    start_epoch = 1
    best_rmse = float("inf")
    if cfg.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_rmse = float(ckpt.get("best_val_rmse", float("inf")))
        stopper.best = best_rmse
        # Restore accumulated non-improving epochs so early stopping is not reset
        # by a resume (otherwise a resumed run can overshoot by up to `patience`).
        stopper.bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed))

    def predict(q_lin: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
        nb_idx = indexer.gather_observed_neighbors(q_lin, exclude_self=True)
        return net.forward_observed(q_lin, obs_coords, obs_vals, nb_idx)

    @torch.no_grad()
    def val_rmse() -> float:
        ema_model.eval()
        preds, tgts = [], []
        n = val_lins.shape[0]
        for i in range(0, n, cfg.batch_size):
            q = val_lins[i : i + cfg.batch_size]
            preds.append(predict(q, ema_model))
            tgts.append(obs_vals[q])
        return _vector_rmse(torch.cat(preds, 0), torch.cat(tgts, 0))

    history: list[dict[str, float]] = []
    n_train = train_lins.shape[0]
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        model.log_gammas.requires_grad_(epoch > cfg.freeze_gamma_epochs)
        perm = train_lins[torch.randperm(n_train, generator=gen).to(train_lins.device)]
        running = 0.0
        n_batches = 0
        for i in range(0, n_train, cfg.batch_size):
            q = perm[i : i + cfg.batch_size]
            pred = predict(q, model)
            loss = F.mse_loss(pred, obs_vals[q])
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            with torch.no_grad():
                model.log_gammas.clamp_(-2.0, 2.0)
            _ema_update(ema_model, model, cfg.ema_decay)
            running += float(loss.item())
            n_batches += 1

        rmse = val_rmse()
        scheduler.step(rmse)
        lr0 = optimizer.param_groups[0]["lr"]
        train_loss = running / max(1, n_batches)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_vector_rmse": rmse, "lr": lr0})
        # Progress goes to stderr so stdout stays a clean JSON channel for callers
        # (the gate / CLI summary); the merged SLURM log still shows every epoch.
        print(f"[epoch {epoch:03d}/{cfg.epochs}] train_mse={train_loss:.6e} val_vector_rmse={rmse:.6e} lr={lr0:.2e}", file=sys.stderr, flush=True)

        completed_epoch = epoch
        improved = rmse < best_rmse - cfg.min_delta
        # Step the stopper first so the persisted bad_epochs reflects this epoch,
        # making a later --resume pick up early stopping exactly where it left off.
        stopper.step(rmse)
        prov = dict(provenance or {})
        prov.update({"epoch": epoch, "device": device})
        payload = {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_rmse": min(best_rmse, rmse),
            "early_stop_bad_epochs": int(stopper.bad_epochs),
            "config": asdict(cfg),
            "provenance": prov,
        }
        torch.save(payload, last_path)
        if improved:
            best_rmse = rmse
            payload["best_val_rmse"] = best_rmse
            torch.save(payload, best_path)

        if stopper.stopped:
            print(f"[early-stop] patience={cfg.patience} reached at epoch {epoch}.", file=sys.stderr, flush=True)
            break

    return {
        "best_val_rmse": best_rmse,
        "epochs_run": completed_epoch,
        "early_stopped": stopper.stopped,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
        "holdout_indices": supervision["holdout_indices"],
        "n_train_tuples": supervision["n_train_tuples"],
        "n_val_tuples": supervision["n_val_tuples"],
        "domain_bounds": supervision["domain_bounds"],
    }


# --------------------------------------------------------------------------- #
# Data loading + baseline comparison + full entrypoint                        #
# --------------------------------------------------------------------------- #
def load_new_delhi_supervision(cfg: WindTrainConfig) -> dict[str, Any]:
    from data.pol_weather import load_new_delhi_wind_data

    wind = load_new_delhi_wind_data(cfg.data_csv, cfg.locations_csv, start=cfg.start, end=cfg.end)
    vectors = np.asarray(wind.observed_vectors, dtype=np.float32)  # [S, T, 2]
    mask = np.asarray(wind.vector_mask[..., 0], dtype=bool)         # [S, T]
    if cfg.max_timesteps is not None:
        T = min(int(cfg.max_timesteps), vectors.shape[1])
        vectors = vectors[:, :T]
        mask = mask[:, :T]
        timestamps = wind.timestamps[:T]
    else:
        timestamps = wind.timestamps
    station_grid_xy = stations_to_grid(np.asarray(wind.sensors_xy, dtype=np.float64), cfg.grid_shape)
    holdout = choose_holdout(mask, cfg.holdout_frac, cfg.seed)
    return {
        "station_grid_xy": station_grid_xy,
        "observed_vectors": vectors,
        "station_mask": mask,
        "timestamps": timestamps,
        "holdout_indices": holdout,
        "station_ids": [str(s) for s in wind.station_ids],
        "data_hash": _array_hash(vectors),
        "mask_hash": _array_hash(mask.astype(np.uint8)),
    }


def _baseline_reports(data: dict[str, Any], cfg: WindTrainConfig, ff_imputer: Any) -> dict[str, Any]:
    """Held-out validation: FieldFormer vs kernel vs city-mean on the SAME split."""
    from model.iasa.wind import KernelCoordinateQueryImputer, evaluate_gridded_wind_heldout

    stations = data["station_grid_xy"]
    vectors = data["observed_vectors"]
    mask = data["station_mask"]
    timestamps = data["timestamps"]
    holdout = tuple(data["holdout_indices"])

    class _CityMeanImputer:
        name = "city_mean"

        def query(self, coords_xy, t_index, station_coords, station_vectors, station_mask):
            m = np.asarray(station_mask, dtype=bool)[:, int(t_index)]
            v = np.asarray(station_vectors, dtype=np.float32)[:, int(t_index), :]
            mean = v[m].mean(axis=0) if m.any() else np.zeros(2, dtype=np.float32)
            return np.tile(mean.astype(np.float32), (np.asarray(coords_xy).shape[0], 1))

    out: dict[str, Any] = {}
    for label, imp in (("fieldformer", ff_imputer), ("kernel", KernelCoordinateQueryImputer()), ("city_mean", _CityMeanImputer())):
        try:
            out[label] = evaluate_gridded_wind_heldout(
                stations, vectors, mask, timestamps, imputer=imp, holdout_station_indices=holdout
            )
        except Exception as exc:  # pragma: no cover - report, do not crash training
            out[label] = {"error": str(exc)}
    return out


def train_fieldformer_wind(cfg: WindTrainConfig) -> dict[str, Any]:
    if cfg.smoke:
        cfg.d_model = min(cfg.d_model, 16)
        cfg.nhead = 2
        cfg.layers = 1
        cfg.d_ff = min(cfg.d_ff, 32)
        cfg.epochs = min(cfg.epochs, 2)
        cfg.max_timesteps = cfg.max_timesteps or 24
        cfg.grid_shape = (12, 12)
        cfg.batch_size = 128

    data = load_new_delhi_supervision(cfg)
    supervision = build_supervision(
        data["station_grid_xy"], data["observed_vectors"], data["station_mask"],
        holdout_indices=data["holdout_indices"], k_neighbors=cfg.k_neighbors,
        time_radius=cfg.time_radius, device=cfg.resolved_device(),
    )
    provenance = {
        "task": "9D",
        "seed": cfg.seed,
        "out_dim": 2,
        "k_neighbors": cfg.k_neighbors,
        "time_radius": cfg.time_radius,
        "grid_shape": list(cfg.grid_shape),
        "normalization": "minmax_xy_[0,1]_from_station_extent; t=k/(T-1)",
        "domain_bounds_grid": list(supervision["domain_bounds"]),
        "ema_decay": cfg.ema_decay,
        "data_csv": cfg.data_csv,
        "locations_csv": cfg.locations_csv,
        "data_hash": data["data_hash"],
        "mask_hash": data["mask_hash"],
        "station_ids": data["station_ids"],
        "physics_regularizers": "disabled_for_wind (sponge/radiation not transferred)",
        "reproduce": "python3 scripts/train_fieldformer_wind.py "
        + " ".join(f"--{k} {v}" for k, v in (("epochs", cfg.epochs), ("seed", cfg.seed))),
    }
    report = train_from_supervision(supervision, cfg, provenance=provenance)

    # Held-out validation vs baselines (kernel default stays until FF wins).
    try:
        from model.iasa.fieldformer_adapter import build_fieldformer_wind_imputer

        ff_imputer = build_fieldformer_wind_imputer(
            checkpoint_path=report["best_checkpoint"], d_model=cfg.d_model, nhead=cfg.nhead,
            layers=cfg.layers, d_ff=cfg.d_ff, time_radius=cfg.time_radius, k_neighbors=cfg.k_neighbors,
            # Pin the training-time normalization extent instead of recomputing it
            # from the eval station set (identical here, but explicit is safer).
            domain_bounds=tuple(report["domain_bounds"]), device="cpu", use_ema=True,
        )
        report["heldout_validation"] = _baseline_reports(data, cfg, ff_imputer)
        ff = report["heldout_validation"].get("fieldformer", {})
        ker = report["heldout_validation"].get("kernel", {})
        cm = report["heldout_validation"].get("city_mean", {})
        if "vector_rmse" in ff and "vector_rmse" in ker:
            ff_r = float(ff["vector_rmse"])
            ker_r = float(ker["vector_rmse"])
            cm_r = float(cm.get("vector_rmse", float("inf")))
            best_baseline_r = min(ker_r, cm_r)
            report["fieldformer_beats_kernel"] = bool(ff_r < ker_r)
            # Roadmap Task 9D: adopt FieldFormer as the default ONLY if it improves
            # held-out error over BOTH the kernel AND the city-mean baseline. A
            # trivial spatially-constant city mean is a strong baseline for
            # city-scale wind, so beating the kernel alone is not sufficient.
            report["fieldformer_beats_all_baselines"] = bool(ff_r < best_baseline_r)
            report["strongest_baseline"] = "city_mean" if cm_r <= ker_r else "kernel"
            report["strongest_baseline_vector_rmse"] = best_baseline_r
            report["recommended_default"] = "fieldformer" if report["fieldformer_beats_all_baselines"] else "kernel"
    except Exception as exc:  # pragma: no cover
        report["heldout_validation"] = {"error": str(exc)}
        report["recommended_default"] = "kernel"

    report["provenance"] = provenance
    report_path = Path(cfg.checkpoint_dir) / f"{cfg.run_name}.report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    report["report_path"] = str(report_path)
    return report


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _parse_args(argv: list[str] | None = None) -> WindTrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-csv", default="sim/govdata_1H_current.csv")
    p.add_argument("--locations-csv", default="sim/govdata_locations.csv")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--grid", type=int, nargs=2, default=(40, 40))
    p.add_argument("--max-timesteps", type=int, default=None)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--k-neighbors", type=int, default=32)
    p.add_argument("--time-radius", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--run-name", default="fieldformer_wind_new_delhi")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    return WindTrainConfig(
        data_csv=a.data_csv, locations_csv=a.locations_csv, start=a.start, end=a.end,
        grid_shape=(a.grid[0], a.grid[1]), max_timesteps=a.max_timesteps, holdout_frac=a.holdout_frac,
        d_model=a.d_model, nhead=a.nhead, layers=a.layers, d_ff=a.d_ff,
        k_neighbors=a.k_neighbors, time_radius=a.time_radius, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, gamma_lr=a.gamma_lr, weight_decay=a.weight_decay, patience=a.patience, seed=a.seed,
        checkpoint_dir=a.checkpoint_dir, run_name=a.run_name, resume=a.resume, smoke=a.smoke, device=a.device,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    report = train_fieldformer_wind(cfg)
    summary = {k: report[k] for k in ("best_val_rmse", "epochs_run", "early_stopped", "best_checkpoint", "report_path", "recommended_default") if k in report}
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
