from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from model.imputation.imputeformer import FixedNodeImputeFormer


ATM_VECTOR_NAMES = ("U_x", "V_y")
PUSA_SOURCES = ("Pusa_IMD", "Pusa_DPCC")
PUSA_OUTPUT = "Pusa_averaged"
DIRECTION_CONVENTION = "meteorological_from_degrees; transport vector points toward pollutant motion"
DIRECTION_FORMULA = "U_x=-WS*sin(WD*pi/180); V_y=-WS*cos(WD*pi/180)"


@dataclass
class WindData:
    station_ids: np.ndarray
    timestamps: pd.DatetimeIndex
    sensors_xy: np.ndarray
    location_names: np.ndarray
    raw_WD: np.ndarray
    raw_WS: np.ndarray
    raw_pm25: np.ndarray
    raw_WD_mask: np.ndarray
    raw_WS_mask: np.ndarray
    raw_pm25_mask: np.ndarray
    observed_vectors: np.ndarray
    vector_mask: np.ndarray
    merged_sensors: dict[str, list[str]]
    source_columns: tuple[str, ...]
    source_data_csv: str
    source_locations_csv: str


@dataclass
class ImputeFormerWindConfig:
    windows: int = 128
    window_stride: int = 64
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mask_rate: float = 0.25
    input_embedding_dim: int = 32
    learnable_embedding_dim: int = 96
    num_layers: int = 3
    num_temporal_heads: int = 4
    dim_proj: int = 8
    dropout: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 123
    train_frac: float = 0.8
    val_frac: float = 0.1


def _match_monitor_ids(ids: pd.Index, wanted: tuple[str, ...]) -> list[str]:
    by_lower = {str(mid).lower(): str(mid) for mid in ids}
    return [by_lower[name.lower()] for name in wanted if name.lower() in by_lower]


def wind_to_transport_uv(wd_degrees: np.ndarray, ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wd_rad = np.deg2rad(wd_degrees.astype(np.float64))
    speed = ws.astype(np.float64)
    ux = -speed * np.sin(wd_rad)
    vy = -speed * np.cos(wd_rad)
    return ux.astype(np.float32), vy.astype(np.float32)


def transport_uv_to_wind(ux: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(ux.astype(np.float64) ** 2 + vy.astype(np.float64) ** 2)
    wd = np.degrees(np.arctan2(-ux.astype(np.float64), -vy.astype(np.float64)))
    wd = np.mod(wd, 360.0)
    return wd.astype(np.float32), speed.astype(np.float32)


def _series_for_monitor(
    readings: pd.DataFrame,
    monitor_ids: list[str],
    timestamps: pd.DatetimeIndex,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    subset = readings.loc[readings["monitor_id"].isin(monitor_ids), ["timestamp_round", *columns]]
    by_time = subset.groupby("timestamp_round", sort=True)[list(columns)].mean()
    values = by_time.reindex(timestamps).to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    return values.astype(np.float32), mask


def load_new_delhi_wind_data(
    data_csv: str | Path = "sim/govdata_1H_current.csv",
    locations_csv: str | Path = "sim/govdata_locations.csv",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> WindData:
    data_csv = Path(data_csv)
    locations_csv = Path(locations_csv)
    locs = pd.read_csv(locations_csv)
    readings = pd.read_csv(data_csv, parse_dates=["timestamp_round"])

    required_locs = {"Monitor ID", "Latitude", "Longitude", "Location"}
    missing_locs = required_locs.difference(locs.columns)
    if missing_locs:
        raise ValueError(f"{locations_csv} is missing columns: {sorted(missing_locs)}")
    required_readings = {"monitor_id", "timestamp_round", "WD", "WS", "pm25"}
    missing_readings = required_readings.difference(readings.columns)
    if missing_readings:
        raise ValueError(f"{data_csv} is missing columns: {sorted(missing_readings)}")

    locs = locs.drop_duplicates("Monitor ID").set_index("Monitor ID", drop=False)
    readings = readings[readings["monitor_id"].isin(locs.index)].copy()
    for col in ("WD", "WS", "pm25"):
        readings[col] = pd.to_numeric(readings[col], errors="coerce")

    timestamps = pd.DatetimeIndex(sorted(readings["timestamp_round"].dropna().unique()))
    if start is not None:
        timestamps = timestamps[timestamps >= pd.Timestamp(start)]
    if end is not None:
        timestamps = timestamps[timestamps <= pd.Timestamp(end)]
    if timestamps.empty:
        raise ValueError("No timestamps remain after applying start/end filters.")

    pusa_ids = _match_monitor_ids(locs.index, PUSA_SOURCES)
    pusa_set = set(pusa_ids)
    output_ids = [str(mid) for mid in locs.index if str(mid) not in pusa_set]
    merged_sensors: dict[str, list[str]] = {}
    if pusa_ids:
        output_ids.append(PUSA_OUTPUT)
        merged_sensors[PUSA_OUTPUT] = pusa_ids

    station_ids = np.asarray(output_ids, dtype="<U64")
    sensors_xy = np.empty((len(output_ids), 2), dtype=np.float32)
    location_names: list[str] = []
    raw_WD = np.empty((len(output_ids), len(timestamps)), dtype=np.float32)
    raw_WS = np.empty_like(raw_WD)
    raw_pm25 = np.empty_like(raw_WD)
    raw_WD_mask = np.empty_like(raw_WD, dtype=bool)
    raw_WS_mask = np.empty_like(raw_WD, dtype=bool)
    raw_pm25_mask = np.empty_like(raw_WD, dtype=bool)

    for i, monitor_id in enumerate(output_ids):
        if monitor_id == PUSA_OUTPUT:
            src_locs = locs.loc[pusa_ids]
            sensors_xy[i, 0] = float(src_locs["Longitude"].mean())
            sensors_xy[i, 1] = float(src_locs["Latitude"].mean())
            location_names.append("Averaged Pusa sensor from IMD and DPCC monitors")
            vals, masks = _series_for_monitor(readings, pusa_ids, timestamps, ("WD", "WS", "pm25"))
        else:
            row = locs.loc[monitor_id]
            sensors_xy[i, 0] = float(row["Longitude"])
            sensors_xy[i, 1] = float(row["Latitude"])
            location_names.append(str(row["Location"]))
            vals, masks = _series_for_monitor(readings, [monitor_id], timestamps, ("WD", "WS", "pm25"))
        raw_WD[i], raw_WS[i], raw_pm25[i] = vals[:, 0], vals[:, 1], vals[:, 2]
        raw_WD_mask[i], raw_WS_mask[i], raw_pm25_mask[i] = masks[:, 0], masks[:, 1], masks[:, 2]

    ux, vy = wind_to_transport_uv(raw_WD, raw_WS)
    vector_mask_2d = raw_WD_mask & raw_WS_mask & np.isfinite(ux) & np.isfinite(vy)
    observed_vectors = np.stack([np.nan_to_num(ux, nan=0.0), np.nan_to_num(vy, nan=0.0)], axis=-1).astype(np.float32)
    vector_mask = np.repeat(vector_mask_2d[..., None], 2, axis=-1)

    return WindData(
        station_ids=station_ids,
        timestamps=timestamps,
        sensors_xy=sensors_xy,
        location_names=np.asarray(location_names, dtype="<U256"),
        raw_WD=raw_WD,
        raw_WS=raw_WS,
        raw_pm25=raw_pm25,
        raw_WD_mask=raw_WD_mask,
        raw_WS_mask=raw_WS_mask,
        raw_pm25_mask=raw_pm25_mask,
        observed_vectors=observed_vectors,
        vector_mask=vector_mask,
        merged_sensors=merged_sensors,
        source_columns=("WD", "WS", "pm25"),
        source_data_csv=str(data_csv),
        source_locations_csv=str(locations_csv),
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalization(values: np.ndarray, masks: np.ndarray, train_time_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_channel_mask = masks[:, train_time_mask, :]
    train_values = values[:, train_time_mask, :]
    denom = train_channel_mask.sum(axis=(0, 1))
    mean = np.divide(
        (train_values * train_channel_mask).sum(axis=(0, 1)),
        denom,
        out=np.zeros(values.shape[-1], dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)
    std = np.ones(values.shape[-1], dtype=np.float32)
    for c in range(values.shape[-1]):
        valid = train_channel_mask[..., c].astype(bool)
        std[c] = float(train_values[..., c][valid].std() + 1e-6) if valid.any() else 1.0
    return mean, std


def _make_model(num_nodes: int, windows: int, out_dim: int, cfg: ImputeFormerWindConfig, device: torch.device) -> FixedNodeImputeFormer:
    return FixedNodeImputeFormer(
        num_nodes=num_nodes,
        windows=windows,
        input_dim=2 * out_dim,
        output_dim=out_dim,
        input_embedding_dim=cfg.input_embedding_dim,
        learnable_embedding_dim=cfg.learnable_embedding_dim,
        num_layers=cfg.num_layers,
        num_temporal_heads=cfg.num_temporal_heads,
        dim_proj=cfg.dim_proj,
        dropout=cfg.dropout,
    ).to(device)


def train_wind_imputeformer(
    wind: WindData,
    checkpoint_path: str | Path,
    cfg: ImputeFormerWindConfig,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    _set_seed(cfg.seed)
    device_t = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
    values = wind.observed_vectors.astype(np.float32)
    masks = wind.vector_mask.astype(np.float32)
    n_sensors, n_times, out_dim = values.shape
    windows = min(int(cfg.windows), n_times)
    if windows < 2:
        raise ValueError("Need at least two timesteps to train ImputeFormer.")

    n_train = max(windows, int(round(cfg.train_frac * n_times)))
    n_train = min(n_train, n_times)
    n_val = max(1, int(round(cfg.val_frac * n_times)))
    val_start = max(0, n_train)
    val_end = min(n_times, val_start + n_val)
    if val_end - val_start < 1:
        val_start = max(0, n_times - n_val)
        val_end = n_times
    train_time_mask = np.zeros(n_times, dtype=bool)
    val_time_mask = np.zeros(n_times, dtype=bool)
    train_time_mask[:n_train] = True
    val_time_mask[val_start:val_end] = True

    mean, std = _normalization(values, masks, train_time_mask)
    norm_values = ((values - mean) / std).astype(np.float32)
    starts = np.arange(0, max(1, n_times - windows + 1), max(1, cfg.window_stride), dtype=np.int64)

    model = _make_model(n_sensors, windows, out_dim, cfg, device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = None
    bad = 0

    def batch_window(starts_np: np.ndarray, target_time_mask: np.ndarray, context_time_mask: np.ndarray):
        vals, ctx, tgt = [], [], []
        for st in starts_np:
            sl = slice(int(st), int(st) + windows)
            vals.append(norm_values[:, sl])
            ctx.append((context_time_mask[None, sl, None] * masks[:, sl]).astype(np.float32))
            tgt.append((target_time_mask[None, sl, None] * masks[:, sl]).astype(np.float32))
        return (
            torch.from_numpy(np.stack(vals)).float().to(device_t),
            torch.from_numpy(np.stack(ctx)).float().to(device_t),
            torch.from_numpy(np.stack(tgt)).float().to(device_t),
        )

    if cfg.epochs <= 0:
        best_state = model.state_dict()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng = np.random.default_rng(cfg.seed + epoch)
        starts_epoch = starts.copy()
        rng.shuffle(starts_epoch)
        total = 0.0
        batches = 0
        for i in range(0, len(starts_epoch), cfg.batch_size):
            bstarts = starts_epoch[i : i + cfg.batch_size]
            target_time_mask = train_time_mask.copy()
            drop = rng.random((n_sensors, n_times, out_dim)) < cfg.mask_rate
            target_mask = target_time_mask[None, :, None] & drop & masks.astype(bool)
            context_mask = (train_time_mask[None, :, None] & masks.astype(bool)) & ~target_mask
            vals, ctx, tgt = [], [], []
            for st in bstarts:
                sl = slice(int(st), int(st) + windows)
                vals.append(norm_values[:, sl])
                ctx.append(context_mask[:, sl].astype(np.float32))
                tgt.append(target_mask[:, sl].astype(np.float32))
            vals_t = torch.from_numpy(np.stack(vals)).float().to(device_t)
            ctx_t = torch.from_numpy(np.stack(ctx)).float().to(device_t)
            tgt_t = torch.from_numpy(np.stack(tgt)).float().to(device_t)
            if tgt_t.sum() <= 0:
                continue
            pred = model(vals_t, ctx_t)
            loss = (((pred - vals_t) ** 2) * tgt_t).sum() / tgt_t.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item())
            batches += 1

        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(starts), cfg.val_batch_size):
                vals_t, ctx_t, tgt_t = batch_window(starts[i : i + cfg.val_batch_size], val_time_mask, train_time_mask)
                pred = model(vals_t, ctx_t)
                std_t = torch.from_numpy(std).float().to(device_t)
                se += ((((pred - vals_t) * std_t) ** 2) * tgt_t).sum().item()
                n += int(tgt_t.sum().item())
        rmse = math.sqrt(se / max(1, n))
        print(f"[epoch {epoch:03d}] train_mse={total / max(1, batches):.4e} val_rmse={rmse:.6f}")
        if rmse < best:
            best = rmse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "meta": {
                "variant": "imputeformer_wind",
                "val_mean": mean.tolist(),
                "val_std": std.tolist(),
                "normalizes_values": True,
                "channel_names": list(ATM_VECTOR_NAMES),
                "best_val_rmse": best,
                "windows": windows,
                "num_nodes": n_sensors,
            },
        },
        checkpoint_path,
    )
    return {"best_val_rmse": best, "checkpoint_path": str(checkpoint_path), "mean": mean, "std": std, "windows": windows}


def impute_wind_with_checkpoint(
    wind: WindData,
    checkpoint_path: str | Path,
    cfg: Optional[ImputeFormerWindConfig] = None,
    device: str | torch.device = "cuda",
) -> dict[str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ImputeFormerWindConfig(**checkpoint.get("config", {})) if cfg is None else cfg
    meta = checkpoint.get("meta", {})
    mean = np.asarray(meta.get("val_mean", [0.0, 0.0]), dtype=np.float32)
    std = np.asarray(meta.get("val_std", [1.0, 1.0]), dtype=np.float32)
    values = wind.observed_vectors.astype(np.float32)
    masks = wind.vector_mask.astype(np.float32)
    n_sensors, n_times, out_dim = values.shape
    windows = int(meta.get("windows", min(ckpt_cfg.windows, n_times)))
    windows = min(windows, n_times)

    device_t = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
    model = _make_model(n_sensors, windows, out_dim, ckpt_cfg, device_t)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_values = ((values - mean) / std).astype(np.float32)
    starts = list(range(0, max(1, n_times - windows + 1), windows))
    if starts[-1] != n_times - windows:
        starts.append(max(0, n_times - windows))
    pred_sum = np.zeros_like(values, dtype=np.float64)
    pred_count = np.zeros_like(values, dtype=np.float64)
    with torch.no_grad():
        for st in starts:
            sl = slice(int(st), int(st) + windows)
            vals_t = torch.from_numpy(norm_values[:, sl][None]).float().to(device_t)
            ctx_t = torch.from_numpy(masks[:, sl][None]).float().to(device_t)
            pred = model(vals_t, ctx_t)[0].detach().cpu().numpy()
            pred = pred * std + mean
            pred_sum[:, sl] += pred
            pred_count[:, sl] += 1.0
    pred_values = (pred_sum / np.maximum(pred_count, 1.0)).astype(np.float32)
    imputed_vectors = np.where(masks.astype(bool), values, pred_values).astype(np.float32)
    imputed_WD, imputed_WS = transport_uv_to_wind(imputed_vectors[..., 0], imputed_vectors[..., 1])
    city = imputed_vectors.mean(axis=0)
    return {
        "imputed_U_x": imputed_vectors[..., 0],
        "imputed_V_y": imputed_vectors[..., 1],
        "imputed_WD": imputed_WD,
        "imputed_WS": imputed_WS,
        "Vx": city[:, 0].astype(np.float32),
        "Vy": city[:, 1].astype(np.float32),
    }


def save_imputed_wind_product(
    wind: WindData,
    imputed: dict[str, np.ndarray],
    output_path: str | Path,
    imputation_config: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        timestamps=np.asarray([ts.isoformat() for ts in wind.timestamps], dtype="<U32"),
        station_ids=wind.station_ids,
        sensors_xy=wind.sensors_xy,
        location_names=wind.location_names,
        raw_WD=wind.raw_WD,
        raw_WS=wind.raw_WS,
        raw_pm25=wind.raw_pm25,
        raw_WD_mask=wind.raw_WD_mask,
        raw_WS_mask=wind.raw_WS_mask,
        raw_pm25_mask=wind.raw_pm25_mask,
        imputed_U_x=imputed["imputed_U_x"],
        imputed_V_y=imputed["imputed_V_y"],
        imputed_WD=imputed["imputed_WD"],
        imputed_WS=imputed["imputed_WS"],
        Vx=imputed["Vx"],
        Vy=imputed["Vy"],
        imputation_config=np.asarray([json.dumps(imputation_config, sort_keys=True)], dtype="<U4096"),
        direction_conversion_convention=np.asarray([DIRECTION_CONVENTION], dtype="<U128"),
        wind_vector_formula=np.asarray([DIRECTION_FORMULA], dtype="<U128"),
        source_columns=np.asarray(wind.source_columns, dtype="<U16"),
        source_data_csv=np.asarray([wind.source_data_csv], dtype="<U256"),
        source_locations_csv=np.asarray([wind.source_locations_csv], dtype="<U256"),
        merged_sensor_names=np.asarray(list(wind.merged_sensors), dtype="<U64"),
        merged_sensor_sources=np.asarray([json.dumps(v) for v in wind.merged_sensors.values()], dtype="<U256"),
    )


def assert_imputed_product_complete(path: str | Path) -> None:
    data = np.load(path, allow_pickle=True)
    required = ("imputed_U_x", "imputed_V_y", "imputed_WD", "imputed_WS", "Vx", "Vy")
    for key in required:
        arr = data[key]
        if not np.isfinite(arr).all():
            raise ValueError(f"{path} has non-finite values in {key}")

