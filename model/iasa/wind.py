from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_IMPUTED_WIND_PRODUCT = Path("data/new_delhi_wind_imputed.npz")


@dataclass(frozen=True)
class WindSequence:
    timestamps: Any
    vx: np.ndarray
    vy: np.ndarray
    provider: str
    metadata: dict[str, Any]


def _validate_wind_sequence(timestamps: Any, vx: np.ndarray, vy: np.ndarray, provider: str, metadata: dict[str, Any]) -> WindSequence:
    vx_arr = np.asarray(vx, dtype=np.float32)
    vy_arr = np.asarray(vy, dtype=np.float32)
    if vx_arr.ndim != 1 or vy_arr.ndim != 1:
        raise ValueError("vx and vy must have shape [T]")
    if vx_arr.shape != vy_arr.shape:
        raise ValueError("vx and vy must have the same shape")
    if len(timestamps) != vx_arr.shape[0]:
        raise ValueError("timestamps length must match vx/vy length")
    if not np.isfinite(vx_arr).all() or not np.isfinite(vy_arr).all():
        raise ValueError("vx and vy must contain only finite values")
    return WindSequence(timestamps=timestamps, vx=vx_arr, vy=vy_arr, provider=provider, metadata=metadata)


def _regular_timestamps(length: int, start: str = "2018-05-01 00:00:00+05:30") -> pd.DatetimeIndex:
    if int(length) <= 0:
        raise ValueError("length must be positive")
    return pd.date_range(start=start, periods=int(length), freq="h")


def _slice_time_window(
    timestamps: pd.DatetimeIndex,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    start: str | None,
    end: str | None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    keep = np.ones(len(timestamps), dtype=bool)
    if start is not None:
        keep &= timestamps >= pd.Timestamp(start)
    if end is not None:
        keep &= timestamps <= pd.Timestamp(end)
    if not np.any(keep):
        raise ValueError("No imputed wind timestamps remain after applying start/end filters.")
    return timestamps[keep], vx[keep], vy[keep]


def _observed_new_delhi_wind_sequence(
    data_csv: str | Path,
    locations_csv: str | Path,
    *,
    start: str | None,
    end: str | None,
) -> WindSequence:
    from data.pol_weather import load_new_delhi_wind_data

    wind = load_new_delhi_wind_data(data_csv, locations_csv, start=start, end=end)
    vectors = wind.observed_vectors.astype(np.float32, copy=False)
    mask = wind.vector_mask[..., 0].astype(bool, copy=False)
    vx = np.zeros(len(wind.timestamps), dtype=np.float32)
    vy = np.zeros_like(vx)
    for t in range(len(wind.timestamps)):
        valid = mask[:, t]
        if np.any(valid):
            vx[t] = float(np.mean(vectors[valid, t, 0]))
            vy[t] = float(np.mean(vectors[valid, t, 1]))
    metadata: dict[str, Any] = {
        "aggregation": "observed_station_mean_city_level",
        "station_count": int(len(wind.station_ids)),
        "valid_vector_count": int(np.count_nonzero(mask)),
        "source_data_csv": str(data_csv),
        "source_locations_csv": str(locations_csv),
        "imputed_product_missing": True,
    }
    return _validate_wind_sequence(wind.timestamps, vx, vy, "real_observed_new_delhi", metadata)


def real_new_delhi_wind_sequence(
    data_csv: str | Path = "sim/govdata_1H_current.csv",
    locations_csv: str | Path = "sim/govdata_locations.csv",
    *,
    imputed_product_path: str | Path | None = DEFAULT_IMPUTED_WIND_PRODUCT,
    start: str | None = None,
    end: str | None = None,
    allow_observed_fallback: bool = False,
) -> WindSequence:
    from data.pol_weather import assert_imputed_product_complete

    product_path = Path(imputed_product_path) if imputed_product_path is not None else DEFAULT_IMPUTED_WIND_PRODUCT
    if not product_path.exists():
        if allow_observed_fallback:
            return _observed_new_delhi_wind_sequence(
                data_csv,
                locations_csv,
                start=start,
                end=end,
            )
        raise FileNotFoundError(
            f"Missing imputed wind product {product_path}. Run scripts/impute_new_delhi_wind.py "
            "or pass allow_observed_fallback=True for an explicitly observed-only sequence."
        )

    assert_imputed_product_complete(product_path)
    with np.load(product_path, allow_pickle=False) as data:
        timestamps = pd.to_datetime(data["timestamps"].astype(str))
        vx = np.asarray(data["Vx"], dtype=np.float32)
        vy = np.asarray(data["Vy"], dtype=np.float32)
        station_count = int(np.asarray(data["station_ids"]).shape[0])
        source_data_csv = str(np.asarray(data["source_data_csv"]).reshape(-1)[0])
        source_locations_csv = str(np.asarray(data["source_locations_csv"]).reshape(-1)[0])
    timestamps, vx, vy = _slice_time_window(timestamps, vx, vy, start=start, end=end)
    metadata: dict[str, Any] = {
        "aggregation": "imputed_product_city_level",
        "station_count": station_count,
        "imputed_product_path": str(product_path),
        "source_data_csv": source_data_csv,
        "source_locations_csv": source_locations_csv,
    }
    return _validate_wind_sequence(timestamps, vx, vy, "real_imputed_new_delhi", metadata)


def constant_direction(
    *,
    length: int,
    vx: float,
    vy: float,
    start: str = "2018-05-01 00:00:00+05:30",
) -> WindSequence:
    timestamps = _regular_timestamps(length, start=start)
    metadata = {"kind": "constant", "vx": float(vx), "vy": float(vy)}
    return _validate_wind_sequence(
        timestamps,
        np.full(length, float(vx), dtype=np.float32),
        np.full(length, float(vy), dtype=np.float32),
        "constant_direction",
        metadata,
    )


def single_direction_synthetic(
    *,
    length: int,
    speed: float = 1.0,
    direction_degrees: float = 270.0,
    start: str = "2018-05-01 00:00:00+05:30",
) -> WindSequence:
    rad = np.deg2rad(float(direction_degrees))
    vx = float(speed) * np.cos(rad)
    vy = float(speed) * np.sin(rad)
    seq = constant_direction(length=length, vx=vx, vy=vy, start=start)
    return WindSequence(
        timestamps=seq.timestamps,
        vx=seq.vx,
        vy=seq.vy,
        provider="single_direction_synthetic",
        metadata={"speed": float(speed), "direction_degrees_math": float(direction_degrees)},
    )


def diurnal_synthetic(
    *,
    length: int,
    base_vx: float = 1.0,
    base_vy: float = 0.0,
    amplitude: float = 0.35,
    start: str = "2018-05-01 00:00:00+05:30",
) -> WindSequence:
    timestamps = _regular_timestamps(length, start=start)
    phase = np.arange(length, dtype=np.float32) * (2.0 * np.pi / 24.0)
    vx = float(base_vx) + float(amplitude) * np.sin(phase)
    vy = float(base_vy) + float(amplitude) * np.cos(phase)
    metadata = {"base_vx": float(base_vx), "base_vy": float(base_vy), "amplitude": float(amplitude)}
    return _validate_wind_sequence(timestamps, vx, vy, "diurnal_synthetic", metadata)


def ar1_synthetic(
    *,
    length: int,
    base_vx: float = 1.0,
    base_vy: float = 0.0,
    rho: float = 0.9,
    sigma: float = 0.1,
    seed: int = 0,
    start: str = "2018-05-01 00:00:00+05:30",
) -> WindSequence:
    timestamps = _regular_timestamps(length, start=start)
    rng = np.random.default_rng(seed)
    noise = np.zeros((length, 2), dtype=np.float32)
    for t in range(1, length):
        noise[t] = float(rho) * noise[t - 1] + rng.normal(0.0, float(sigma), size=2).astype(np.float32)
    vx = float(base_vx) + noise[:, 0]
    vy = float(base_vy) + noise[:, 1]
    metadata = {
        "base_vx": float(base_vx),
        "base_vy": float(base_vy),
        "rho": float(rho),
        "sigma": float(sigma),
        "seed": int(seed),
    }
    return _validate_wind_sequence(timestamps, vx, vy, "ar1_synthetic", metadata)


def multi_direction_synthetic(
    *,
    length: int,
    speed: float = 1.0,
    directions_degrees: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
    seed: int = 0,
    start: str = "2018-05-01 00:00:00+05:30",
) -> WindSequence:
    timestamps = _regular_timestamps(length, start=start)
    rng = np.random.default_rng(seed)
    directions = np.asarray(directions_degrees, dtype=np.float32)
    chosen = directions[rng.integers(0, len(directions), size=length)]
    rad = np.deg2rad(chosen.astype(np.float64))
    vx = float(speed) * np.cos(rad)
    vy = float(speed) * np.sin(rad)
    metadata = {
        "speed": float(speed),
        "directions_degrees_math": [float(x) for x in directions_degrees],
        "seed": int(seed),
    }
    return _validate_wind_sequence(timestamps, vx, vy, "multi_direction_synthetic", metadata)


__all__ = [
    "DEFAULT_IMPUTED_WIND_PRODUCT",
    "WindSequence",
    "ar1_synthetic",
    "constant_direction",
    "diurnal_synthetic",
    "multi_direction_synthetic",
    "real_new_delhi_wind_sequence",
    "single_direction_synthetic",
]
