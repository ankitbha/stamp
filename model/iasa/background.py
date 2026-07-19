from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch

from model.iasa.backend import (
    dtype_name,
    resolve_device,
    resolve_dtype,
    runtime_provenance,
)


@dataclass(frozen=True)
class BackgroundBasisConfig:
    include_constant: bool = True
    temporal_polynomial_degree: int = 0
    daily_harmonics: int = 0
    day_intercepts: bool = False
    regional_coordinate_trends: bool = False
    sensor_offsets: bool = False
    max_background_rank: int = 8
    basis_mode: str = "normal"
    device: str = "cpu"
    dtype: str = "float64"


@dataclass(frozen=True)
class BackgroundBasisResult:
    Q: torch.Tensor
    column_names: list[str]
    row_index: list[dict[str, Any]]
    metadata: dict[str, Any]


def _validated_rows(row_index: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    rows = [dict(row) for row in row_index]
    if not rows:
        return rows, [], []
    required = {"time_index", "sensor_index", "sensor_id"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("row_index entries require time_index, sensor_index, and sensor_id")
    first_t = int(rows[0]["time_index"])
    first_block = [row for row in rows if int(row["time_index"]) == first_t]
    sensor_ids = [str(row["sensor_id"]) for row in first_block]
    if not sensor_ids or len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("row_index must contain a nonempty unique sensor ordering")
    expected_sensor_indices = list(range(len(sensor_ids)))
    times: list[int] = []
    cursor = 0
    while cursor < len(rows):
        t = int(rows[cursor]["time_index"])
        block = rows[cursor: cursor + len(sensor_ids)]
        if len(block) != len(sensor_ids) or any(int(row["time_index"]) != t for row in block):
            raise ValueError("row_index must contain complete time-major sensor blocks")
        if [int(row["sensor_index"]) for row in block] != expected_sensor_indices:
            raise ValueError("row_index sensor indices must be contiguous and stable at every time")
        if [str(row["sensor_id"]) for row in block] != sensor_ids:
            raise ValueError("row_index sensor IDs must have identical ordering at every time")
        times.append(t)
        cursor += len(sensor_ids)
    if times != list(range(times[0], times[0] + len(times))):
        raise ValueError("row_index time indices must be contiguous and increasing")
    return rows, times, sensor_ids


def _timestamp_hours(timestamps: Any, time_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not time_indices:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype="datetime64[D]"),
        )
    if len(timestamps) <= max(time_indices):
        raise ValueError("timestamps must cover every row_index time_index")
    try:
        selected = np.asarray([np.datetime64(timestamps[i]) for i in time_indices], dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise TypeError("timestamps must contain datetime-like values") from exc
    if np.isnat(selected).any():
        raise ValueError("timestamps must not contain NaT")
    elapsed_hours = np.asarray((selected - selected[0]) / np.timedelta64(1, "h"), dtype=np.float64)
    calendar_days = selected.astype("datetime64[D]")
    clock_hours = np.asarray((selected - calendar_days) / np.timedelta64(1, "h"), dtype=np.float64)
    if np.any(np.diff(elapsed_hours) < 0):
        raise ValueError("timestamps referenced by row_index must be monotonic")
    return elapsed_hours, clock_hours, calendar_days


def _standardized(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    scale = float(values.std()) if values.size else 1.0
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return (values - mean) / scale, mean, scale


def _effective_rank(Q: torch.Tensor) -> tuple[int, float, torch.Tensor]:
    if min(Q.shape) == 0:
        return 0, 0.0, torch.empty(0, dtype=Q.dtype, device=Q.device)
    singular_values = torch.linalg.svdvals(Q)
    tolerance = float(max(Q.shape) * torch.finfo(Q.dtype).eps * float(singular_values[0]))
    return int(torch.count_nonzero(singular_values > tolerance)), tolerance, singular_values


def build_background_basis(
    row_index: Sequence[dict[str, Any]],
    timestamps: Any,
    sensor_xy: np.ndarray | None = None,
    config: BackgroundBasisConfig | None = None,
    user_basis: np.ndarray | None = None,
    user_basis_names: Sequence[str] | None = None,
) -> BackgroundBasisResult:
    cfg = config or BackgroundBasisConfig()
    if cfg.temporal_polynomial_degree < 0 or cfg.daily_harmonics < 0:
        raise ValueError("polynomial degree and daily harmonic count must be nonnegative")
    if cfg.max_background_rank < 0:
        raise ValueError("max_background_rank must be nonnegative")
    if cfg.basis_mode not in {"normal", "stress"}:
        raise ValueError("basis_mode must be 'normal' or 'stress'")

    rows, time_indices, sensor_ids = _validated_rows(row_index)
    elapsed_hours, clock_hours, day_values = _timestamp_hours(timestamps, time_indices)
    M, T = len(sensor_ids), len(time_indices)
    mT = len(rows)
    row_elapsed_hours = np.repeat(elapsed_hours, M) if M else np.empty(0, dtype=np.float64)
    row_clock_hours = np.repeat(clock_hours, M) if M else np.empty(0, dtype=np.float64)
    columns: list[np.ndarray] = []
    names: list[str] = []
    provenance: list[dict[str, Any]] = []
    scaling: dict[str, Any] = {}

    def add(name: str, values: np.ndarray, kind: str, **details: Any) -> None:
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)
        provenance.append({"name": name, "type": kind, **details})

    if cfg.include_constant:
        add("constant", np.ones(mT), "constant")
    if cfg.temporal_polynomial_degree:
        standardized_time, center, scale = _standardized(row_elapsed_hours)
        scaling["temporal_hours"] = {"center": center, "scale": scale, "semantics": "elapsed_from_first_timestamp"}
        for degree in range(1, cfg.temporal_polynomial_degree + 1):
            add(f"time_polynomial_{degree}", standardized_time ** degree, "temporal_polynomial", degree=degree)
    for harmonic in range(1, cfg.daily_harmonics + 1):
        angle = 2.0 * np.pi * harmonic * row_clock_hours / 24.0
        add(
            f"daily_sin_{harmonic}", np.sin(angle), "daily_harmonic",
            harmonic=harmonic, component="sin", time_semantics="fractional_hours_since_midnight",
        )
        add(
            f"daily_cos_{harmonic}", np.cos(angle), "daily_harmonic",
            harmonic=harmonic, component="cos", time_semantics="fractional_hours_since_midnight",
        )
    if cfg.day_intercepts and T:
        unique_days = list(dict.fromkeys(str(day) for day in day_values))
        row_days = np.repeat(np.asarray([str(day) for day in day_values]), M)
        for day in unique_days[1:]:
            add(f"day_{day}", (row_days == day).astype(np.float64), "day_intercept", reference_day=unique_days[0])

    if cfg.regional_coordinate_trends:
        if sensor_xy is None:
            raise ValueError("sensor_xy is required for regional coordinate trends")
        xy = np.asarray(sensor_xy, dtype=np.float64)
        if xy.shape != (M, 2) or not np.isfinite(xy).all():
            raise ValueError("sensor_xy must be finite with shape [sensor_count,2]")
        for axis, label in enumerate(("x", "y")):
            standardized_xy, center, scale = _standardized(xy[:, axis])
            scaling[f"sensor_{label}"] = {"center": center, "scale": scale}
            add(f"regional_{label}_trend", np.tile(standardized_xy, T), "regional_coordinate_trend", axis=label)
    if cfg.sensor_offsets:
        for sensor_index, sensor_id in enumerate(sensor_ids[1:], start=1):
            values = np.zeros((T, M), dtype=np.float64)
            values[:, sensor_index] = 1.0
            add(f"sensor_offset_{sensor_id}", values.ravel(), "sensor_offset", reference_sensor=sensor_ids[0])

    if user_basis is not None:
        user = np.asarray(user_basis, dtype=np.float64)
        if user.ndim != 2 or user.shape[0] != mT:
            raise ValueError("user_basis must have shape [row_count,r_user]")
        if not np.isfinite(user).all():
            raise ValueError("user_basis must contain only finite values")
        if user_basis_names is None:
            supplied_names = [f"user_{i}" for i in range(user.shape[1])]
        else:
            supplied_names = [str(name) for name in user_basis_names]
            if len(supplied_names) != user.shape[1]:
                raise ValueError("user_basis_names length must match user_basis columns")
        for index, name in enumerate(supplied_names):
            add(name, user[:, index], "user")
    elif user_basis_names is not None:
        raise ValueError("user_basis_names requires user_basis")

    if len(set(names)) != len(names):
        raise ValueError("background column names must be unique")
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, default=torch.float64)
    if columns:
        Q = torch.stack([torch.as_tensor(col, dtype=dtype, device=device) for col in columns], dim=1)
    else:
        Q = torch.empty((mT, 0), dtype=dtype, device=device)
    rank, tolerance, singular_values = _effective_rank(Q)
    if cfg.basis_mode == "normal" and rank > cfg.max_background_rank:
        raise ValueError(f"normal background effective rank {rank} exceeds max_background_rank={cfg.max_background_rank}")
    metadata = {
        "config": asdict(cfg),
        "basis_mode": cfg.basis_mode,
        "column_provenance": provenance,
        "requested_column_count": int(Q.shape[1]),
        "effective_rank": rank,
        "rank_tolerance": tolerance,
        "singular_values": [float(v) for v in singular_values.detach().cpu().tolist()],
        "row_count": mT,
        "time_count": T,
        "sensor_count": M,
        "time_indices": time_indices,
        "sensor_ids": sensor_ids,
        "row_ordering": "time_major_sensor_minor",
        "centering_and_scaling": scaling,
        "time_semantics": {
            "temporal_polynomials": "elapsed_hours_from_first_timestamp",
            "daily_harmonics": "fractional_local_clock_hours_since_calendar_day_boundary",
        },
        "max_background_rank": cfg.max_background_rank,
        **runtime_provenance(device, dtype),
    }
    return BackgroundBasisResult(Q=Q, column_names=names, row_index=rows, metadata=metadata)


__all__ = ["BackgroundBasisConfig", "BackgroundBasisResult", "build_background_basis"]
