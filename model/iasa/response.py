from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from model.iasa.activity import TemporalBasis
from model.iasa.wind import WindSequence


RESPONSE_IMPLEMENTATION = "open_boundary_gaussian_puff"
BOUNDARY_MODE = "open"


class WindSampler(Protocol):
    provider: str
    metadata: dict[str, Any]

    def sample(self, t_index: float, position_xy: np.ndarray) -> np.ndarray:
        """Return wind vector [vx, vy] at time index and position."""


@dataclass(frozen=True)
class ResponseConfig:
    dt: float = 1.0
    lag_window_steps: int = 12
    substep_dt: float = 0.25
    kernel_truncation_radius: float = 3.0
    source_cell_threshold: float = 1e-6
    baseline_policy: str = "zero_source"
    wind_interpolation: str = "linear"
    zero_wind_orientation: float = 0.0
    trim_initial_lag: bool = False
    max_kernel_diagnostic_records: int | None = 500


@dataclass(frozen=True)
class DispersionConfig:
    sigma_parallel: float = 0.7
    sigma_perp: float = 0.25
    min_dispersion_time: float = 0.25


@dataclass(frozen=True)
class Observer:
    sensor_ids: list[str]
    sensor_xy: np.ndarray


@dataclass(frozen=True)
class CityWindSampler:
    timestamps: Any
    vx: np.ndarray
    vy: np.ndarray
    provider: str
    metadata: dict[str, Any]
    interpolation: str = "linear"

    @classmethod
    def from_wind_sequence(cls, wind: WindSequence, *, interpolation: str = "linear") -> "CityWindSampler":
        return cls(
            timestamps=wind.timestamps,
            vx=np.asarray(wind.vx, dtype=np.float32),
            vy=np.asarray(wind.vy, dtype=np.float32),
            provider=wind.provider,
            metadata=dict(wind.metadata),
            interpolation=interpolation,
        )

    def sample(self, t_index: float, position_xy: np.ndarray) -> np.ndarray:
        del position_xy
        t = float(np.clip(t_index, 0.0, max(0, len(self.vx) - 1)))
        if self.interpolation == "nearest":
            idx = int(round(t))
            return np.asarray([self.vx[idx], self.vy[idx]], dtype=np.float32)
        if self.interpolation != "linear":
            raise ValueError("CityWindSampler interpolation must be 'linear' or 'nearest'")
        lo = int(np.floor(t))
        hi = min(lo + 1, len(self.vx) - 1)
        frac = t - lo
        vx = (1.0 - frac) * float(self.vx[lo]) + frac * float(self.vx[hi])
        vy = (1.0 - frac) * float(self.vy[lo]) + frac * float(self.vy[hi])
        return np.asarray([vx, vy], dtype=np.float32)


@dataclass(frozen=True)
class ResponseMatrixResult:
    H_lag: np.ndarray
    metadata: dict[str, Any]
    row_index: list[dict[str, Any]]
    column_index: list[dict[str, Any]]
    baseline: np.ndarray


def _as_basis(activity_basis: TemporalBasis | np.ndarray, T: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if isinstance(activity_basis, TemporalBasis):
        values = np.asarray(activity_basis.values, dtype=np.float32)
        names = list(activity_basis.names)
        metadata = dict(activity_basis.metadata)
    else:
        values = np.asarray(activity_basis, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("activity_basis must have shape [T,B]")
        names = [f"basis_{i}" for i in range(values.shape[1])]
        metadata = {}
    if values.ndim != 2:
        raise ValueError("activity_basis must have shape [T,B]")
    if values.shape[0] != T:
        raise ValueError(f"activity_basis has {values.shape[0]} rows but expected {T}")
    if not np.isfinite(values).all():
        raise ValueError("activity_basis must contain only finite values")
    return values, names, metadata


def _as_wind_sampler(wind: WindSampler | WindSequence, config: ResponseConfig) -> WindSampler:
    if isinstance(wind, WindSequence):
        return CityWindSampler.from_wind_sequence(wind, interpolation=config.wind_interpolation)
    if not hasattr(wind, "sample"):
        raise TypeError("wind_sampler_or_sequence must be a WindSequence or implement sample(t, position)")
    return wind


def _validate_inputs(
    source_maps: np.ndarray,
    source_names: Sequence[str],
    observer: Observer,
    config: ResponseConfig,
    dispersion: DispersionConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = np.asarray(source_maps, dtype=np.float32)
    names = [str(name) for name in source_names]
    sensors = np.asarray(observer.sensor_xy, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError("source_maps must have shape [K,Nx,Ny]")
    if len(names) != maps.shape[0]:
        raise ValueError("source_names length must match source_maps K")
    if len(set(names)) != len(names):
        raise ValueError("source_names must be unique")
    if not np.isfinite(maps).all() or np.any(maps < 0):
        raise ValueError("source_maps must be finite and nonnegative")
    if sensors.ndim != 2 or sensors.shape[1] != 2:
        raise ValueError("observer.sensor_xy must have shape [M,2]")
    if len(observer.sensor_ids) != sensors.shape[0]:
        raise ValueError("observer.sensor_ids length must match sensor count")
    Nx, Ny = maps.shape[1], maps.shape[2]
    if np.any(sensors[:, 0] < -0.5) or np.any(sensors[:, 0] > Nx - 0.5):
        raise ValueError("observer x coordinates must lie within physical domain extents")
    if np.any(sensors[:, 1] < -0.5) or np.any(sensors[:, 1] > Ny - 0.5):
        raise ValueError("observer y coordinates must lie within physical domain extents")
    if config.dt <= 0 or config.substep_dt <= 0:
        raise ValueError("dt and substep_dt must be positive")
    if config.lag_window_steps < 1:
        raise ValueError("lag_window_steps must be >= 1")
    if config.kernel_truncation_radius <= 0:
        raise ValueError("kernel_truncation_radius must be positive")
    if config.baseline_policy != "zero_source":
        raise ValueError("Task 5 supports baseline_policy='zero_source' only")
    if config.max_kernel_diagnostic_records is not None:
        if (
            isinstance(config.max_kernel_diagnostic_records, bool)
            or not isinstance(config.max_kernel_diagnostic_records, (int, np.integer))
            or config.max_kernel_diagnostic_records < 0
        ):
            raise ValueError("max_kernel_diagnostic_records must be None or a nonnegative integer")
    if dispersion.sigma_parallel <= 0 or dispersion.sigma_perp <= 0 or dispersion.min_dispersion_time <= 0:
        raise ValueError("dispersion parameters must be positive")
    return maps, names, sensors


def _inside_extents(pos: np.ndarray, Nx: int, Ny: int) -> bool:
    return bool((-0.5 <= pos[0] <= Nx - 0.5) and (-0.5 <= pos[1] <= Ny - 0.5))


def _advect(
    start: np.ndarray,
    tau: int,
    target_t: int,
    sampler: WindSampler,
    config: ResponseConfig,
    Nx: int,
    Ny: int,
) -> tuple[np.ndarray, bool, float | None, np.ndarray]:
    pos = np.asarray(start, dtype=np.float32).copy()
    if target_t == tau:
        wind0 = np.asarray(sampler.sample(float(tau), pos), dtype=np.float32)
        if wind0.shape != (2,) or not np.isfinite(wind0).all():
            raise ValueError("wind sampler must return finite vectors with shape (2,)")
        return pos, False, None, wind0
    elapsed = 0.0
    duration = float(target_t - tau) * float(config.dt)
    wind_sum = np.zeros(2, dtype=np.float64)
    wind_weight = 0.0
    while elapsed < duration - 1e-12:
        step = min(float(config.substep_dt), duration - elapsed)
        t_index = float(tau) + elapsed / float(config.dt)
        wind = np.asarray(sampler.sample(t_index, pos), dtype=np.float32)
        if wind.shape != (2,) or not np.isfinite(wind).all():
            raise ValueError("wind sampler must return finite vectors with shape (2,)")
        wind_sum += wind.astype(np.float64) * step
        wind_weight += step
        pos = pos + step * wind
        elapsed += step
        if not _inside_extents(pos, Nx, Ny):
            return pos, True, float(tau) + elapsed / float(config.dt), (wind_sum / max(wind_weight, 1e-12)).astype(np.float32)
    mean_wind = (wind_sum / max(wind_weight, 1e-12)).astype(np.float32)
    return pos, False, None, mean_wind


def _orientation(mean_wind: np.ndarray, zero_wind_orientation: float) -> tuple[np.ndarray, np.ndarray]:
    norm = float(np.linalg.norm(mean_wind))
    if norm > 1e-8:
        e_parallel = mean_wind.astype(np.float64) / norm
    else:
        angle = float(zero_wind_orientation)
        e_parallel = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    e_perp = np.asarray([-e_parallel[1], e_parallel[0]], dtype=np.float64)
    return e_parallel, e_perp


def _gaussian_at_points(
    points: np.ndarray,
    center: np.ndarray,
    e_parallel: np.ndarray,
    e_perp: np.ndarray,
    var_parallel: float,
    var_perp: float,
) -> np.ndarray:
    delta = points.astype(np.float64) - center.astype(np.float64)
    d_parallel = delta @ e_parallel
    d_perp = delta @ e_perp
    norm = 2.0 * np.pi * np.sqrt(var_parallel * var_perp)
    values = np.exp(-0.5 * ((d_parallel ** 2) / var_parallel + (d_perp ** 2) / var_perp)) / norm
    return values.astype(np.float32)


def _retained_kernel_mass(
    center: np.ndarray,
    e_parallel: np.ndarray,
    e_perp: np.ndarray,
    var_parallel: float,
    var_perp: float,
    Nx: int,
    Ny: int,
    truncation: float,
) -> float:
    max_sigma = float(np.sqrt(max(var_parallel, var_perp)))
    radius = float(truncation) * max_sigma
    xmin = max(0, int(np.floor(center[0] - radius)))
    xmax = min(Nx - 1, int(np.ceil(center[0] + radius)))
    ymin = max(0, int(np.floor(center[1] - radius)))
    ymax = min(Ny - 1, int(np.ceil(center[1] + radius)))
    if xmin > xmax or ymin > ymax:
        return 0.0
    offsets = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
    xs, ys, ox, oy = np.meshgrid(
        np.arange(xmin, xmax + 1, dtype=np.float32),
        np.arange(ymin, ymax + 1, dtype=np.float32),
        offsets,
        offsets,
        indexing="ij",
    )
    points = np.stack([(xs + ox).ravel(), (ys + oy).ravel()], axis=1).astype(np.float32)
    inside = (
        (points[:, 0] >= -0.5)
        & (points[:, 0] <= Nx - 0.5)
        & (points[:, 1] >= -0.5)
        & (points[:, 1] <= Ny - 0.5)
    )
    if not np.any(inside):
        return 0.0
    return float(np.sum(_gaussian_at_points(points[inside], center, e_parallel, e_perp, var_parallel, var_perp)) / 25.0)


def _make_row_index(sensor_ids: list[str], T: int, trim_start: int) -> list[dict[str, Any]]:
    rows = []
    for t in range(trim_start, T):
        for sensor_idx, sensor_id in enumerate(sensor_ids):
            rows.append({"time_index": t, "sensor_index": sensor_idx, "sensor_id": sensor_id})
    return rows


def _column_index(source_names: list[str], basis_names: list[str]) -> list[dict[str, Any]]:
    cols = []
    for k, source_name in enumerate(source_names):
        for b, basis_name in enumerate(basis_names):
            cols.append({"source_index": k, "source_name": source_name, "basis_index": b, "basis_name": basis_name})
    return cols


def build_lagged_response_matrix(
    source_maps: np.ndarray,
    source_names: Sequence[str],
    activity_basis: TemporalBasis | np.ndarray,
    observer: Observer,
    wind_sampler_or_sequence: WindSampler | WindSequence,
    response_config: ResponseConfig | None = None,
    dispersion_config: DispersionConfig | None = None,
) -> ResponseMatrixResult:
    config = response_config or ResponseConfig()
    dispersion = dispersion_config or DispersionConfig()
    maps, names, sensors = _validate_inputs(source_maps, source_names, observer, config, dispersion)
    sampler = _as_wind_sampler(wind_sampler_or_sequence, config)
    if isinstance(sampler, CityWindSampler):
        T = len(sampler.vx)
        basis_values, basis_names, basis_metadata = _as_basis(activity_basis, T)
    else:
        if not hasattr(activity_basis, "values") and np.asarray(activity_basis).ndim == 2:
            T = int(np.asarray(activity_basis).shape[0])
        elif isinstance(activity_basis, TemporalBasis):
            T = int(np.asarray(activity_basis.values).shape[0])
        else:
            raise ValueError("Cannot infer T for custom wind sampler without activity basis values")
        basis_values, basis_names, basis_metadata = _as_basis(activity_basis, T)

    K, Nx, Ny = maps.shape
    B = basis_values.shape[1]
    M = sensors.shape[0]
    trim_start = config.lag_window_steps - 1 if config.trim_initial_lag else 0
    T_effective = T - trim_start
    if T_effective <= 0:
        raise ValueError("trim_initial_lag removed all rows")
    H = np.zeros((M * T_effective, K * B), dtype=np.float32)
    baseline = np.zeros(M * T_effective, dtype=np.float32)
    col_index = _column_index(names, basis_names)
    row_index = _make_row_index(list(observer.sensor_ids), T, trim_start)

    retained_by_col = np.zeros(K * B, dtype=np.float64)
    dropped_by_col = np.zeros(K * B, dtype=np.float64)
    emitted_by_col = np.zeros(K * B, dtype=np.float64)
    kernel_count_by_col = np.zeros(K * B, dtype=np.int64)
    quadrature_clip_count_by_col = np.zeros(K * B, dtype=np.int64)
    max_raw_retained_fraction_by_col = np.zeros(K * B, dtype=np.float64)
    exited_by_col = np.zeros(K * B, dtype=np.int64)
    released_exit_mass_by_col = np.zeros(K * B, dtype=np.float64)
    first_exit_by_release: list[dict[str, Any]] = []
    kernel_summaries: list[dict[str, Any]] = []
    kernel_diagnostic_total_count = 0

    for k in range(K):
        cells = np.argwhere(maps[k] > float(config.source_cell_threshold))
        for b in range(B):
            col = k * B + b
            for tau in range(T):
                basis_mass = float(basis_values[tau, b])
                if basis_mass == 0.0:
                    continue
                for ix, iy in cells:
                    cell_mass = float(maps[k, ix, iy]) * basis_mass
                    if cell_mass <= 0.0:
                        continue
                    start = np.asarray([float(ix), float(iy)], dtype=np.float32)
                    release_exited = False
                    for t in range(tau, min(T, tau + int(config.lag_window_steps))):
                        if t < trim_start:
                            continue
                        pos, exited, exit_time, mean_wind = _advect(start, tau, t, sampler, config, Nx, Ny)
                        if exited:
                            if not release_exited:
                                exited_by_col[col] += 1
                                released_exit_mass_by_col[col] += cell_mass
                                first_exit_by_release.append({
                                    "source_index": k,
                                    "basis_index": b,
                                    "release_time_index": tau,
                                    "cell": [int(ix), int(iy)],
                                    "exit_time_index": exit_time,
                                    "released_mass": cell_mass,
                                })
                                release_exited = True
                            break
                        age = float(t - tau) * float(config.dt)
                        effective_age = max(age, float(dispersion.min_dispersion_time))
                        var_parallel = float(dispersion.sigma_parallel) ** 2 * effective_age
                        var_perp = float(dispersion.sigma_perp) ** 2 * effective_age
                        e_parallel, e_perp = _orientation(mean_wind, config.zero_wind_orientation)
                        sensor_values = _gaussian_at_points(sensors, pos, e_parallel, e_perp, var_parallel, var_perp)
                        row_base = (t - trim_start) * M
                        H[row_base: row_base + M, col] += (cell_mass * sensor_values).astype(np.float32)
                        raw_retained_fraction = _retained_kernel_mass(
                            pos,
                            e_parallel,
                            e_perp,
                            var_parallel,
                            var_perp,
                            Nx,
                            Ny,
                            config.kernel_truncation_radius,
                        )
                        retained_fraction = float(np.clip(raw_retained_fraction, 0.0, 1.0))
                        retained_mass = cell_mass * retained_fraction
                        dropped_mass = cell_mass - retained_mass
                        emitted_by_col[col] += cell_mass
                        retained_by_col[col] += retained_mass
                        dropped_by_col[col] += dropped_mass
                        kernel_count_by_col[col] += 1
                        max_raw_retained_fraction_by_col[col] = max(
                            max_raw_retained_fraction_by_col[col], raw_retained_fraction
                        )
                        if raw_retained_fraction < 0.0 or raw_retained_fraction > 1.0:
                            quadrature_clip_count_by_col[col] += 1
                        kernel_diagnostic_total_count += 1
                        record_limit = config.max_kernel_diagnostic_records
                        if record_limit is None or len(kernel_summaries) < record_limit:
                            kernel_summaries.append({
                                "source_index": k,
                                "basis_index": b,
                                "release_time_index": tau,
                                "observation_time_index": t,
                                "cell": [int(ix), int(iy)],
                                "age": age,
                                "effective_age": effective_age,
                                "emitted_mass": cell_mass,
                                "raw_retained_fraction": raw_retained_fraction,
                                "retained_fraction": retained_fraction,
                                "retained_mass": retained_mass,
                                "dropped_mass": dropped_mass,
                            })

    metadata: dict[str, Any] = {
        "boundary_mode": BOUNDARY_MODE,
        "response_implementation": RESPONSE_IMPLEMENTATION,
        "response_config": asdict(config),
        "dispersion_config": asdict(dispersion),
        "source_names": names,
        "basis_names": basis_names,
        "basis_metadata": basis_metadata,
        "wind_provider": getattr(sampler, "provider", "custom_wind_sampler"),
        "wind_metadata": getattr(sampler, "metadata", {}),
        "wind_vx": getattr(sampler, "vx", None).tolist() if hasattr(sampler, "vx") else None,
        "wind_vy": getattr(sampler, "vy", None).tolist() if hasattr(sampler, "vy") else None,
        "T": T,
        "T_effective": T_effective,
        "trim_start": trim_start,
        "row_index": row_index,
        "column_index": col_index,
        "baseline_policy": config.baseline_policy,
        "baseline": baseline.astype(float).tolist(),
        "kernel_emitted_mass_by_column": emitted_by_col.astype(float).tolist(),
        "kernel_observation_count_by_column": kernel_count_by_col.astype(int).tolist(),
        "kernel_mass_retained_by_column": retained_by_col.astype(float).tolist(),
        "dropped_mass_by_column": dropped_by_col.astype(float).tolist(),
        "kernel_diagnostic_total_count": int(kernel_diagnostic_total_count),
        "kernel_diagnostic_stored_count": len(kernel_summaries),
        "kernel_diagnostics_truncated": len(kernel_summaries) < kernel_diagnostic_total_count,
        "kernel_quadrature_clip_count_by_column": quadrature_clip_count_by_col.astype(int).tolist(),
        "max_raw_retained_fraction_by_column": max_raw_retained_fraction_by_col.astype(float).tolist(),
        "exit_count_by_column": exited_by_col.astype(int).tolist(),
        "released_mass_exited_by_column": released_exit_mass_by_col.astype(float).tolist(),
        "first_exit_by_release": first_exit_by_release,
        "kernel_mass_summaries": kernel_summaries,
    }
    return ResponseMatrixResult(H_lag=H, metadata=metadata, row_index=row_index, column_index=col_index, baseline=baseline)


__all__ = [
    "BOUNDARY_MODE",
    "RESPONSE_IMPLEMENTATION",
    "CityWindSampler",
    "DispersionConfig",
    "Observer",
    "ResponseConfig",
    "ResponseMatrixResult",
    "WindSampler",
    "build_lagged_response_matrix",
]
