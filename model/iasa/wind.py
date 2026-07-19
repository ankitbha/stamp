from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


def transport_vectors_from_wd_ws(wd_deg: Any, ws: Any) -> np.ndarray:
    """Convert meteorological WD (degrees, "from") and speed WS to transport vectors.

    Paper eq. wind_direction_conversion:
        Ux = -WS*sin(WD*pi/180),  Vy = -WS*cos(WD*pi/180).
    WD is the direction the wind comes from; the returned vector points where
    pollution is transported. Returns an array with a trailing axis of size 2.
    """

    wd = np.asarray(wd_deg, dtype=np.float64)
    speed = np.asarray(ws, dtype=np.float64)
    if wd.shape != speed.shape:
        raise ValueError("wd_deg and ws must have the same shape")
    rad = wd * np.pi / 180.0
    ux = -speed * np.sin(rad)
    vy = -speed * np.cos(rad)
    return np.stack([ux, vy], axis=-1).astype(np.float32)


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


# --------------------------------------------------------------------------- #
# Gridded FieldFormer-style wind field and transport ensembles (Task 9A)       #
# --------------------------------------------------------------------------- #
class CoordinateQueryImputer(Protocol):
    """A coordinate-query wind imputer: predict transport vectors anywhere.

    Given sparse masked station observations, ``query`` returns transport vectors
    at arbitrary spatial query coordinates for one hour. FieldFormer is one such
    model; the default here is a kernel interpolator satisfying the same contract.
    """

    name: str

    def query(
        self,
        coords_xy: np.ndarray,
        t_index: int,
        station_coords: np.ndarray,
        station_vectors: np.ndarray,
        station_mask: np.ndarray,
    ) -> np.ndarray:
        """Return ``[n, 2]`` transport vectors at ``coords_xy`` for hour ``t_index``."""


@dataclass(frozen=True)
class KernelCoordinateQueryImputer:
    """Default coordinate-query imputer: normalized Gaussian-kernel interpolation.

    For each hour, predicts at a query location as a distance-weighted average of
    the observed station vectors, producing a spatially varying field (not a
    city-averaged sequence). Not FieldFormer, but pluggable through the same
    protocol so a trained model can replace it without API changes.
    """

    length_scale: float = 4.0
    name: str = "kernel_coordinate_query"

    def query(
        self,
        coords_xy: np.ndarray,
        t_index: int,
        station_coords: np.ndarray,
        station_vectors: np.ndarray,
        station_mask: np.ndarray,
    ) -> np.ndarray:
        coords = np.asarray(coords_xy, dtype=np.float64)
        stations = np.asarray(station_coords, dtype=np.float64)
        vectors = np.asarray(station_vectors, dtype=np.float64)[:, int(t_index), :]
        mask = np.asarray(station_mask, dtype=bool)[:, int(t_index)]
        if not np.any(mask):
            raise ValueError(f"hour {t_index} has no observed station vectors to query from")
        active_coords = stations[mask]
        active_vectors = vectors[mask]
        if float(self.length_scale) <= 0:
            raise ValueError("length_scale must be positive")
        # [n, n_active] squared distances -> Gaussian weights.
        diff = coords[:, None, :] - active_coords[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        weights = np.exp(-0.5 * sq / (float(self.length_scale) ** 2))
        denom = np.sum(weights, axis=1, keepdims=True)
        # Fall back to the observed-station mean where a query is far from all stations.
        far = denom[:, 0] <= 1e-12
        predicted = np.zeros((coords.shape[0], 2), dtype=np.float64)
        safe = ~far
        predicted[safe] = (weights[safe] @ active_vectors) / denom[safe]
        if np.any(far):
            predicted[far] = active_vectors.mean(axis=0)
        return predicted.astype(np.float32)


@dataclass(frozen=True)
class GriddedWindField:
    field: np.ndarray  # [T, Nx, Ny, 2] grid-displacement (cells per response step)
    physical_field: np.ndarray  # [T, Nx, Ny, 2] physical transport vectors
    grid_shape: tuple[int, int]
    timestamps: Any
    station_coords: np.ndarray
    station_vectors: np.ndarray
    station_mask: np.ndarray
    dt_s: float
    dx_m: float
    dy_m: float
    convention: str
    provider: str
    ensemble_kind: str
    metadata: dict[str, Any]


def _cell_centers(nx: int, ny: int) -> np.ndarray:
    # x-major ordering: g = ix*ny + iy.
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float64)


def _grid_displacement(physical: np.ndarray, dt_s: float, dx_m: float, dy_m: float) -> np.ndarray:
    if dt_s <= 0 or dx_m <= 0 or dy_m <= 0:
        raise ValueError("dt_s, dx_m, dy_m must be positive")
    out = np.empty_like(physical)
    out[..., 0] = physical[..., 0] * float(dt_s) / float(dx_m)
    out[..., 1] = physical[..., 1] * float(dt_s) / float(dy_m)
    return out


def build_gridded_wind_field(
    station_coords: np.ndarray,
    station_vectors: np.ndarray,
    station_mask: np.ndarray,
    timestamps: Any,
    grid_shape: tuple[int, int],
    *,
    imputer: CoordinateQueryImputer | None = None,
    dt_s: float = 3600.0,
    dx_m: float = 1000.0,
    dy_m: float = 1000.0,
    seed: int = 0,
    provider: str = "gridded_kernel",
    ensemble_kind: str = "transport",
    extra_metadata: dict[str, Any] | None = None,
) -> GriddedWindField:
    """Query a coordinate-query imputer on every response-grid cell and hour.

    Produces ``W_hat in R^{T x Nx x Ny x 2}`` (paper eq. fieldformer_wind_field),
    both as physical transport vectors and as grid displacement
    (eq. wind_unit_conversion), recording the scales and convention.
    """

    from model.iasa.backend import validate_ensemble_kind

    validate_ensemble_kind(ensemble_kind)
    stations = np.asarray(station_coords, dtype=np.float64)
    vectors = np.asarray(station_vectors, dtype=np.float32)
    mask = np.asarray(station_mask, dtype=bool)
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    T = len(timestamps)
    if stations.ndim != 2 or stations.shape[1] != 2:
        raise ValueError("station_coords must have shape [S, 2] in grid coordinates")
    if vectors.shape != (stations.shape[0], T, 2):
        raise ValueError("station_vectors must have shape [S, T, 2]")
    if mask.shape != (stations.shape[0], T):
        raise ValueError("station_mask must have shape [S, T]")
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_shape must be positive")

    model = imputer or KernelCoordinateQueryImputer()
    coords = _cell_centers(nx, ny)  # [n, 2], x-major
    physical = np.empty((T, nx * ny, 2), dtype=np.float32)
    for t in range(T):
        physical[t] = model.query(coords, t, stations, vectors, mask)
    physical = physical.reshape(T, nx, ny, 2)
    if not np.isfinite(physical).all():
        raise ValueError("gridded wind field contains non-finite values")
    displacement = _grid_displacement(physical, dt_s, dx_m, dy_m)
    metadata: dict[str, Any] = {
        "imputer": getattr(model, "name", "unknown"),
        "seed": int(seed),
        "grid_shape": [nx, ny],
        "n_cells": nx * ny,
        "T": int(T),
        "station_count": int(stations.shape[0]),
        "observed_vector_count": int(np.count_nonzero(mask)),
        "dt_s": float(dt_s),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "convention": "transport_vectors_wd_ws_eq_wind_direction_conversion",
        "units_physical": "meters_per_second",
        "units_field": "grid_cells_per_response_step",
        "cell_ordering": "x_major",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return GriddedWindField(
        field=displacement,
        physical_field=physical,
        grid_shape=(nx, ny),
        timestamps=timestamps,
        station_coords=stations,
        station_vectors=vectors,
        station_mask=mask,
        dt_s=float(dt_s),
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        convention=metadata["convention"],
        provider=provider,
        ensemble_kind=ensemble_kind,
        metadata=metadata,
    )


def _direction_error_degrees(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    a = np.degrees(np.arctan2(pred[..., 1], pred[..., 0]))
    b = np.degrees(np.arctan2(obs[..., 1], obs[..., 0]))
    diff = np.abs(a - b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def evaluate_gridded_wind_heldout(
    station_coords: np.ndarray,
    station_vectors: np.ndarray,
    station_mask: np.ndarray,
    timestamps: Any,
    *,
    imputer: CoordinateQueryImputer | None = None,
    holdout_station_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Mask held-out stations from the imputer inputs and measure held-out error.

    Reports vector RMSE, mean absolute direction error (degrees, circular), and
    mean absolute speed error over held-out station-times that are observed.
    Dense city-wide truth is not assumed.
    """

    stations = np.asarray(station_coords, dtype=np.float64)
    vectors = np.asarray(station_vectors, dtype=np.float32)
    mask = np.asarray(station_mask, dtype=bool)
    T = len(timestamps)
    holdout = sorted(set(int(i) for i in holdout_station_indices))
    if not holdout or any(i < 0 or i >= stations.shape[0] for i in holdout):
        raise ValueError("holdout_station_indices must be a nonempty valid subset of stations")
    train_mask = mask.copy()
    train_mask[holdout, :] = False
    if not np.any(train_mask):
        raise ValueError("no training stations remain after holding out the requested split")

    model = imputer or KernelCoordinateQueryImputer()
    pred_list = []
    obs_list = []
    for t in range(T):
        observed_holdout = [i for i in holdout if mask[i, t]]
        if not observed_holdout:
            continue
        coords = stations[observed_holdout]
        predicted = model.query(coords, t, stations, vectors, train_mask)
        pred_list.append(predicted)
        obs_list.append(vectors[observed_holdout, t, :])
    if not pred_list:
        raise ValueError("no held-out observed station-times available for validation")
    pred = np.concatenate(pred_list, axis=0).astype(np.float64)
    obs = np.concatenate(obs_list, axis=0).astype(np.float64)
    vector_rmse = float(np.sqrt(np.mean(np.sum((pred - obs) ** 2, axis=1))))
    direction_mae = float(np.mean(_direction_error_degrees(pred, obs)))
    speed_mae = float(np.mean(np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(obs, axis=1))))
    return {
        "holdout_station_indices": holdout,
        "n_heldout_station_times": int(pred.shape[0]),
        "vector_rmse": vector_rmse,
        "direction_mae_degrees": direction_mae,
        "speed_mae": speed_mae,
        "imputer": getattr(model, "name", "unknown"),
    }


def build_wind_field_ensemble(
    station_coords: np.ndarray,
    station_vectors: np.ndarray,
    station_mask: np.ndarray,
    timestamps: Any,
    grid_shape: tuple[int, int],
    *,
    n_members: int,
    method: str = "station_bootstrap",
    imputer: CoordinateQueryImputer | None = None,
    dt_s: float = 3600.0,
    dx_m: float = 1000.0,
    dy_m: float = 1000.0,
    seed: int = 0,
    residual_scale: float = 0.0,
    provider: str = "gridded_kernel_ensemble",
) -> list[GriddedWindField]:
    """Build R transport wind-field ensemble members.

    ``station_bootstrap`` resamples stations with replacement per member;
    ``residual_perturbation`` adds member-specific Gaussian noise (scaled by
    ``residual_scale``, e.g. a held-out-calibrated error) to the station vectors.
    Every member is tagged ``ensemble_kind="transport"`` and never pooled with
    inventory scenarios.
    """

    if int(n_members) <= 0:
        raise ValueError("n_members must be positive")
    if method not in {"station_bootstrap", "residual_perturbation"}:
        raise ValueError("method must be 'station_bootstrap' or 'residual_perturbation'")
    stations = np.asarray(station_coords, dtype=np.float64)
    vectors = np.asarray(station_vectors, dtype=np.float32)
    mask = np.asarray(station_mask, dtype=bool)
    S = stations.shape[0]
    members: list[GriddedWindField] = []
    for r in range(int(n_members)):
        rng = np.random.default_rng(int(seed) + r)
        if method == "station_bootstrap":
            pick = rng.integers(0, S, size=S)
            member = build_gridded_wind_field(
                stations[pick], vectors[pick], mask[pick], timestamps, grid_shape,
                imputer=imputer, dt_s=dt_s, dx_m=dx_m, dy_m=dy_m, seed=int(seed) + r,
                provider=f"{provider}_member{r}",
                extra_metadata={"ensemble_method": method, "member_index": r, "resampled_stations": pick.tolist()},
            )
        else:
            noise = (residual_scale * rng.standard_normal(size=vectors.shape)).astype(np.float32)
            member = build_gridded_wind_field(
                stations, vectors + noise, mask, timestamps, grid_shape,
                imputer=imputer, dt_s=dt_s, dx_m=dx_m, dy_m=dy_m, seed=int(seed) + r,
                provider=f"{provider}_member{r}",
                extra_metadata={"ensemble_method": method, "member_index": r, "residual_scale": float(residual_scale)},
            )
        members.append(member)
    return members


def gridded_new_delhi_wind_field(
    data_csv: str | Path = "sim/govdata_1H_current.csv",
    locations_csv: str | Path = "sim/govdata_locations.csv",
    grid_shape: tuple[int, int] = (40, 40),
    *,
    start: str | None = None,
    end: str | None = None,
    imputer: CoordinateQueryImputer | None = None,
    dt_s: float = 3600.0,
    dx_m: float = 1000.0,
    dy_m: float = 1000.0,
    seed: int = 0,
) -> GriddedWindField:
    """Real New Delhi gridded wind field aligned to an Nx x Ny response grid.

    Loads observed station WD/WS-derived transport vectors and maps station
    lon/lat to grid coordinates, then queries the coordinate-query imputer on
    every grid cell. The v1 city-level provider (``real_new_delhi_wind_sequence``)
    remains available for smoke checks.
    """

    from data.pol_weather import load_new_delhi_wind_data

    wind = load_new_delhi_wind_data(data_csv, locations_csv, start=start, end=end)
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    lon = np.asarray(wind.sensors_xy[:, 0], dtype=np.float64)
    lat = np.asarray(wind.sensors_xy[:, 1], dtype=np.float64)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-12)
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-12)
    ix = (lon - np.nanmin(lon)) / lon_scale * (nx - 1)
    iy = (lat - np.nanmin(lat)) / lat_scale * (ny - 1)
    station_coords = np.stack([ix, iy], axis=1)
    station_vectors = np.asarray(wind.observed_vectors, dtype=np.float32)
    station_mask = np.asarray(wind.vector_mask[..., 0], dtype=bool)
    return build_gridded_wind_field(
        station_coords, station_vectors, station_mask, wind.timestamps, grid_shape,
        imputer=imputer, dt_s=dt_s, dx_m=dx_m, dy_m=dy_m, seed=seed,
        provider="gridded_kernel_new_delhi",
        extra_metadata={
            "source_data_csv": str(data_csv),
            "source_locations_csv": str(locations_csv),
            "station_ids": [str(s) for s in wind.station_ids],
        },
    )


__all__ = [
    "DEFAULT_IMPUTED_WIND_PRODUCT",
    "CoordinateQueryImputer",
    "GriddedWindField",
    "KernelCoordinateQueryImputer",
    "WindSequence",
    "ar1_synthetic",
    "build_gridded_wind_field",
    "build_wind_field_ensemble",
    "constant_direction",
    "diurnal_synthetic",
    "evaluate_gridded_wind_heldout",
    "gridded_new_delhi_wind_field",
    "multi_direction_synthetic",
    "real_new_delhi_wind_sequence",
    "single_direction_synthetic",
    "transport_vectors_from_wd_ws",
]
