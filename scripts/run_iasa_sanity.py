#!/usr/bin/env python3
"""Minimal IASA sanity runner for the active pollution inventory path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "sim"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.activity import TemporalBasis  # noqa: E402
from model.iasa.response import (  # noqa: E402
    DispersionConfig,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import constant_direction  # noqa: E402


def _station_grid_indices(sensors_xy: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    lon = sensors_xy[:, 0].astype(np.float64)
    lat = sensors_xy[:, 1].astype(np.float64)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-12)
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-12)
    ix = np.rint((lon - np.nanmin(lon)) / lon_scale * (nx - 1)).astype(np.int64)
    iy = np.rint((lat - np.nanmin(lat)) / lat_scale * (ny - 1)).astype(np.int64)
    return np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)


def run_task3a_sanity(*, start: str, end: str) -> dict[str, Any]:
    from data.pol_weather import load_new_delhi_wind_data
    from model.iasa.activity import build_default_activity_profile, combine_inventory_sources
    from model.iasa.wind import real_new_delhi_wind_sequence
    import sim.polsim as polsim

    inventory = polsim.load_pol_source_inventory(src_dir=SIM_DIR)
    grid = polsim.make_grid(Nx=40, Ny=40, src_dir=str(SIM_DIR), load_inventory=True)
    wind = load_new_delhi_wind_data(
        SIM_DIR / "govdata_1H_current.csv",
        SIM_DIR / "govdata_locations.csv",
        start=start,
        end=end,
    )
    wind_sequence = real_new_delhi_wind_sequence(
        SIM_DIR / "govdata_1H_current.csv",
        SIM_DIR / "govdata_locations.csv",
        start=start,
        end=end,
        allow_observed_fallback=True,
    )

    if grid.source_names != inventory.source_names:
        raise RuntimeError("Grid and inventory source names disagree.")
    if grid.source_maps is None or grid.source_matrix is None or grid.source_metadata is None:
        raise RuntimeError("Grid inventory fields were not populated.")
    if inventory.raw_metadata.get("normalization") != "per_source_cropped_p99":
        raise RuntimeError("Expected per-source p99 source normalization.")
    if wind.observed_vectors.shape[:2] != (len(wind.station_ids), len(wind.timestamps)):
        raise RuntimeError("Wind vector shape does not match station/time axes.")
    if wind_sequence.vx.shape != (len(wind.timestamps),) or wind_sequence.vy.shape != (len(wind.timestamps),):
        raise RuntimeError("WindSequence must expose city-level vx/vy with shape [T].")

    activity = build_default_activity_profile(inventory.source_names, wind.timestamps)
    if activity.theta.shape != (len(wind.timestamps), len(inventory.source_names)):
        raise RuntimeError("Activity matrix has unexpected shape.")
    if np.any(activity.theta < 0):
        raise RuntimeError("Activity matrix must be nonnegative.")

    source_terms = combine_inventory_sources(inventory.source_maps, activity.theta)
    if source_terms.shape != (len(wind.timestamps), grid.Nx, grid.Ny):
        raise RuntimeError("Inventory activity source terms have unexpected shape.")
    if not np.isfinite(source_terms).all():
        raise RuntimeError("Inventory activity source terms contain non-finite values.")

    ix, iy = _station_grid_indices(wind.sensors_xy, grid.Nx, grid.Ny)
    sensor_source_terms = source_terms[:, ix, iy].T
    pm25_mask_count = int(np.count_nonzero(wind.raw_pm25_mask))
    wind_mask_count = int(np.count_nonzero(wind.vector_mask[..., 0]))
    if pm25_mask_count == 0:
        raise RuntimeError("Sanity window has no observed pm25 values.")
    if wind_mask_count == 0:
        raise RuntimeError("Sanity window has no observed wind vectors.")
    if not np.isfinite(sensor_source_terms).all():
        raise RuntimeError("Sensor source-term matrix contains non-finite values.")

    return {
        "status": "ok",
        "gate": "task3a_minimal_iasa",
        "source_names": inventory.source_names,
        "source_maps_shape": list(inventory.source_maps.shape),
        "source_matrix_shape": list(inventory.source_matrix.shape),
        "activity_shape": list(activity.theta.shape),
        "activity_metadata": activity.metadata,
        "wind_provider": wind_sequence.provider,
        "wind_metadata": wind_sequence.metadata,
        "source_terms_shape": list(source_terms.shape),
        "sensor_source_terms_shape": list(sensor_source_terms.shape),
        "station_count": int(len(wind.station_ids)),
        "timestamp_count": int(len(wind.timestamps)),
        "observed_pm25_count": pm25_mask_count,
        "observed_wind_vector_count": wind_mask_count,
        "normalization": inventory.raw_metadata["normalization"],
        "all_zero_sources": inventory.raw_metadata["all_zero_sources"],
    }


def _compact_source(nx: int, ny: int, center: tuple[float, float], sigma: float = 0.6) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    src = np.exp(-0.5 * (dx * dx + dy * dy) / float(sigma * sigma)).astype(np.float32)
    src[src < 0.05] = 0.0
    return src / max(float(src.max()), 1e-12)


def _response_sanity_inputs() -> tuple[np.ndarray, list[str], TemporalBasis, Observer, ResponseConfig, DispersionConfig]:
    nx = ny = 16
    source_names = ["west_source", "east_edge_source", "south_source", "interior_source"]
    source_maps = np.stack(
        [
            _compact_source(nx, ny, (3.0, 8.0)),
            _compact_source(nx, ny, (14.0, 8.0)),
            _compact_source(nx, ny, (8.0, 3.0)),
            _compact_source(nx, ny, (8.0, 8.0)),
        ],
        axis=0,
    )
    T = 60
    basis_values = np.zeros((T, 2), dtype=np.float32)
    basis_values[2, 0] = 1.0
    basis_values[:, 1] = 1.0
    basis = TemporalBasis(
        names=["impulse_t2", "constant"],
        values=basis_values,
        metadata={"gate": "response", "purpose": "impulse_and_constant_coverage"},
    )
    observer = Observer(
        sensor_ids=["west_sensor", "east_sensor", "north_sensor", "south_sensor"],
        sensor_xy=np.asarray([[1.0, 8.0], [12.0, 8.0], [8.0, 14.0], [8.0, 3.0]], dtype=np.float32),
    )
    response_config = ResponseConfig(dt=1.0, lag_window_steps=12, substep_dt=0.25, kernel_truncation_radius=3.0)
    dispersion_config = DispersionConfig(sigma_parallel=0.7, sigma_perp=0.25, min_dispersion_time=0.25)
    return source_maps, source_names, basis, observer, response_config, dispersion_config


def run_response_gate() -> dict[str, Any]:
    source_maps, source_names, basis, observer, response_config, dispersion_config = _response_sanity_inputs()
    wind = constant_direction(length=basis.values.shape[0], vx=1.0, vy=0.0)
    result = build_lagged_response_matrix(
        source_maps,
        source_names,
        basis,
        observer,
        wind,
        response_config=response_config,
        dispersion_config=dispersion_config,
    )
    H = result.H_lag
    expected_shape = (len(observer.sensor_ids) * basis.values.shape[0], len(source_names) * len(basis.names))
    if H.shape != expected_shape:
        raise RuntimeError(f"response H_lag shape {H.shape}; expected {expected_shape}")
    if not np.isfinite(H).all() or not np.any(H > 0):
        raise RuntimeError("response H_lag must be finite and nonzero")
    west_impulse_col = 0
    east_edge_impulse_col = 2
    interior_impulse_col = 6
    M = len(observer.sensor_ids)
    release_t = 2
    downwind_slice = slice((release_t + 5) * M, (release_t + 9) * M)
    west_rows = H[downwind_slice, west_impulse_col].reshape(-1, M)
    if float(west_rows[:, 1].max()) <= float(west_rows[:, 0].max()):
        raise RuntimeError("eastward impulse should produce stronger positive-age downwind than upwind response")
    exit_count = result.metadata["exit_count_by_column"][east_edge_impulse_col]
    if exit_count <= 0:
        raise RuntimeError("east-edge source should record at least one open-boundary exit")
    late_east_edge = H[(release_t + 8) * M:, east_edge_impulse_col]
    if float(late_east_edge.max()) > 1e-5:
        raise RuntimeError("east-edge impulse should stop contributing after exit")

    retained = result.metadata["kernel_mass_retained_by_column"]
    dropped = result.metadata["dropped_mass_by_column"]
    if retained[east_edge_impulse_col] >= retained[interior_impulse_col]:
        raise RuntimeError("boundary impulse must retain less mass than the matched interior impulse")
    if dropped[east_edge_impulse_col] <= 0.0:
        raise RuntimeError("boundary impulse must report positive dropped kernel mass")

    required_metadata = {
        "boundary_mode",
        "response_implementation",
        "response_config",
        "dispersion_config",
        "wind_provider",
        "wind_metadata",
        "wind_vx",
        "wind_vy",
        "row_index",
        "column_index",
        "baseline_policy",
        "baseline",
        "kernel_mass_retained_by_column",
        "dropped_mass_by_column",
        "exit_count_by_column",
        "released_mass_exited_by_column",
        "first_exit_by_release",
        "kernel_mass_summaries",
    }
    missing_metadata = sorted(required_metadata.difference(result.metadata))
    if missing_metadata:
        raise RuntimeError(f"response metadata is missing required provenance keys: {missing_metadata}")
    if result.metadata["row_index"] != result.row_index:
        raise RuntimeError("metadata row_index must match ResponseMatrixResult.row_index")
    if result.metadata["column_index"] != result.column_index:
        raise RuntimeError("metadata column_index must match ResponseMatrixResult.column_index")
    if len(result.metadata["baseline"]) != H.shape[0] or np.any(np.asarray(result.metadata["baseline"]) != 0.0):
        raise RuntimeError("zero-source baseline metadata must be a zero vector aligned to H_lag rows")
    if len(result.metadata["wind_vx"]) != basis.values.shape[0] or len(result.metadata["wind_vy"]) != basis.values.shape[0]:
        raise RuntimeError("metadata must preserve the exact wind arrays")

    iso = build_lagged_response_matrix(
        source_maps,
        source_names,
        basis,
        observer,
        wind,
        response_config=response_config,
        dispersion_config=DispersionConfig(sigma_parallel=0.4, sigma_perp=0.4, min_dispersion_time=0.25),
    )
    if np.allclose(iso.H_lag, H):
        raise RuntimeError("anisotropic and isotropic dispersion should produce different response matrices")

    repeat = build_lagged_response_matrix(
        source_maps,
        source_names,
        basis,
        observer,
        wind,
        response_config=response_config,
        dispersion_config=dispersion_config,
    )
    np.testing.assert_allclose(repeat.H_lag, H, rtol=0.0, atol=0.0)

    return {
        "status": "ok",
        "gate": "response",
        "response_implementation": result.metadata["response_implementation"],
        "boundary_mode": result.metadata["boundary_mode"],
        "H_lag_shape": list(H.shape),
        "column_count": len(result.column_index),
        "row_count": len(result.row_index),
        "east_edge_exit_count": int(exit_count),
        "east_edge_retained_mass": float(retained[east_edge_impulse_col]),
        "interior_retained_mass": float(retained[interior_impulse_col]),
        "metadata_keys_checked": sorted(required_metadata),
        "kernel_mass_summary_count": len(result.metadata["kernel_mass_summaries"]),
    }


def run_sanity(*, start: str, end: str) -> dict[str, Any]:
    return run_task3a_sanity(start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=("task3a", "response", "projection", "diagnostics", "fit", "merge", "all"),
        default="task3a",
    )
    parser.add_argument("--strict-all", action="store_true")
    parser.add_argument("--start", default="2018-05-01 00:00:00+05:30")
    parser.add_argument("--end", default="2018-05-01 23:00:00+05:30")
    args = parser.parse_args()
    if args.gate == "task3a":
        result = run_task3a_sanity(start=args.start, end=args.end)
    elif args.gate == "response":
        result = run_response_gate()
    elif args.gate == "all":
        result = {
            "status": "ok",
            "gate": "all",
            "task3a": run_task3a_sanity(start=args.start, end=args.end),
            "response": run_response_gate(),
            "skipped_gates": ["projection", "diagnostics", "fit", "merge"],
        }
        if args.strict_all:
            raise NotImplementedError("Strict all-gate mode requires Tasks 6-9 gates.")
    else:
        raise NotImplementedError(f"Gate {args.gate!r} is implemented by a later roadmap task.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
