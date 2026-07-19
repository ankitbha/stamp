#!/usr/bin/env python3
"""Minimal IASA sanity runner for the active pollution inventory path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The open-boundary response builder issues many tiny per-kernel tensor ops in a
# Python loop; default CPU multithreading thrashes on them. Pin to one thread for
# the sanity/parity harness so builds stay fast and do not trip node CPU-time
# guards. This is a harness-level choice and does not constrain library callers.
torch.set_num_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "sim"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.activity import TemporalBasis  # noqa: E402
from model.iasa.backend import to_numpy  # noqa: E402
from model.iasa.background import BackgroundBasisConfig, build_background_basis  # noqa: E402
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_identifiability  # noqa: E402
from model.iasa.projection import fit_background_projector, project_response_and_observations  # noqa: E402
from model.iasa.response import (  # noqa: E402
    DispersionConfig,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import WindSequence, constant_direction  # noqa: E402


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


def _build_response_sanity_result() -> tuple[Any, np.ndarray, Observer, TemporalBasis]:
    source_maps, source_names, basis, observer, response_config, dispersion_config = _response_sanity_inputs()
    wind = constant_direction(length=basis.values.shape[0], vx=1.0, vy=0.0)
    result = build_lagged_response_matrix(
        source_maps, source_names, basis, observer, wind,
        response_config=response_config, dispersion_config=dispersion_config,
    )
    timestamps = np.datetime64("2026-06-01T00:00") + np.arange(basis.values.shape[0]) * np.timedelta64(1, "h")
    return result, timestamps, observer, basis


def _dense_grid_observer(nx: int, ny: int) -> Observer:
    xy = np.asarray([(float(x), float(y)) for x in range(nx) for y in range(ny)], dtype=np.float32)
    ids = [f"cell_{x}_{y}" for x in range(nx) for y in range(ny)]
    return Observer(sensor_ids=ids, sensor_xy=xy)


def _response_moments(values: np.ndarray, sensor_xy: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(values, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("dispersion moment response has zero mass")
    xy = np.asarray(sensor_xy, dtype=np.float64)
    mean = (weights[:, None] * xy).sum(axis=0) / total
    centered = xy - mean
    moments = (weights[:, None] * centered ** 2).sum(axis=0) / total
    return float(moments[0]), float(moments[1])


def _moment_case(sigma_parallel: float, sigma_perp: float) -> tuple[float, float]:
    nx = ny = 16
    T = 8
    source = np.zeros((1, nx, ny), dtype=np.float32)
    source[0, 8, 8] = 1.0
    impulse = np.zeros((T, 1), dtype=np.float32)
    impulse[0, 0] = 1.0
    basis = TemporalBasis(names=["impulse_t0"], values=impulse, metadata={"purpose": "dispersion_moment"})
    dense = _dense_grid_observer(nx, ny)
    result = build_lagged_response_matrix(
        source,
        ["interior_single_cell"],
        basis,
        dense,
        constant_direction(length=T, vx=0.5, vy=0.0),
        response_config=ResponseConfig(
            dt=1.0,
            lag_window_steps=8,
            substep_dt=0.25,
            kernel_truncation_radius=4.0,
            max_kernel_diagnostic_records=20,
        ),
        dispersion_config=DispersionConfig(
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            min_dispersion_time=0.25,
        ),
    )
    age_index = 3
    M = len(dense.sensor_ids)
    values = result.H_lag[age_index * M:(age_index + 1) * M, 0]
    return _response_moments(values, dense.sensor_xy)


def run_response_gate() -> dict[str, Any]:
    source_maps, source_names, basis, observer, response_config, dispersion_config = _response_sanity_inputs()
    wind = constant_direction(length=basis.values.shape[0], vx=1.0, vy=0.0)
    result, _, _, _ = _build_response_sanity_result()
    H = to_numpy(result.H_lag)
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
        "kernel_emitted_mass_by_column",
        "kernel_observation_count_by_column",
        "kernel_diagnostic_total_count",
        "kernel_diagnostic_stored_count",
        "kernel_diagnostics_truncated",
        "kernel_quadrature_clip_count_by_column",
        "max_raw_retained_fraction_by_column",
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
    emitted = np.asarray(result.metadata["kernel_emitted_mass_by_column"], dtype=np.float64)
    retained_arr = np.asarray(result.metadata["kernel_mass_retained_by_column"], dtype=np.float64)
    dropped_arr = np.asarray(result.metadata["dropped_mass_by_column"], dtype=np.float64)
    conservation_error = float(np.max(np.abs(retained_arr + dropped_arr - emitted)))
    if conservation_error > 1e-5:
        raise RuntimeError(f"kernel mass conservation error {conservation_error:.3e} exceeds 1e-5")
    if result.metadata["kernel_diagnostic_total_count"] <= result.metadata["kernel_diagnostic_stored_count"]:
        raise RuntimeError("default response sanity should exercise explicit diagnostic truncation")
    if not result.metadata["kernel_diagnostics_truncated"]:
        raise RuntimeError("diagnostic truncation flag must be true when total count exceeds stored count")

    impulse_only = TemporalBasis(
        names=["impulse_t2"],
        values=basis.values[:, :1],
        metadata={"gate": "response", "purpose": "directional_arrival"},
    )
    north_wind = constant_direction(length=basis.values.shape[0], vx=0.0, vy=1.0)
    north = build_lagged_response_matrix(
        source_maps[2:3],
        ["south_source"],
        impulse_only,
        observer,
        north_wind,
        response_config=response_config,
        dispersion_config=dispersion_config,
    )
    north_rows = to_numpy(north.H_lag)[(release_t + 8) * M:(release_t + 12) * M, 0].reshape(-1, M)
    north_peak = float(north_rows[:, 2].max())
    south_late_peak = float(north_rows[:, 3].max())
    if north_peak <= south_late_peak:
        raise RuntimeError("northward south-source impulse should reach north sensor more strongly at positive ages")

    constant_basis = TemporalBasis(
        names=["constant"],
        values=np.ones((basis.values.shape[0], 1), dtype=np.float32),
        metadata={"gate": "response", "purpose": "wind_diversity"},
    )
    two_vx = np.ones(basis.values.shape[0], dtype=np.float32)
    two_vy = np.zeros_like(two_vx)
    two_vx[30:] = 0.0
    two_vy[30:] = 1.0
    two_direction = WindSequence(
        timestamps=wind.timestamps,
        vx=two_vx,
        vy=two_vy,
        provider="two_direction_synthetic",
        metadata={"switch_time_index": 30},
    )
    east_constant = build_lagged_response_matrix(
        source_maps[3:4], ["interior_source"], constant_basis, observer, wind,
        response_config=response_config, dispersion_config=dispersion_config,
    )
    two_constant = build_lagged_response_matrix(
        source_maps[3:4], ["interior_source"], constant_basis, observer, two_direction,
        response_config=response_config, dispersion_config=dispersion_config,
    )
    two_direction_max_difference = float(np.max(np.abs(to_numpy(two_constant.H_lag) - to_numpy(east_constant.H_lag))))
    if two_direction_max_difference <= 1e-6:
        raise RuntimeError("two-direction fingerprint must differ from eastward-only fingerprint")

    anisotropic_moments = _moment_case(0.8, 0.2)
    larger_parallel_moments = _moment_case(1.1, 0.2)
    larger_perp_moments = _moment_case(0.8, 0.5)
    isotropic_moments = _moment_case(0.4, 0.4)
    anisotropy_ratio = anisotropic_moments[0] / max(anisotropic_moments[1], 1e-12)
    if anisotropy_ratio <= 1.25:
        raise RuntimeError("anisotropic along-wind/crosswind moment ratio must exceed 1.25")
    parallel_increase = larger_parallel_moments[0] - anisotropic_moments[0]
    parallel_cross_change = abs(larger_parallel_moments[1] - anisotropic_moments[1])
    if parallel_increase <= parallel_cross_change:
        raise RuntimeError("increasing sigma_parallel must primarily increase the along-wind moment")
    if larger_perp_moments[1] <= anisotropic_moments[1]:
        raise RuntimeError("increasing sigma_perp must increase the crosswind moment")
    isotropic_relative_difference = abs(isotropic_moments[0] - isotropic_moments[1]) / max(isotropic_moments)
    if isotropic_relative_difference > 0.25:
        raise RuntimeError("isotropic dispersion moments must agree within 25 percent")

    iso = build_lagged_response_matrix(
        source_maps,
        source_names,
        basis,
        observer,
        wind,
        response_config=response_config,
        dispersion_config=DispersionConfig(sigma_parallel=0.4, sigma_perp=0.4, min_dispersion_time=0.25),
    )
    if np.allclose(to_numpy(iso.H_lag), H):
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
    np.testing.assert_allclose(to_numpy(repeat.H_lag), H, rtol=0.0, atol=0.0)

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
        "kernel_mass_conservation_error": conservation_error,
        "kernel_diagnostic_total_count": result.metadata["kernel_diagnostic_total_count"],
        "kernel_diagnostic_stored_count": result.metadata["kernel_diagnostic_stored_count"],
        "kernel_diagnostics_truncated": result.metadata["kernel_diagnostics_truncated"],
        "northward_north_sensor_peak": north_peak,
        "northward_south_sensor_late_peak": south_late_peak,
        "two_direction_max_difference": two_direction_max_difference,
        "anisotropic_moments": list(anisotropic_moments),
        "larger_parallel_moments": list(larger_parallel_moments),
        "larger_perp_moments": list(larger_perp_moments),
        "isotropic_moments": list(isotropic_moments),
        "anisotropy_ratio": anisotropy_ratio,
        "isotropic_relative_difference": isotropic_relative_difference,
        "metadata_keys_checked": sorted(required_metadata),
        "kernel_mass_summary_count": len(result.metadata["kernel_mass_summaries"]),
    }


def run_projection_gate() -> dict[str, Any]:
    response, timestamps, observer, _ = _build_response_sanity_result()
    expected_columns = [
        (source, basis)
        for source in ("west_source", "east_edge_source", "south_source", "interior_source")
        for basis in ("impulse_t2", "constant")
    ]
    actual_columns = [(item["source_name"], item["basis_name"]) for item in response.column_index]
    if actual_columns != expected_columns:
        raise RuntimeError("Task 5 response column order changed")

    normal_config = BackgroundBasisConfig(
        include_constant=True, temporal_polynomial_degree=1, daily_harmonics=1,
        max_background_rank=8, basis_mode="normal",
    )
    normal = build_background_basis(response.row_index, timestamps, observer.sensor_xy, normal_config)
    if normal.column_names != ["constant", "time_polynomial_1", "daily_sin_1", "daily_cos_1"]:
        raise RuntimeError("normal Gate S2 background columns changed")
    H_lag_np = to_numpy(response.H_lag).astype(np.float64)
    normal_Q_np = to_numpy(normal.Q)
    c_true = np.asarray([1.0, 0.5, 0.0, 0.25, 0.75, 0.0, 0.4, 0.2], dtype=np.float64)
    beta = np.asarray([0.3, -0.1, 0.2, 0.15], dtype=np.float64)
    Y = H_lag_np @ c_true + normal_Q_np @ beta
    projected = project_response_and_observations(
        response.H_lag, Y, normal, response.row_index, response.column_index,
    )
    projector = fit_background_projector(normal)
    U_r_np = to_numpy(projector.U_r)

    empty = build_background_basis(
        response.row_index, timestamps, observer.sensor_xy,
        BackgroundBasisConfig(include_constant=False),
    )
    empty_projection = project_response_and_observations(
        response.H_lag, Y, empty, response.row_index, response.column_index,
    )
    np.testing.assert_allclose(to_numpy(empty_projection.H_tilde), H_lag_np, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(to_numpy(empty_projection.Y_tilde), Y, rtol=0.0, atol=1e-12)

    projected_H_tilde = to_numpy(projected.H_tilde)
    projected_Y_tilde = to_numpy(projected.Y_tilde)
    expected_y_tilde = projected_H_tilde @ c_true
    np.testing.assert_allclose(projected_Y_tilde, expected_y_tilde, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(to_numpy(projected.H_removed) + projected_H_tilde, H_lag_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(to_numpy(projected.Y_removed) + projected_Y_tilde, Y, rtol=1e-10, atol=1e-12)
    reconstructed = H_lag_np - U_r_np @ (U_r_np.T @ H_lag_np)
    np.testing.assert_allclose(reconstructed, projected_H_tilde, rtol=0.0, atol=1e-12)
    if projector.effective_rank != 4:
        raise RuntimeError("normal Gate S2 background must have effective rank four")

    redundant = build_background_basis(
        response.row_index, timestamps, observer.sensor_xy,
        BackgroundBasisConfig(include_constant=False, basis_mode="stress"),
        user_basis=np.column_stack([normal_Q_np, normal_Q_np[:, 0]]),
        user_basis_names=normal.column_names + ["constant_duplicate"],
    )
    redundant_projection = project_response_and_observations(
        response.H_lag, Y, redundant, response.row_index, response.column_index,
    )
    if redundant_projection.metadata["effective_rank"] != 4:
        raise RuntimeError("redundant background must retain effective rank four")
    np.testing.assert_allclose(to_numpy(redundant_projection.H_tilde), projected_H_tilde, rtol=1e-10, atol=1e-12)

    stress_column_index = 7
    source_like = H_lag_np[:, stress_column_index].copy()
    source_like /= max(float(np.linalg.norm(source_like)), np.finfo(np.float64).eps)
    stress = build_background_basis(
        response.row_index, timestamps, observer.sensor_xy,
        BackgroundBasisConfig(include_constant=False, basis_mode="stress"),
        user_basis=np.column_stack([normal_Q_np, source_like]),
        user_basis_names=normal.column_names + ["stress_interior_source_constant"],
    )
    stress_projection = project_response_and_observations(
        response.H_lag, Y, stress, response.row_index, response.column_index,
    )
    normal_visibility = projected.metadata["H_visibility_ratio_by_column"]
    normal_absorption = projected.metadata["H_absorption_ratio_by_column"]
    stress_visibility = stress_projection.metadata["H_visibility_ratio_by_column"]
    stress_absorption = stress_projection.metadata["H_absorption_ratio_by_column"]
    if min(normal_visibility) < 0.8 or max(normal_absorption) > 0.6:
        raise RuntimeError("normal background removes too much source response")
    if stress_visibility[stress_column_index] >= normal_visibility[stress_column_index]:
        raise RuntimeError("stress source-like background must lower target visibility")
    if stress_absorption[stress_column_index] <= normal_absorption[stress_column_index]:
        raise RuntimeError("stress source-like background must increase target absorption")
    if projected.metadata["H_orthogonality_residual"] > 1e-6:
        raise RuntimeError("projected response is not orthogonal to the background")
    if projected.metadata["Y_orthogonality_residual"] > 1e-6:
        raise RuntimeError("projected observations are not orthogonal to the background")
    if projected.metadata["idempotence_residual"] > 1e-8:
        raise RuntimeError("background projection is not idempotent")

    return {
        "status": "ok",
        "gate": "projection",
        "H_lag_shape": list(response.H_lag.shape),
        "H_tilde_shape": list(projected.H_tilde.shape),
        "column_mapping": actual_columns,
        "c_true": c_true.tolist(),
        "beta": beta.tolist(),
        "normal_background_columns": normal.column_names,
        "normal_effective_rank": projector.effective_rank,
        "redundant_effective_rank": redundant_projection.metadata["effective_rank"],
        "stress_effective_rank": stress_projection.metadata["effective_rank"],
        "H_orthogonality_residual": projected.metadata["H_orthogonality_residual"],
        "Y_orthogonality_residual": projected.metadata["Y_orthogonality_residual"],
        "idempotence_residual": projected.metadata["idempotence_residual"],
        "normal_visibility_ratio_by_column": normal_visibility,
        "normal_absorption_ratio_by_column": normal_absorption,
        "stress_visibility_ratio_by_column": stress_visibility,
        "stress_absorption_ratio_by_column": stress_absorption,
        "stress_label": "stress_interior_source_constant",
        "stress_target_column": stress_column_index,
        "empty_basis_no_op": True,
        "exact_background_removal": True,
        "saved_U_r_reconstruction": True,
    }


def _build_response_on_device(device: str):
    source_maps, source_names, basis, observer, response_config, dispersion_config = _response_sanity_inputs()
    wind = constant_direction(length=basis.values.shape[0], vx=1.0, vy=0.0)
    from dataclasses import replace

    result = build_lagged_response_matrix(
        source_maps, source_names, basis, observer, wind,
        response_config=replace(response_config, device=device),
        dispersion_config=dispersion_config,
    )
    timestamps = np.datetime64("2026-06-01T00:00") + np.arange(basis.values.shape[0]) * np.timedelta64(1, "h")
    return result, timestamps, observer


def _project_on_device(response, timestamps, observer, device: str):
    normal_config = BackgroundBasisConfig(
        include_constant=True, temporal_polynomial_degree=1, daily_harmonics=1,
        max_background_rank=8, basis_mode="normal", device=device,
    )
    normal = build_background_basis(response.row_index, timestamps, observer.sensor_xy, normal_config)
    c_true = np.asarray([1.0, 0.5, 0.0, 0.25, 0.75, 0.0, 0.4, 0.2], dtype=np.float64)
    beta = np.asarray([0.3, -0.1, 0.2, 0.15], dtype=np.float64)
    Y = to_numpy(response.H_lag).astype(np.float64) @ c_true + to_numpy(normal.Q) @ beta
    return project_response_and_observations(
        response.H_lag, Y, normal, response.row_index, response.column_index,
    )


def run_parity_gate() -> dict[str, Any]:
    """CPU/CUDA parity summary for Gate S1 (response) and Gate S2 (projection)."""
    import torch

    cpu_response, timestamps, observer = _build_response_on_device("cpu")
    cpu_projection = _project_on_device(cpu_response, timestamps, observer, "cpu")
    if cpu_response.H_lag.device.type != "cpu" or cpu_projection.H_tilde.device.type != "cpu":
        raise RuntimeError("CPU outputs must remain on the requested cpu device")

    summary: dict[str, Any] = {
        "status": "ok",
        "gate": "parity",
        "cuda_available": bool(torch.cuda.is_available()),
        "response_dtype": cpu_response.metadata["response_dtype"],
        "projection_dtype": cpu_projection.metadata["dtype"],
        "torch_version": cpu_response.metadata["torch_version"],
        "cuda_version": cpu_response.metadata["cuda_version"],
    }
    if not torch.cuda.is_available():
        summary["status"] = "cpu_only"
        summary["note"] = "CUDA unavailable on this node; run the parity gate on a SLURM GPU allocation."
        return summary

    cuda_response, _, _ = _build_response_on_device("cuda")
    if cuda_response.H_lag.device.type != "cuda":
        raise RuntimeError("CUDA response must remain on the requested cuda device")

    # Response parity compares the float32 response built independently on each
    # device (relative tolerance appropriate to float32 accumulation order).
    response_ref = float(np.max(np.abs(to_numpy(cpu_response.H_lag))))
    response_diff = float(np.max(np.abs(to_numpy(cpu_response.H_lag) - to_numpy(cuda_response.H_lag))))
    response_rel = response_diff / max(response_ref, 1e-12)
    response_tol = 1e-4

    # Projection parity is isolated by projecting the SAME (CPU-built) response on
    # each device, so the comparison reflects only the float64 background SVD /
    # projection backend, not the upstream float32 response divergence.
    cpu_only_projection = _project_on_device(cpu_response, timestamps, observer, "cpu")
    cuda_only_projection = _project_on_device(cpu_response, timestamps, observer, "cuda")
    if cuda_only_projection.H_tilde.device.type != "cuda":
        raise RuntimeError("CUDA projection must remain on the requested cuda device")
    projection_ref = float(np.max(np.abs(to_numpy(cpu_only_projection.H_tilde))))
    projection_diff = float(np.max(np.abs(
        to_numpy(cpu_only_projection.H_tilde) - to_numpy(cuda_only_projection.H_tilde)
    )))
    projection_rel = projection_diff / max(projection_ref, 1e-12)
    projection_tol = 1e-6
    if response_rel > response_tol:
        raise RuntimeError(f"response CPU/CUDA relative parity {response_rel:.3e} exceeds {response_tol:.1e}")
    if projection_rel > projection_tol:
        raise RuntimeError(f"projection CPU/CUDA relative parity {projection_rel:.3e} exceeds {projection_tol:.1e}")
    summary.update({
        "response_cpu_cuda_max_abs_diff": response_diff,
        "response_cpu_cuda_relative_diff": response_rel,
        "projection_cpu_cuda_max_abs_diff": projection_diff,
        "projection_cpu_cuda_relative_diff": projection_rel,
        "response_relative_tolerance": response_tol,
        "projection_relative_tolerance": projection_tol,
    })
    return summary


def _column_index(source_names: list[str], basis_names: list[str]) -> list[dict[str, Any]]:
    cols = []
    for k, source in enumerate(source_names):
        for b, basis in enumerate(basis_names):
            cols.append({"source_index": k, "source_name": source, "basis_index": b, "basis_name": basis})
    return cols


def _max_eligible_coherence(diag) -> float:
    coh = to_numpy(diag.coherence)
    best = 0.0
    J = coh.shape[0]
    for i in range(J):
        for j in range(i + 1, J):
            v = coh[i, j]
            if not np.isnan(v) and v > best:
                best = float(v)
    return best


def _matched_wind_diagnostics() -> dict[str, Any]:
    # Two compact sources on the same y-line, offset in x. Under pure eastward
    # advection both stream along y=8 to the east sensor with nearly proportional
    # fingerprints (high coherence). A two-direction regime (east then north)
    # separates them, so sigma_J must not drop and coherence must not rise.
    nx = ny = 16
    source_maps = np.stack(
        [_compact_source(nx, ny, (5.0, 8.0)), _compact_source(nx, ny, (8.0, 8.0))], axis=0
    )
    source_names = ["source_a", "source_b"]
    T = 40
    basis = TemporalBasis(
        names=["constant"], values=np.ones((T, 1), dtype=np.float32),
        metadata={"gate": "diagnostics", "purpose": "matched_wind"},
    )
    observer = Observer(
        sensor_ids=["west_sensor", "east_sensor", "north_sensor"],
        sensor_xy=np.asarray([[1.0, 8.0], [13.0, 8.0], [8.0, 13.0]], dtype=np.float32),
    )
    response_config = ResponseConfig(dt=1.0, lag_window_steps=12, substep_dt=0.25, kernel_truncation_radius=3.0)
    dispersion_config = DispersionConfig(sigma_parallel=0.7, sigma_perp=0.25, min_dispersion_time=0.25)
    col_index = _column_index(source_names, list(basis.names))

    eastward = constant_direction(length=T, vx=1.0, vy=0.0)
    east_vx = np.ones(T, dtype=np.float32)
    east_vy = np.zeros(T, dtype=np.float32)
    east_vx[T // 2:] = 0.0
    east_vy[T // 2:] = 1.0
    two_direction = WindSequence(
        timestamps=eastward.timestamps, vx=east_vx, vy=east_vy,
        provider="two_direction_synthetic", metadata={"switch_time_index": T // 2},
    )

    def _diag(wind):
        response = build_lagged_response_matrix(
            source_maps, source_names, basis, observer, wind,
            response_config=response_config, dispersion_config=dispersion_config,
        )
        return diagnose_identifiability(response.H_lag, col_index, config=DiagnosticsConfig())

    eastward_diag = _diag(eastward)
    two_direction_diag = _diag(two_direction)
    return {
        "eastward_sigma_J": eastward_diag.sigma_J,
        "two_direction_sigma_J": two_direction_diag.sigma_J,
        "eastward_max_coherence": _max_eligible_coherence(eastward_diag),
        "two_direction_max_coherence": _max_eligible_coherence(two_direction_diag),
    }


def run_diagnostics_gate() -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = torch.float64

    # orthogonal_case: near-orthogonal columns -> full rank, low coherence.
    ortho = torch.eye(6, 3, dtype=dtype, device=device)
    ortho = ortho + 1e-3 * torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype, device=device
    )
    ortho_cols = _column_index(["s0", "s1", "s2"], ["c"])
    ortho_diag = diagnose_identifiability(ortho, ortho_cols, config=DiagnosticsConfig())
    ortho_max_coh = _max_eligible_coherence(ortho_diag)
    if ortho_diag.numerical_rank != 3:
        raise RuntimeError("orthogonal_case must be full numerical rank")
    if ortho_max_coh > 0.1:
        raise RuntimeError(f"orthogonal_case max coherence {ortho_max_coh:.3f} exceeds 0.1")
    if ortho_diag.condition_status != "finite":
        raise RuntimeError("orthogonal_case must have finite condition number")

    # duplicate_case: column 1 equals column 0 exactly.
    base = torch.tensor(
        [[1.0, 0.0, 0.2], [0.5, 0.0, 0.9], [0.2, 0.0, 0.1], [0.0, 0.0, 0.7], [0.3, 0.0, 0.4], [0.8, 0.0, 0.6]],
        dtype=dtype, device=device,
    )
    base[:, 1] = base[:, 0]
    dup_cols = _column_index(["dup_a", "dup_b", "other"], ["c"])
    dup_diag = diagnose_identifiability(base, dup_cols, config=DiagnosticsConfig())
    dup_pair_coh = float(to_numpy(dup_diag.coherence)[0, 1])
    if dup_pair_coh < 0.999:
        raise RuntimeError(f"duplicate_case pair coherence {dup_pair_coh:.5f} below 0.999")
    if dup_diag.sigma_J > 1e-8:
        raise RuntimeError(f"duplicate_case sigma_J {dup_diag.sigma_J:.3e} should be ~0")
    if dup_diag.condition_status != "infinite":
        raise RuntimeError("duplicate_case condition status must be infinite")
    if dup_diag.numerical_rank >= 3:
        raise RuntimeError("duplicate_case must be rank deficient")

    # weak_case: one near-zero column.
    weak = torch.tensor(
        [[1.0, 0.0, 0.2], [0.5, 0.0, 0.9], [0.2, 0.0, 0.1], [0.0, 0.0, 0.7], [0.3, 0.0, 0.4], [0.8, 0.0, 0.6]],
        dtype=dtype, device=device,
    )
    weak[:, 1] = 1e-10
    weak_cols = _column_index(["strong_a", "weak_b", "strong_c"], ["c"])
    weak_diag = diagnose_identifiability(weak, weak_cols, config=DiagnosticsConfig(tau_v=1e-6))
    weak_vis = float(to_numpy(weak_diag.visibility)[1])
    if not (weak_vis <= 1e-8 or weak_diag.weak_flags[1]):
        raise RuntimeError("weak_case near-zero column must be flagged weakly visible")
    if 1 not in weak_diag.weak_set:
        raise RuntimeError("weak_case weak_set must contain the near-zero column")
    if not math.isnan(to_numpy(weak_diag.coherence)[0, 1]):
        raise RuntimeError("weak_case pairwise metrics involving the weak column must be null (NaN sentinel)")

    matched = _matched_wind_diagnostics()
    if matched["two_direction_sigma_J"] < matched["eastward_sigma_J"] - 1e-8:
        raise RuntimeError(
            f"wind diversity reduced sigma_J: two_direction {matched['two_direction_sigma_J']:.4g} "
            f"< eastward {matched['eastward_sigma_J']:.4g}"
        )
    if matched["two_direction_max_coherence"] > matched["eastward_max_coherence"] + 1e-8:
        raise RuntimeError(
            f"wind diversity raised max coherence: two_direction {matched['two_direction_max_coherence']:.4g} "
            f"> eastward {matched['eastward_max_coherence']:.4g}"
        )

    return {
        "status": "ok",
        "gate": "diagnostics",
        "orthogonal_numerical_rank": ortho_diag.numerical_rank,
        "orthogonal_max_coherence": ortho_max_coh,
        "orthogonal_condition_status": ortho_diag.condition_status,
        "duplicate_pair_coherence": dup_pair_coh,
        "duplicate_sigma_J": dup_diag.sigma_J,
        "duplicate_condition_status": dup_diag.condition_status,
        "duplicate_numerical_rank": dup_diag.numerical_rank,
        "weak_column_visibility": weak_vis,
        "weak_visibility_flag": bool(weak_diag.weak_flags[1]),
        "weak_set": weak_diag.weak_set,
        "eastward_sigma_J": matched["eastward_sigma_J"],
        "two_direction_sigma_J": matched["two_direction_sigma_J"],
        "eastward_max_coherence": matched["eastward_max_coherence"],
        "two_direction_max_coherence": matched["two_direction_max_coherence"],
        "diagnostics_keys": sorted(dup_diag.to_json_summary().keys()),
    }


def run_sanity(*, start: str, end: str) -> dict[str, Any]:
    return run_task3a_sanity(start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=("task3a", "response", "projection", "parity", "diagnostics", "fit", "merge", "all"),
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
    elif args.gate == "projection":
        result = run_projection_gate()
    elif args.gate == "parity":
        result = run_parity_gate()
    elif args.gate == "diagnostics":
        result = run_diagnostics_gate()
    elif args.gate == "all":
        result = {
            "status": "ok",
            "gate": "all",
            "task3a": run_task3a_sanity(start=args.start, end=args.end),
            "response": run_response_gate(),
            "projection": run_projection_gate(),
            "parity": run_parity_gate(),
            "diagnostics": run_diagnostics_gate(),
            "skipped_gates": ["fit", "merge"],
        }
        if args.strict_all:
            raise NotImplementedError("Strict all-gate mode requires Tasks 7-9 gates.")
    else:
        raise NotImplementedError(f"Gate {args.gate!r} is implemented by a later roadmap task.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
