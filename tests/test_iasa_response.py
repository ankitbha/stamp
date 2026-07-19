from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.activity import TemporalBasis  # noqa: E402
import model.iasa as iasa_package  # noqa: E402
from model.iasa.response import (  # noqa: E402
    BOUNDARY_MODE,
    RESPONSE_IMPLEMENTATION,
    DispersionConfig,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import constant_direction  # noqa: E402


class PositionDependentWind:
    provider = "position_dependent_test"
    metadata = {"kind": "test"}

    def sample(self, t_index: float, position_xy: np.ndarray) -> np.ndarray:
        return np.asarray([0.1 + 0.01 * position_xy[0], 0.0], dtype=np.float32)


class RecordingWind:
    provider = "recording_test"
    metadata = {"kind": "test"}

    def __init__(self, vector: tuple[float, float] = (1.0, 0.0)) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.sample_times: list[float] = []

    def sample(self, t_index: float, position_xy: np.ndarray) -> np.ndarray:
        del position_xy
        self.sample_times.append(float(t_index))
        return self.vector.copy()


class NonfiniteWind:
    provider = "nonfinite_test"
    metadata = {}

    def sample(self, t_index: float, position_xy: np.ndarray) -> np.ndarray:
        del t_index, position_xy
        return np.asarray([np.nan, 0.0], dtype=np.float32)


def impulse_basis(T: int, tau: int = 0) -> TemporalBasis:
    values = np.zeros((T, 1), dtype=np.float32)
    values[tau, 0] = 1.0
    return TemporalBasis(names=["impulse"], values=values, metadata={"tau": tau})


def single_source_map(nx: int = 8, ny: int = 8, x: int = 2, y: int = 4) -> np.ndarray:
    maps = np.zeros((1, nx, ny), dtype=np.float32)
    maps[0, x, y] = 1.0
    return maps


def observer() -> Observer:
    return Observer(sensor_ids=["upwind", "downwind"], sensor_xy=np.asarray([[1.0, 4.0], [5.0, 4.0]], dtype=np.float32))


class IasaResponseTests(unittest.TestCase):
    def test_response_shape_and_metadata(self) -> None:
        T = 10
        result = build_lagged_response_matrix(
            single_source_map(),
            ["src"],
            impulse_basis(T),
            observer(),
            constant_direction(length=T, vx=1.0, vy=0.0),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=5, substep_dt=0.25),
        )
        self.assertEqual(result.H_lag.shape, (2 * T, 1))
        self.assertIsInstance(result.H_lag, torch.Tensor)
        self.assertIsInstance(result.baseline, torch.Tensor)
        self.assertEqual(result.H_lag.device.type, "cpu")
        self.assertEqual(str(result.H_lag.dtype), "torch.float32")
        for key in ("device", "dtype", "response_dtype", "torch_version", "cuda_version",
                    "ensemble_kind", "inventory_hash", "wind_realization_hash"):
            self.assertIn(key, result.metadata)
        self.assertEqual(result.metadata["ensemble_kind"], "transport")
        self.assertEqual(result.metadata["boundary_mode"], BOUNDARY_MODE)
        self.assertEqual(result.metadata["response_implementation"], RESPONSE_IMPLEMENTATION)
        self.assertEqual(result.column_index[0]["source_name"], "src")
        self.assertEqual(result.row_index[1]["sensor_id"], "downwind")
        required_metadata = {
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
        self.assertFalse(required_metadata.difference(result.metadata))
        self.assertEqual(result.metadata["row_index"], result.row_index)
        self.assertEqual(result.metadata["column_index"], result.column_index)
        self.assertEqual(result.metadata["baseline_policy"], "zero_source")
        np.testing.assert_allclose(result.metadata["baseline"], result.baseline)
        np.testing.assert_allclose(result.metadata["wind_vx"], np.ones(T, dtype=np.float32))
        np.testing.assert_allclose(result.metadata["wind_vy"], np.zeros(T, dtype=np.float32))
        self.assertEqual(len(result.metadata["kernel_mass_retained_by_column"]), result.H_lag.shape[1])
        self.assertEqual(len(result.metadata["dropped_mass_by_column"]), result.H_lag.shape[1])
        self.assertEqual(len(result.metadata["exit_count_by_column"]), result.H_lag.shape[1])
        json.dumps(result.metadata)

    def test_same_time_release_uses_min_dispersion_time(self) -> None:
        T = 4
        result = build_lagged_response_matrix(
            single_source_map(x=2, y=4),
            ["src"],
            impulse_basis(T, tau=0),
            Observer(sensor_ids=["source_sensor"], sensor_xy=np.asarray([[2.0, 4.0]], dtype=np.float32)),
            constant_direction(length=T, vx=1.0, vy=0.0),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=3, substep_dt=0.5),
            dispersion_config=DispersionConfig(sigma_parallel=0.7, sigma_perp=0.25, min_dispersion_time=0.25),
        )
        self.assertGreater(float(result.H_lag[0, 0]), 0.0)
        first_kernel = result.metadata["kernel_mass_summaries"][0]
        self.assertEqual(first_kernel["age"], 0.0)
        self.assertEqual(first_kernel["effective_age"], 0.25)
        self.assertLessEqual(first_kernel["retained_fraction"], 1.0)
        self.assertGreaterEqual(first_kernel["retained_fraction"], 0.0)
        self.assertLessEqual(first_kernel["retained_mass"], first_kernel["emitted_mass"])
        self.assertAlmostEqual(
            first_kernel["retained_mass"] + first_kernel["dropped_mass"],
            first_kernel["emitted_mass"],
            places=5,
        )

    def test_open_boundary_exit_stops_contribution_without_wrap(self) -> None:
        T = 12
        result = build_lagged_response_matrix(
            single_source_map(nx=8, ny=8, x=7, y=4),
            ["edge"],
            impulse_basis(T, tau=0),
            Observer(sensor_ids=["west", "east"], sensor_xy=np.asarray([[0.0, 4.0], [7.0, 4.0]], dtype=np.float32)),
            constant_direction(length=T, vx=1.0, vy=0.0),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=8, substep_dt=0.25),
        )
        self.assertGreater(result.metadata["exit_count_by_column"][0], 0)
        self.assertEqual(float(result.H_lag[4:, 0].max()), 0.0)

    def test_boundary_mass_is_dropped_not_renormalized(self) -> None:
        T = 4
        cfg = ResponseConfig(dt=1.0, lag_window_steps=2, substep_dt=0.5)
        wind = constant_direction(length=T, vx=0.0, vy=0.0)
        interior = build_lagged_response_matrix(
            single_source_map(nx=8, ny=8, x=4, y=4),
            ["interior"],
            impulse_basis(T),
            observer(),
            wind,
            response_config=cfg,
        )
        boundary = build_lagged_response_matrix(
            single_source_map(nx=8, ny=8, x=0, y=4),
            ["boundary"],
            impulse_basis(T),
            observer(),
            wind,
            response_config=cfg,
        )
        self.assertLess(
            boundary.metadata["kernel_mass_retained_by_column"][0],
            interior.metadata["kernel_mass_retained_by_column"][0],
        )
        self.assertGreater(boundary.metadata["dropped_mass_by_column"][0], 0.0)

    def test_anisotropic_dispersion_changes_response(self) -> None:
        T = 8
        common = dict(
            source_maps=single_source_map(x=2, y=4),
            source_names=["src"],
            activity_basis=impulse_basis(T),
            observer=observer(),
            wind_sampler_or_sequence=constant_direction(length=T, vx=1.0, vy=0.0),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=6, substep_dt=0.25),
        )
        aniso = build_lagged_response_matrix(**common, dispersion_config=DispersionConfig(sigma_parallel=0.8, sigma_perp=0.2))
        iso = build_lagged_response_matrix(**common, dispersion_config=DispersionConfig(sigma_parallel=0.4, sigma_perp=0.4))
        self.assertFalse(np.allclose(aniso.H_lag, iso.H_lag))

    def test_custom_position_dependent_sampler_is_supported(self) -> None:
        T = 5
        result = build_lagged_response_matrix(
            single_source_map(),
            ["src"],
            impulse_basis(T),
            observer(),
            PositionDependentWind(),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=3, substep_dt=0.5),
        )
        self.assertEqual(result.metadata["wind_provider"], "position_dependent_test")
        self.assertTrue(np.isfinite(result.H_lag).all())

    def test_zero_source_and_zero_basis_have_zero_diagnostics(self) -> None:
        T = 4
        wind = constant_direction(length=T, vx=1.0, vy=0.0)
        zero_source = build_lagged_response_matrix(
            np.zeros((1, 8, 8), dtype=np.float32), ["zero"], impulse_basis(T), observer(), wind
        )
        zero_basis = TemporalBasis(names=["zero"], values=np.zeros((T, 1), dtype=np.float32), metadata={})
        zero_activity = build_lagged_response_matrix(
            single_source_map(), ["src"], zero_basis, observer(), wind
        )
        for result in (zero_source, zero_activity):
            self.assertTrue(np.array_equal(result.H_lag, np.zeros_like(result.H_lag)))
            self.assertEqual(result.metadata["kernel_diagnostic_total_count"], 0)
            self.assertEqual(result.metadata["kernel_diagnostic_stored_count"], 0)
            self.assertFalse(result.metadata["kernel_diagnostics_truncated"])
            self.assertEqual(result.metadata["kernel_emitted_mass_by_column"], [0.0])
            self.assertEqual(result.metadata["kernel_observation_count_by_column"], [0])

    def test_fractional_final_substep_samples_without_overshoot(self) -> None:
        sampler = RecordingWind()
        build_lagged_response_matrix(
            single_source_map(),
            ["src"],
            impulse_basis(2),
            observer(),
            sampler,
            response_config=ResponseConfig(dt=1.0, lag_window_steps=2, substep_dt=0.3),
        )
        self.assertTrue(any(abs(t - 0.9) < 1e-7 for t in sampler.sample_times))
        self.assertLessEqual(max(sampler.sample_times), 0.9 + 1e-7)

    def test_diagnostic_truncation_metadata_and_package_export(self) -> None:
        T = 6
        result = build_lagged_response_matrix(
            single_source_map(),
            ["src"],
            impulse_basis(T),
            observer(),
            constant_direction(length=T, vx=0.0, vy=0.0),
            response_config=ResponseConfig(lag_window_steps=5, max_kernel_diagnostic_records=2),
        )
        self.assertEqual(result.metadata["kernel_diagnostic_total_count"], 5)
        self.assertEqual(result.metadata["kernel_diagnostic_stored_count"], 2)
        self.assertTrue(result.metadata["kernel_diagnostics_truncated"])
        self.assertEqual(len(result.metadata["kernel_mass_summaries"]), 2)
        for key in (
            "kernel_emitted_mass_by_column",
            "kernel_observation_count_by_column",
            "kernel_mass_retained_by_column",
            "dropped_mass_by_column",
            "kernel_quadrature_clip_count_by_column",
            "max_raw_retained_fraction_by_column",
        ):
            self.assertEqual(len(result.metadata[key]), result.H_lag.shape[1])
        self.assertIs(iasa_package.WindSampler, __import__("model.iasa.response", fromlist=["WindSampler"]).WindSampler)
        json.dumps(result.metadata)

    def test_trim_initial_lag_changes_row_count(self) -> None:
        T = 8
        result = build_lagged_response_matrix(
            single_source_map(),
            ["src"],
            impulse_basis(T),
            observer(),
            constant_direction(length=T, vx=1.0, vy=0.0),
            response_config=ResponseConfig(dt=1.0, lag_window_steps=4, trim_initial_lag=True),
        )
        self.assertEqual(result.H_lag.shape, (2 * (T - 3), 1))
        self.assertEqual(result.metadata["T_effective"], T - 3)
        self.assertEqual(result.metadata["trim_start"], 3)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            build_lagged_response_matrix(
                -single_source_map(),
                ["src"],
                impulse_basis(4),
                observer(),
                constant_direction(length=4, vx=1.0, vy=0.0),
            )
        with self.assertRaises(ValueError):
            build_lagged_response_matrix(
                np.zeros((8, 8), dtype=np.float32), ["src"], impulse_basis(4), observer(),
                constant_direction(length=4, vx=1.0, vy=0.0),
            )
        bad_basis = impulse_basis(4)
        bad_basis.values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_lagged_response_matrix(
                single_source_map(), ["src"], bad_basis, observer(),
                constant_direction(length=4, vx=1.0, vy=0.0),
            )
        with self.assertRaises(ValueError):
            build_lagged_response_matrix(
                single_source_map(), ["src"], impulse_basis(4), observer(), NonfiniteWind(),
            )
        for dispersion in (
            DispersionConfig(sigma_parallel=0.0),
            DispersionConfig(sigma_perp=-1.0),
            DispersionConfig(min_dispersion_time=0.0),
        ):
            with self.assertRaises(ValueError):
                build_lagged_response_matrix(
                    single_source_map(), ["src"], impulse_basis(4), observer(),
                    constant_direction(length=4, vx=1.0, vy=0.0), dispersion_config=dispersion,
                )
        for config in (
            ResponseConfig(lag_window_steps=0),
            ResponseConfig(max_kernel_diagnostic_records=-1),
            ResponseConfig(max_kernel_diagnostic_records=1.5),
        ):
            with self.assertRaises(ValueError):
                build_lagged_response_matrix(
                    single_source_map(), ["src"], impulse_basis(4), observer(),
                    constant_direction(length=4, vx=1.0, vy=0.0), response_config=config,
                )
        with self.assertRaises(ValueError):
            build_lagged_response_matrix(
                single_source_map(),
                ["src"],
                impulse_basis(4),
                Observer(sensor_ids=["bad"], sensor_xy=np.asarray([[99.0, 0.0]], dtype=np.float32)),
                constant_direction(length=4, vx=1.0, vy=0.0),
            )


if __name__ == "__main__":
    unittest.main()
