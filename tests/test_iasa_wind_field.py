from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model.iasa as iasa  # noqa: E402
from model.iasa.backend import to_numpy  # noqa: E402
from model.iasa.activity import TemporalBasis  # noqa: E402
from model.iasa.response import (  # noqa: E402
    DispersionConfig,
    GriddedWindSampler,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import (  # noqa: E402
    KernelCoordinateQueryImputer,
    build_gridded_wind_field,
    build_wind_field_ensemble,
    evaluate_gridded_wind_heldout,
    transport_vectors_from_wd_ws,
)


def synthetic_stations(nx=16, ny=16, T=12):
    coords = np.asarray([[2.0, 2.0], [13.0, 2.0], [2.0, 13.0], [13.0, 13.0]], dtype=np.float64)
    S = coords.shape[0]
    wd = np.asarray([[45.0], [135.0], [225.0], [315.0]], dtype=np.float64) + np.zeros((S, T))
    ws = np.full((S, T), 1.0, dtype=np.float64)
    vectors = transport_vectors_from_wd_ws(wd, ws).astype(np.float32)
    mask = np.ones((S, T), dtype=bool)
    return coords, vectors, mask, (nx, ny), T


class WindFieldTests(unittest.TestCase):
    def test_transport_convention(self) -> None:
        self.assertTrue(np.allclose(transport_vectors_from_wd_ws(0.0, 1.0), [0.0, -1.0], atol=1e-6))
        self.assertTrue(np.allclose(transport_vectors_from_wd_ws(90.0, 1.0), [-1.0, 0.0], atol=1e-6))
        self.assertTrue(np.allclose(transport_vectors_from_wd_ws(180.0, 1.0), [0.0, 1.0], atol=1e-6))
        self.assertTrue(np.allclose(transport_vectors_from_wd_ws(270.0, 1.0), [1.0, 0.0], atol=1e-6))

    def test_field_shape_and_variation(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        field = build_gridded_wind_field(
            coords, vectors, mask, np.arange(T), grid,
            imputer=KernelCoordinateQueryImputer(length_scale=2.0), dt_s=1.0, dx_m=1.0, dy_m=1.0,
        )
        self.assertEqual(field.field.shape, (T, grid[0], grid[1], 2))
        self.assertEqual(field.physical_field.shape, (T, grid[0], grid[1], 2))
        self.assertGreater(float(np.mean(np.std(field.field.reshape(T, -1, 2), axis=1))), 1e-3)
        self.assertEqual(field.convention, "transport_vectors_wd_ws_eq_wind_direction_conversion")

    def test_grid_displacement_scaling(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        field = build_gridded_wind_field(
            coords, vectors, mask, np.arange(T), grid,
            imputer=KernelCoordinateQueryImputer(length_scale=2.0), dt_s=2.0, dx_m=4.0, dy_m=4.0,
        )
        # vx_grid = Ux * dt_s/dx_m = Ux * 0.5.
        np.testing.assert_allclose(field.field, field.physical_field * 0.5, rtol=0, atol=1e-6)

    def test_sampler_constant_field(self) -> None:
        field = np.zeros((3, 8, 8, 2), dtype=np.float32)
        field[..., 0] = 1.5
        field[..., 1] = -0.5
        sampler = GriddedWindSampler(field=field, provider="test", metadata={})
        for pos in ([3.4, 2.1], [0.0, 7.0], [7.0, 0.0]):
            np.testing.assert_allclose(sampler.sample(1.3, np.asarray(pos)), [1.5, -0.5], atol=1e-6)

    def test_sampler_bilinear_ramp(self) -> None:
        nx = ny = 4
        field = np.zeros((1, nx, ny, 2), dtype=np.float32)
        for ix in range(nx):
            field[0, ix, :, 0] = float(ix)  # ramps with x
        sampler = GriddedWindSampler(field=field, provider="test", metadata={})
        self.assertAlmostEqual(float(sampler.sample(0.0, np.asarray([1.5, 2.0]))[0]), 1.5, places=5)
        self.assertAlmostEqual(float(sampler.sample(0.0, np.asarray([2.25, 0.0]))[0]), 2.25, places=5)

    def test_sampler_station_recovery(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        field = build_gridded_wind_field(
            coords, vectors, mask, np.arange(T), grid,
            imputer=KernelCoordinateQueryImputer(length_scale=2.0), dt_s=1.0, dx_m=1.0, dy_m=1.0,
        )
        sampler = GriddedWindSampler.from_gridded_wind_field(field)
        for s in range(coords.shape[0]):
            got = sampler.sample(0.0, coords[s])
            np.testing.assert_allclose(got, vectors[s, 0], atol=0.05)

    def test_drives_response_builder_unchanged(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        nx, ny = grid
        field = build_gridded_wind_field(
            coords, vectors, mask, np.arange(T), grid,
            imputer=KernelCoordinateQueryImputer(length_scale=2.0), dt_s=1.0, dx_m=1.0, dy_m=1.0,
        )
        sampler = GriddedWindSampler.from_gridded_wind_field(field)
        source = np.zeros((1, nx, ny), dtype=np.float32)
        source[0, 8, 8] = 1.0
        basis = TemporalBasis(names=["constant"], values=np.ones((T, 1), np.float32), metadata={})
        observer = Observer(sensor_ids=["a", "b"], sensor_xy=np.asarray([[4.0, 8.0], [12.0, 8.0]], np.float32))
        result = build_lagged_response_matrix(
            source, ["s"], basis, observer, sampler,
            response_config=ResponseConfig(dt=1.0, lag_window_steps=8, substep_dt=0.25, kernel_truncation_radius=3.0),
            dispersion_config=DispersionConfig(),
        )
        H = to_numpy(result.H_lag)
        self.assertTrue(np.isfinite(H).all() and np.any(H > 0))

    def test_heldout_evaluation(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        result = evaluate_gridded_wind_heldout(
            coords, vectors, mask, np.arange(T),
            imputer=KernelCoordinateQueryImputer(length_scale=6.0), holdout_station_indices=(0,),
        )
        self.assertTrue(np.isfinite(result["vector_rmse"]))
        self.assertGreaterEqual(result["direction_mae_degrees"], 0.0)
        self.assertLessEqual(result["direction_mae_degrees"], 180.0)
        self.assertGreater(result["n_heldout_station_times"], 0)

    def test_ensemble_transport_tagged_and_distinct(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        members = build_wind_field_ensemble(
            coords, vectors, mask, np.arange(T), grid, n_members=4,
            method="station_bootstrap", imputer=KernelCoordinateQueryImputer(length_scale=2.0),
            dt_s=1.0, dx_m=1.0, dy_m=1.0, seed=0,
        )
        self.assertEqual(len(members), 4)
        self.assertTrue(all(m.ensemble_kind == "transport" for m in members))
        self.assertGreater(max(float(np.max(np.abs(m.field - members[0].field))) for m in members[1:]), 0.0)

    def test_ensemble_deterministic(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        kwargs = dict(n_members=3, method="residual_perturbation", residual_scale=0.1,
                      imputer=KernelCoordinateQueryImputer(length_scale=2.0), dt_s=1.0, dx_m=1.0, dy_m=1.0, seed=7)
        a = build_wind_field_ensemble(coords, vectors, mask, np.arange(T), grid, **kwargs)
        b = build_wind_field_ensemble(coords, vectors, mask, np.arange(T), grid, **kwargs)
        for ma, mb in zip(a, b):
            np.testing.assert_array_equal(ma.field, mb.field)

    def test_public_exports(self) -> None:
        for name in (
            "GriddedWindField", "GriddedWindSampler", "KernelCoordinateQueryImputer",
            "CoordinateQueryImputer", "transport_vectors_from_wd_ws", "build_gridded_wind_field",
            "build_wind_field_ensemble", "evaluate_gridded_wind_heldout", "gridded_new_delhi_wind_field",
        ):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
