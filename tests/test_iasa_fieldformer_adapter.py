from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model.iasa as iasa  # noqa: E402
from baselines.fieldformer.model import (  # noqa: E402
    FieldFormerCoordinateQuery,
    load_fieldformer_checkpoint,
)
from model.iasa.fieldformer_adapter import (  # noqa: E402
    FieldFormerCoordinateQueryImputer,
    build_fieldformer_wind_imputer,
    build_untrained_wind_model,
)
from model.iasa.wind import build_gridded_wind_field, transport_vectors_from_wd_ws


def synthetic_stations(nx=12, ny=12, T=6):
    coords = np.asarray([[2.0, 2.0], [9.0, 2.0], [2.0, 9.0], [9.0, 9.0]], dtype=np.float64)
    S = coords.shape[0]
    wd = np.asarray([[45.0], [135.0], [225.0], [315.0]], dtype=np.float64) + np.zeros((S, T))
    ws = np.full((S, T), 1.0, dtype=np.float64)
    vectors = transport_vectors_from_wd_ws(wd, ws).astype(np.float32)
    mask = np.ones((S, T), dtype=bool)
    return coords, vectors, mask, (nx, ny), T


class FieldFormerAdapterTests(unittest.TestCase):
    def test_out_dim_guard(self) -> None:
        scalar_model = FieldFormerCoordinateQuery(d_model=16, nhead=2, layers=1, d_ff=32, out_dim=1)
        with self.assertRaisesRegex(ValueError, "out_dim=2"):
            FieldFormerCoordinateQueryImputer(model=scalar_model)

    def test_query_plumbing_shape_and_finite(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        imputer = build_fieldformer_wind_imputer(model=build_untrained_wind_model(), k_neighbors=8, time_radius=2)
        query_cells = np.asarray([[x, y] for x in (1.0, 5.0, 9.0) for y in (1.0, 5.0, 9.0)], dtype=np.float64)
        out = imputer.query(query_cells, 0, coords, vectors, mask)
        self.assertEqual(out.shape, (query_cells.shape[0], 2))
        self.assertTrue(np.isfinite(out).all())

    def test_drives_build_gridded_wind_field(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        imputer = build_fieldformer_wind_imputer(model=build_untrained_wind_model(), k_neighbors=8, time_radius=2)
        field = build_gridded_wind_field(
            coords, vectors, mask, np.arange(T), grid, imputer=imputer, dt_s=1.0, dx_m=1.0, dy_m=1.0,
        )
        self.assertEqual(field.field.shape, (T, grid[0], grid[1], 2))
        self.assertTrue(np.isfinite(field.field).all())
        self.assertEqual(field.metadata["imputer"], "fieldformer_coordinate_query")

    def test_checkpoint_round_trip(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        model = build_untrained_wind_model()
        model.eval()
        imputer = FieldFormerCoordinateQueryImputer(model=model, k_neighbors=8, time_radius=2)
        query_cells = np.asarray([[3.0, 4.0], [7.0, 8.0]], dtype=np.float64)
        before = imputer.query(query_cells, 1, coords, vectors, mask)
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "wind_ff.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
            loaded = build_fieldformer_wind_imputer(
                checkpoint_path=str(ckpt), d_model=32, nhead=4, layers=2, d_ff=64,
                k_neighbors=8, time_radius=2,
            )
        after = loaded.query(query_cells, 1, coords, vectors, mask)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)

    def test_builder_requires_exactly_one_source(self) -> None:
        with self.assertRaises(ValueError):
            build_fieldformer_wind_imputer()  # neither
        with self.assertRaises(ValueError):
            build_fieldformer_wind_imputer(checkpoint_path="x", model=build_untrained_wind_model())  # both

    def test_default_imputer_is_kernel(self) -> None:
        coords, vectors, mask, grid, T = synthetic_stations()
        field = build_gridded_wind_field(coords, vectors, mask, np.arange(T), grid, dt_s=1.0, dx_m=1.0, dy_m=1.0)
        self.assertEqual(field.metadata["imputer"], "kernel_coordinate_query")

    def test_public_exports(self) -> None:
        for name in ("FieldFormerCoordinateQueryImputer", "build_fieldformer_wind_imputer", "build_untrained_wind_model"):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
