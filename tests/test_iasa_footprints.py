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
from model.iasa.activity import TemporalBasis  # noqa: E402
from model.iasa.backend import to_numpy  # noqa: E402
from model.iasa.background import BackgroundBasisConfig, build_background_basis  # noqa: E402
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_projection  # noqa: E402
from model.iasa.fit import FitConfig, fit_projection  # noqa: E402
from model.iasa.footprints import (  # noqa: E402
    compute_sensor_footprints,
    decompose_per_sensor,
    per_sensor_identifiability,
)
from model.iasa.merge import recommend_merges  # noqa: E402
from model.iasa.projection import project_response_and_observations  # noqa: E402
from model.iasa.response import (  # noqa: E402
    DispersionConfig,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import constant_direction  # noqa: E402


def compact_source(nx, ny, center, sigma=0.6):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    src = np.exp(-0.5 * (dx * dx + dy * dy) / float(sigma * sigma)).astype(np.float32)
    src[src < 0.05] = 0.0
    return src / max(float(src.max()), 1e-12)


def build_scenario(nx=16, ny=16, T=14):
    source_maps = np.stack([compact_source(nx, ny, (5.0, 8.0)), compact_source(nx, ny, (8.0, 8.0))], axis=0)
    names = ["src_west", "src_mid"]
    basis = TemporalBasis(names=["constant"], values=np.ones((T, 1), np.float32), metadata={})
    observer = Observer(sensor_ids=["west", "east", "north"],
                        sensor_xy=np.asarray([[1.0, 8.0], [13.0, 8.0], [8.0, 13.0]], np.float32))
    wind = constant_direction(length=T, vx=1.0, vy=0.0)
    rc = ResponseConfig(dt=1.0, lag_window_steps=10, substep_dt=0.25, kernel_truncation_radius=3.0)
    dc = DispersionConfig(sigma_parallel=0.7, sigma_perp=0.25, min_dispersion_time=0.25)
    response = build_lagged_response_matrix(source_maps, names, basis, observer, wind, response_config=rc, dispersion_config=dc)
    ts = np.datetime64("2026-06-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    bg = build_background_basis(response.row_index, ts, observer.sensor_xy,
                                BackgroundBasisConfig(include_constant=True, temporal_polynomial_degree=1))
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    beta = np.full(len(bg.column_names), 0.1, dtype=np.float64)
    Y = to_numpy(response.H_lag).astype(np.float64) @ c_true + to_numpy(bg.Q) @ beta
    projection = project_response_and_observations(response.H_lag, Y, bg, response.row_index, response.column_index)
    fit = fit_projection(projection, config=FitConfig())
    return source_maps, names, basis, observer, wind, rc, dc, projection, fit


class FootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (cls.maps, cls.names, cls.basis, cls.observer, cls.wind, cls.rc, cls.dc,
         cls.projection, cls.fit) = build_scenario()
        cls.diag = diagnose_projection(cls.projection, DiagnosticsConfig())
        cls.merge = recommend_merges(cls.diag, fit=cls.fit, H_tilde=cls.projection.H_tilde)
        cls.groups = [c["members"] for c in cls.merge.report_components]
        cls.fp = compute_sensor_footprints(
            cls.maps, cls.names, cls.basis, cls.observer, cls.wind, fit=cls.fit, projection=cls.projection,
            response_config=cls.rc, dispersion_config=cls.dc, groups=cls.groups,
        )

    def test_contributions_sum_to_fitted_signal(self) -> None:
        fitted = to_numpy(self.fit.fitted_sensor_vector)
        dec = decompose_per_sensor(self.projection.H_tilde, self.projection.H_tilde + self.projection.H_removed,
                                   self.fit.c_hat, self.projection.row_index, self.projection.column_index)
        for sid, rows in dec["sensor_rows"].items():
            proj_total = sum(self.fp.per_sensor_source_contribution_projected[sid].values())
            fitted_total = float(fitted[np.asarray(rows, dtype=np.int64)].sum())
            self.assertAlmostEqual(proj_total, fitted_total, places=6)

    def test_group_aggregation_equals_member_sum(self) -> None:
        for sid, per_src in self.fp.per_sensor_source_contribution_raw.items():
            for members in self.groups:
                key = "+".join(sorted(self.names[m] for m in members))
                self.assertAlmostEqual(
                    self.fp.per_sensor_group_contribution_raw[sid][key],
                    sum(per_src[self.names[m]] for m in members), places=8,
                )

    def test_footprints_nonnegative(self) -> None:
        for f in self.fp.geometric_footprint.values():
            self.assertGreaterEqual(np.asarray(f).min(), -1e-9)
        for gd in self.fp.fitted_footprint.values():
            for field in gd.values():
                self.assertGreaterEqual(np.asarray(field).min(), -1e-9)

    def test_footprint_localizes_upwind_origin(self) -> None:
        geom_east = np.asarray(self.fp.geometric_footprint["east"])
        geom_west = np.asarray(self.fp.geometric_footprint["west"])
        peak_ix, _ = np.unravel_index(int(np.argmax(geom_east)), geom_east.shape)
        self.assertLess(peak_ix, 13)  # upwind of the east sensor
        self.assertGreater(float(geom_east.sum()), float(geom_west.sum()))

    def test_fitted_footprint_sums_to_raw_contribution(self) -> None:
        for sid, gd in self.fp.fitted_footprint.items():
            for key, field in gd.items():
                self.assertAlmostEqual(
                    float(np.asarray(field).sum()),
                    self.fp.per_sensor_group_contribution_raw[sid][key], places=5,
                )

    def test_inheritance(self) -> None:
        for si in sorted({int(r["sensor_index"]) for r in self.projection.row_index}):
            info = per_sensor_identifiability(self.projection, si, pooled=self.diag)
            self.assertLessEqual(info["sigma_J_sensor"], info["sigma_J_pooled"] + 1e-9)
            self.assertLessEqual(info["rank_sensor"], info["rank_pooled"])
            self.assertTrue(info["inherited"])

    def test_public_exports(self) -> None:
        for name in ("FootprintResult", "compute_sensor_footprints", "decompose_per_sensor", "per_sensor_identifiability"):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
