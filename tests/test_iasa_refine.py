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
from model.iasa.fit import FitConfig, fit_projection  # noqa: E402
from model.iasa.projection import project_response_and_observations  # noqa: E402
from model.iasa.refine import (  # noqa: E402
    RefineConfig,
    refine_end_to_end,
)
from model.iasa.response import (  # noqa: E402
    DispersionConfig,
    Observer,
    ResponseConfig,
    build_lagged_response_matrix,
)
from model.iasa.wind import constant_direction


# Small, fast scenario: the refinement optimizer rebuilds the response many times
# per call, so the grid/time dimensions are kept small deliberately.
NX = NY = 10
T = 8
RC = ResponseConfig(dt=1.0, lag_window_steps=6, substep_dt=0.5, kernel_truncation_radius=3.0)
DC = DispersionConfig(sigma_parallel=0.7, sigma_perp=0.25, min_dispersion_time=0.25)


def compact_source(center, sigma=0.6):
    xs, ys = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    src = np.exp(-0.5 * (dx * dx + dy * dy) / float(sigma * sigma)).astype(np.float32)
    src[src < 0.05] = 0.0
    return src / max(float(src.max()), 1e-12)


def build_case(*, data_vx=1.0, data_vy=0.0, base_vx=1.0, base_vy=0.0, identical=False):
    """Data generated with wind (data_vx, data_vy); base/declared wind is (base_vx, base_vy)."""
    if identical:
        same = compact_source((5.0, 5.0))
        source_maps = np.stack([same, same.copy()], axis=0)
    else:
        source_maps = np.stack([compact_source((3.0, 5.0)), compact_source((5.0, 5.0))], axis=0)
    names = ["src_a", "src_b"]
    basis = TemporalBasis(names=["constant"], values=np.ones((T, 1), np.float32), metadata={})
    observer = Observer(sensor_ids=["west", "east", "north"],
                        sensor_xy=np.asarray([[1.0, 5.0], [8.0, 5.0], [5.0, 8.0]], np.float32))
    ts = np.datetime64("2026-06-01T00:00") + np.arange(T) * np.timedelta64(1, "h")

    data_wind = constant_direction(length=T, vx=data_vx, vy=data_vy)
    truth = build_lagged_response_matrix(source_maps, names, basis, observer, data_wind,
                                         response_config=RC, dispersion_config=DC)
    bg = build_background_basis(truth.row_index, ts, observer.sensor_xy,
                                BackgroundBasisConfig(include_constant=True, temporal_polynomial_degree=1))
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    beta = np.full(len(bg.column_names), 0.1, dtype=np.float64)
    Y = to_numpy(truth.H_lag).astype(np.float64) @ c_true + to_numpy(bg.Q) @ beta

    base_wind = constant_direction(length=T, vx=base_vx, vy=base_vy)
    base = build_lagged_response_matrix(source_maps, names, basis, observer, base_wind,
                                        response_config=RC, dispersion_config=DC)
    projection = project_response_and_observations(base.H_lag, Y, bg, base.row_index, base.column_index)
    fit = fit_projection(projection, config=FitConfig())
    return dict(source_maps=source_maps, names=names, basis=basis, observer=observer,
                base_wind=base_wind, Y=Y, bg=bg, projection=projection, fit=fit)


def _run(case, cfg):
    return refine_end_to_end(
        case["source_maps"], case["names"], case["basis"], case["observer"],
        case["base_wind"], case["Y"], case["bg"],
        fixed_response_fit=case["fit"], baseline_projection=case["projection"],
        response_config=RC, dispersion_config=DC, config=cfg,
    )


class RefineTests(unittest.TestCase):
    def test_feasibility_and_default_preserved(self) -> None:
        case = build_case(data_vx=1.0, data_vy=0.2)
        cfg = RefineConfig(lambda_w=0.01, lambda_psi=0.01, eps_w=0.4,
                           refine_dispersion=False, correction_basis="constant_linear", max_outer_iters=6)
        res = _run(case, cfg)
        self.assertLessEqual(res.constraint_satisfaction["wind_correction_inf_norm"], cfg.eps_w + 1e-9)
        self.assertTrue(res.constraint_satisfaction["eps_w_satisfied"])
        self.assertTrue(res.constraint_satisfaction["psi_in_box"])
        self.assertTrue(torch.allclose(res.fixed_response_fit.c_hat, case["fit"].c_hat))

    def test_fit_improves_under_wind_mismatch(self) -> None:
        case = build_case(data_vx=1.0, data_vy=0.25)
        cfg = RefineConfig(lambda_w=0.001, lambda_psi=0.001, eps_w=0.6,
                           refine_dispersion=True, correction_basis="constant", max_outer_iters=8)
        res = _run(case, cfg)
        self.assertLessEqual(res.objective_end["data"], res.objective_start["data"] + 1e-9)
        self.assertLessEqual(res.objective_end["total"], res.objective_start["total"] + 1e-9)

    def test_rejection_on_coherence_threshold(self) -> None:
        case = build_case(data_vx=1.0, data_vy=0.2)
        cfg = RefineConfig(lambda_w=0.001, lambda_psi=0.001, eps_w=0.5, tau_rho_ref=0.0,
                           refine_dispersion=False, correction_basis="constant", max_outer_iters=4)
        res = _run(case, cfg)
        self.assertFalse(res.accepted)
        self.assertIn("coherence", res.reason)

    def test_sigma_J_effective_is_zero_on_rank_deficiency(self) -> None:
        case = build_case(identical=True)
        cfg = RefineConfig(refine_wind=False, refine_dispersion=False, max_outer_iters=2)
        res = _run(case, cfg)
        self.assertEqual(res.sigma_J_baseline_eff, 0.0)
        self.assertEqual(res.sigma_J_refined_eff, 0.0)
        self.assertLess(res.rank_baseline, 2)

    def test_determinism(self) -> None:
        case = build_case(data_vx=1.0, data_vy=0.2)
        cfg = RefineConfig(lambda_w=0.01, lambda_psi=0.01, eps_w=0.4,
                           refine_dispersion=False, correction_basis="constant", max_outer_iters=5)
        r1 = _run(case, cfg)
        r2 = _run(case, cfg)
        self.assertEqual(r1.wind_correction, r2.wind_correction)
        self.assertEqual(r1.refined_psi, r2.refined_psi)
        self.assertEqual(r1.objective_end, r2.objective_end)

    def test_noop_under_strong_anchor(self) -> None:
        case = build_case(data_vx=1.0, data_vy=0.0)
        cfg = RefineConfig(lambda_w=1e6, lambda_psi=1e6, eps_w=0.3,
                           refine_dispersion=True, correction_basis="constant", max_outer_iters=5)
        res = _run(case, cfg)
        self.assertTrue(res.accepted)
        self.assertLessEqual(res.constraint_satisfaction["wind_correction_inf_norm"], 1e-6)
        self.assertAlmostEqual(res.refined_psi[0], DC.sigma_parallel, places=6)

    def test_public_exports(self) -> None:
        for name in ("RefineConfig", "RefineResult", "refine_end_to_end", "projected_data_objective"):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
