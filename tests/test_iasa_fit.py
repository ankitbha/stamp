from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model.iasa as iasa  # noqa: E402
from model.iasa.backend import to_numpy  # noqa: E402
from model.iasa.background import BackgroundBasisConfig, build_background_basis  # noqa: E402
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_identifiability  # noqa: E402
from model.iasa.fit import (  # noqa: E402
    AdequacyConfig,
    FitConfig,
    NoiseModel,
    _projected_fista,
    aggregate_inventory_scenarios,
    aggregate_transport_ensemble,
    fit_projection,
    fit_sources,
    residual_adequacy_check,
    summarize_report_groups,
)
from model.iasa.projection import project_response_and_observations  # noqa: E402

import numpy as np  # noqa: E402


DTYPE = torch.float64


def cols(source_names, basis_names=("c",)):
    out = []
    for k, s in enumerate(source_names):
        for b, name in enumerate(basis_names):
            out.append({"source_index": k, "source_name": s, "basis_index": b, "basis_name": name})
    return out


def well_conditioned_H():
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.05, 0.0],
            [0.0, 0.1, 0.05],
            [0.05, 0.0, 0.1],
        ],
        dtype=DTYPE,
    )


def row_index_for(n_rows):
    return [{"time_index": t, "sensor_index": 0, "sensor_id": "s"} for t in range(n_rows)]


def empty_projection(H, Y):
    rows = row_index_for(H.shape[0])
    ts = np.datetime64("2026-06-01T00:00") + np.arange(H.shape[0]) * np.timedelta64(1, "h")
    bg = build_background_basis(rows, ts, config=BackgroundBasisConfig(include_constant=False))
    return project_response_and_observations(H, Y, bg, rows, cols(["s0", "s1", "s2"]))


class FitTests(unittest.TestCase):
    def test_noiseless_recovery(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]))
        err = float(torch.linalg.vector_norm(fit.c_hat - c_true) / torch.linalg.vector_norm(c_true))
        self.assertLessEqual(err, 1e-4)
        self.assertGreaterEqual(float(fit.c_hat.min()), -1e-8)
        self.assertLessEqual(fit.kkt_residual, FitConfig().tol_kkt)
        self.assertEqual(fit.convergence_status, "converged")

    def test_nonnegativity_enforced(self) -> None:
        H = well_conditioned_H()
        # A target that unconstrained least squares would fit with a negative coeff.
        c_true = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]))
        self.assertGreaterEqual(float(fit.c_hat.min()), -1e-9)
        self.assertEqual(float(fit.c_hat[1]), max(0.0, float(fit.c_hat[1])))

    def test_noisy_residual_below_zero_model(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        gen = torch.Generator().manual_seed(0)
        noise = 0.01 * torch.randn(H.shape[0], dtype=DTYPE, generator=gen)
        fit = fit_sources(H, H @ c_true + noise, cols(["s0", "s1", "s2"]))
        err = float(torch.linalg.vector_norm(fit.c_hat - c_true) / torch.linalg.vector_norm(c_true))
        self.assertLessEqual(err, 0.1)
        self.assertLess(fit.residual_norm, fit.zero_model_residual_norm)

    def test_duplicate_columns_sum_stable(self) -> None:
        H = well_conditioned_H()
        H[:, 1] = H[:, 0]
        pair_sum = 2.2
        Y = H[:, 0] * pair_sum
        fit = fit_sources(H, Y, cols(["dup_a", "dup_b", "other"]))
        self.assertLessEqual(abs(float(fit.c_hat[0] + fit.c_hat[1]) - pair_sum), 1e-4)

    def test_ill_conditioned_warning(self) -> None:
        H = well_conditioned_H()
        H[:, 1] = H[:, 0] + 1e-9
        fit = fit_sources(H, H @ torch.tensor([1.0, 0.5, 0.2], dtype=DTYPE), cols(["a", "b", "c"]))
        self.assertTrue(any("ill_conditioned" in w for w in fit.warnings))

    def test_mask_restores_zero_and_mapping(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.0, 1e-7], dtype=DTYPE)
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]), config=FitConfig(fixed_zero_indices=(1,)))
        self.assertEqual(float(fit.c_hat[1]), 0.0)
        self.assertEqual(fit.reduced_to_original, [0, 2])
        self.assertEqual(fit.original_to_reduced[1], -1)
        self.assertEqual(fit.original_to_reduced[2], 1)
        # The unmasked near-zero column 2 is still fitted (present in reduced set).
        self.assertIn(2, fit.reduced_to_original)

    def test_fit_diagnostic_mask_mismatch_rejected(self) -> None:
        H = well_conditioned_H()
        Y = H @ torch.tensor([1.0, 0.5, 0.2], dtype=DTYPE)
        diag_full = diagnose_identifiability(H, cols(["s0", "s1", "s2"]))
        with self.assertRaisesRegex(ValueError, "mask mismatch"):
            fit_sources(H, Y, cols(["s0", "s1", "s2"]), config=FitConfig(fixed_zero_indices=(1,)), diagnostics=diag_full)
        diag_masked = diagnose_identifiability(H, cols(["s0", "s1", "s2"]), config=DiagnosticsConfig(fixed_zero_indices=(1,)))
        fit_sources(H, Y, cols(["s0", "s1", "s2"]), config=FitConfig(fixed_zero_indices=(1,)), diagnostics=diag_masked)

    def test_batched_fista_matches_single(self) -> None:
        H = well_conditioned_H()
        targets = [torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE), torch.tensor([0.2, 0.0, 1.1], dtype=DTYPE)]
        Ys = torch.stack([H @ c for c in targets], dim=1)
        prior = torch.zeros(3, dtype=DTYPE)
        sigma1 = float(torch.linalg.svdvals(H)[0])
        batched = _projected_fista(H, Ys, prior, 0.0, sigma1=sigma1, max_iters=10000, tol_kkt=1e-8, tol_rel_obj=1e-14)["c"]
        for b, c in enumerate(targets):
            single = _projected_fista(H, H @ c, prior, 0.0, sigma1=sigma1, max_iters=10000, tol_kkt=1e-8, tol_rel_obj=1e-14)["c"]
            torch.testing.assert_close(batched[:, b], single, atol=1e-6, rtol=1e-6)

    def test_active_set_covariance_shape(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]))
        # Columns 0 and 1 are active; 2 is not.
        self.assertEqual(fit.active_indices, [0, 1])
        self.assertIsNotNone(fit.active_covariance)
        self.assertEqual(tuple(fit.active_covariance.shape), (2, 2))
        cov = to_numpy(fit.active_covariance)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) >= -1e-12))

    def test_theta_and_contribution_summaries(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        basis = torch.ones((4, 1), dtype=DTYPE)  # T=4, B=1
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]), temporal_basis=basis)
        self.assertIsNotNone(fit.theta)
        self.assertEqual(tuple(fit.theta.shape), (4, 3))
        totals = fit.source_contribution_summaries["total_contribution"]
        self.assertAlmostEqual(totals["s0"], 4.0 * float(fit.c_hat[0]), places=6)

    def test_summarize_report_groups(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.3], dtype=DTYPE)
        basis = torch.ones((4, 1), dtype=DTYPE)
        fit = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]), temporal_basis=basis)
        grouped = summarize_report_groups(fit, [[0, 1], [2]])
        totals = fit.source_contribution_summaries["total_contribution"]
        self.assertAlmostEqual(grouped["groups"][0]["total_contribution"], totals["s0"] + totals["s1"], places=6)

    def test_ensemble_kind_pooling_rejected(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        transport = [fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]), config=FitConfig(ensemble_kind="transport")) for _ in range(3)]
        inventory = fit_sources(H, H @ c_true, cols(["s0", "s1", "s2"]), config=FitConfig(ensemble_kind="inventory"))
        agg = aggregate_transport_ensemble(transport)
        self.assertEqual(agg["ensemble_kind"], "transport")
        self.assertEqual(agg["n_members"], 3)
        with self.assertRaisesRegex(ValueError, "transport"):
            aggregate_transport_ensemble(transport + [inventory])
        with self.assertRaisesRegex(ValueError, "inventory"):
            aggregate_inventory_scenarios(transport)
        scenarios = aggregate_inventory_scenarios([inventory], scenario_names=["baseline"])
        self.assertEqual(scenarios["scenarios"][0]["scenario"], "baseline")

    def test_uncalibrated_adequacy(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        proj = empty_projection(H, H @ c_true)
        fit = fit_projection(proj)
        result = residual_adequacy_check(fit, proj, None)
        self.assertEqual(result.calibration_status, "uncalibrated")
        self.assertIsNone(result.p_value)
        self.assertIsNone(result.inadequate)
        self.assertGreaterEqual(result.raw_residual_norm, 0.0)
        text = json.dumps(result.to_json_summary())
        self.assertIn("uncalibrated", text)

    def test_uncalibrated_when_estimated_from_fit_residual(self) -> None:
        H = well_conditioned_H()
        proj = empty_projection(H, H @ torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE))
        fit = fit_projection(proj)
        noise = NoiseModel(covariance=0.01, calibrated=True, source="x", estimated_from_fit_residual=True)
        result = residual_adequacy_check(fit, proj, noise)
        self.assertEqual(result.calibration_status, "uncalibrated")
        self.assertIsNone(result.inadequate)

    def test_calibrated_adequacy_detects_omission(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        gen = torch.Generator().manual_seed(3)
        Y_obs = H @ c_true + 0.05 * torch.randn(H.shape[0], dtype=DTYPE, generator=gen)
        proj = empty_projection(H, Y_obs)
        fit = fit_projection(proj)
        noise = NoiseModel(covariance=0.05 ** 2, calibrated=True, source="external_v1")
        adequacy = residual_adequacy_check(fit, proj, noise, config=AdequacyConfig(n_replicates=200, seed=1))
        self.assertEqual(adequacy.calibration_status, "calibrated")
        self.assertIsNotNone(adequacy.T_res)
        self.assertTrue(0.0 < adequacy.p_value <= 1.0)

        Y_omit = Y_obs + 2.0 * H[:, 2]
        proj_omit = empty_projection(H, Y_omit)
        fit_wrong = fit_sources(H, proj_omit.Y_tilde, cols(["s0", "s1", "s2"]), config=FitConfig(fixed_zero_indices=(2,)),
                                H_lag=H, U_r=proj_omit.U_r, Y=Y_omit)
        adequacy_omit = residual_adequacy_check(fit_wrong, proj_omit, noise, config=AdequacyConfig(n_replicates=200, seed=1))
        self.assertGreater(adequacy_omit.T_res, adequacy.T_res)

    def test_temporal_multi_basis_recovery(self) -> None:
        from datetime import datetime, timedelta
        import math

        T = 24
        diurnal = [math.exp(-0.5 * ((h - 8) / 2.0) ** 2) + math.exp(-0.5 * ((h - 18) / 2.0) ** 2) for h in range(T)]
        block = [1.0 if h < 12 else 0.0 for h in range(T)]
        Phi = torch.tensor([[diurnal[h], block[h]] for h in range(T)], dtype=DTYPE)
        C_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
        theta_true = Phi @ C_true.transpose(0, 1)
        H = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.05, 0.0, 0.0],
                [0.0, 0.1, 0.05, 0.0],
                [0.0, 0.0, 0.1, 0.05],
                [0.05, 0.0, 0.0, 0.1],
            ],
            dtype=DTYPE,
        )
        c_true = C_true.reshape(-1)
        temporal_cols = cols(["traffic", "brick_kilns"], ["diurnal", "block"])
        ts = [datetime(2026, 6, 1, 0, 0) + timedelta(hours=h) for h in range(T)]
        fit = fit_sources(H, H @ c_true, temporal_cols, temporal_basis=Phi, timestamps=ts)
        err = float(torch.linalg.matrix_norm(fit.theta - theta_true) / torch.linalg.matrix_norm(theta_true))
        self.assertLessEqual(err, 0.1)
        for key in ("total_contribution", "diurnal_hourly_mean", "active_period_fraction", "daily_totals"):
            self.assertIn(key, fit.source_contribution_summaries)
        # Brick kiln is intermittent (block on for the first half only).
        self.assertAlmostEqual(fit.source_contribution_summaries["active_period_fraction"]["brick_kilns"], 0.5, places=6)

    def test_calibrated_adequacy_span_absorbed_not_detected(self) -> None:
        H = well_conditioned_H()
        c_true = torch.tensor([1.5, 0.7, 0.0], dtype=DTYPE)
        gen = torch.Generator().manual_seed(3)
        Y_obs = H @ c_true + 0.05 * torch.randn(H.shape[0], dtype=DTYPE, generator=gen)
        noise = NoiseModel(covariance=0.05 ** 2, calibrated=True, source="external_v1")
        cfg = AdequacyConfig(n_replicates=200, seed=1)

        proj = empty_projection(H, Y_obs)
        base = residual_adequacy_check(fit_projection(proj), proj, noise, config=cfg)

        # An omitted signal in span(H) (column 1 is free) is absorbed by the fit.
        Y_span = Y_obs + 1.5 * H[:, 1]
        proj_span = empty_projection(H, Y_span)
        span = residual_adequacy_check(fit_projection(proj_span), proj_span, noise, config=cfg)
        self.assertEqual(span.calibration_status, "calibrated")
        self.assertFalse(span.inadequate)
        self.assertLessEqual(span.T_res, span.bootstrap_quantile)
        # The absorbed case behaves like the correctly specified fit, not the
        # residual-visible omission.
        self.assertLess(abs(span.T_res - base.T_res), base.bootstrap_quantile)

    def test_device_and_dtype_preserved(self) -> None:
        H = well_conditioned_H()
        fit = fit_sources(H, H @ torch.tensor([1.0, 0.5, 0.2], dtype=DTYPE), cols(["s0", "s1", "s2"]))
        self.assertEqual(fit.c_hat.dtype, torch.float64)
        self.assertEqual(fit.c_hat.device.type, "cpu")
        self.assertEqual(fit.metadata["dtype"], "float64")

    def test_public_exports(self) -> None:
        for name in (
            "FitConfig", "FitResult", "NoiseModel", "AdequacyConfig", "AdequacyResult",
            "fit_sources", "fit_projection", "residual_adequacy_check",
            "aggregate_transport_ensemble", "aggregate_inventory_scenarios", "summarize_report_groups",
        ):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
