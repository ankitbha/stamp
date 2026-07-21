from __future__ import annotations

import json
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

from experiments.iasa_pol import runio  # noqa: E402
from experiments.iasa_pol.experiments import run_named_experiment  # noqa: E402
from experiments.iasa_pol.nd_platform import PlatformConfig, build_platform  # noqa: E402


def _tiny_platform():
    return build_platform(PlatformConfig(grid_shape=(12, 12), T=12, lag_window_steps=6))


FAST = {
    "exp01": {"noise_fracs": [0.0, 0.1]},
    "exp02": {"offsets": [4.0, 1.0]},
    "exp03": {"background_modes": ["none", "primary", "stress"], "noise_frac": 0.05},
    "exp04": {"wind_kinds": ["constant", "multi"], "layouts": ["regulatory", "downwind"],
              "ensemble_members": 3, "n_sensors": 5},
    "exp05": {"wind_direction_perturbations_deg": [0.0, 10.0],
              "structural_adequacy_trials": 2, "structural_n_replicates": 40},
    "exp06": {},
    "exp07": {"tau_L": 1e-3, "lag_grid": [4, 6, 8]},
    "exp08": {"N": 32, "n_trials": 6, "n_replicates": 40, "omission_amplitude": 1.2},
    "exp09": {"noise_fracs": [0.0, 0.05]},
    "exp10": {},
    "observed": {"wind_kind": "real", "use_real_pm25": True, "T": 12},
}


class ExperimentSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.platform = _tiny_platform()

    def test_all_experiments_runnable_with_accuracy_and_diagnostics(self):
        for name, params in FAST.items():
            out = run_named_experiment(name, self.platform, params, seed=0)
            result = out["result"]
            # JSON-serializable (no tensors/objects leak through).
            json.dumps(result)
            self.assertEqual(result.get("experiment", "").split("_")[0][:3] in ("exp", "obs") or
                             name == "observed", True)
            if name == "observed":
                continue
            text = json.dumps(result)
            self.assertTrue(any(t in text for t in (
                "coefficient_relative_error", "rejection_rate", "contribution_sum_error",
                "activity_relative_error", "grouped_relative_error")), f"{name} lacks accuracy")
            self.assertTrue(any(t in text for t in ("sigma_J", "numerical_rank")), f"{name} lacks diagnostics")

    def test_exp05_structural_generator_labeled(self):
        out = run_named_experiment("exp05", self.platform, FAST["exp05"], seed=0)["result"]
        self.assertEqual(out["structural"]["generator"], "edge_hold_pde")
        self.assertEqual(out["structural"]["edge_hold_config"]["boundary"], "edge_hold")
        self.assertEqual(out["parametric"]["transport_ensemble_kind"], "transport")

    def test_exp06_type_separation_enforced(self):
        out = run_named_experiment("exp06", self.platform, FAST["exp06"], seed=0)["result"]
        self.assertTrue(out["transport_inventory_pooling_rejected"])
        self.assertEqual(out["scenario_aggregate_kind"], "inventory")

    def test_observed_mode_has_no_recovery_metric(self):
        out = run_named_experiment("observed", self.platform, FAST["observed"], seed=0)["result"]
        self.assertIsNone(out["recovery_error"])
        self.assertFalse(out["has_ground_truth"])
        # Observed mode reports sensor-signal-space contribution shares (not raw
        # coefficient-magnitude ratios) and never imputes PM2.5.
        if out.get("status") != "insufficient_observed_rows":
            self.assertIn("sensor_signal_contribution_shares", out)
            self.assertFalse(out["pm25_imputed"])
            self.assertEqual(out["n_source_groups"], 4)

    def test_reproducible_under_fixed_seed(self):
        r1 = run_named_experiment("exp01", self.platform, FAST["exp01"], seed=0)["result"]
        r2 = run_named_experiment("exp01", self.platform, FAST["exp01"], seed=0)["result"]
        self.assertEqual(json.dumps(r1["rows"], sort_keys=True), json.dumps(r2["rows"], sort_keys=True))

    def test_exp07_row_count_fixed_and_selection_not_from_fit(self):
        out = run_named_experiment("exp07", self.platform, FAST["exp07"], seed=0)["result"]
        self.assertTrue(out["row_count_fixed"])
        self.assertFalse(out["coefficients_used_for_selection"])

    def test_exp08_negative_control_is_genuine(self):
        out = run_named_experiment("exp08", self.platform, FAST["exp08"], seed=0)["result"]
        # A residual-visible (out-of-span) omission is detected often; a REAL in-span
        # omission (absorbed by the fit) is not -- so the test must fail if the aligned
        # case were secretly detectable or the visible case undetectable.
        self.assertGreaterEqual(out["residual_visible_power"], 0.5)
        self.assertLessEqual(out["aligned_negative_control_rejection_rate"], 0.5)
        self.assertGreaterEqual(
            out["residual_visible_power"] - out["aligned_negative_control_rejection_rate"], 0.25)

    def test_exp10_contributions_sum_to_fitted_signal(self):
        out = run_named_experiment("exp10", self.platform, FAST["exp10"], seed=0)["result"]
        self.assertLess(out["contribution_sum_error"], 1e-5)
        self.assertTrue(out["footprints_nonnegative"])

    def test_runner_roundtrip_and_provenance(self):
        out = run_named_experiment("exp01", self.platform, FAST["exp01"], seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            resolved = runio.resolved_config(
                {"experiment": "exp01", "params": FAST["exp01"]}, seed=3, device="cpu",
                platform_meta=self.platform.metadata,
                platform_config=self.platform.config.to_json(),
            )
            run_dir = runio.write_run(Path(tmp), experiment="exp01", seed=3,
                                      resolved=resolved, result=out["result"], arrays=out["arrays"])
            self.assertTrue((run_dir / "config.resolved.json").exists())
            self.assertTrue((run_dir / "result.json").exists())
            saved = json.loads((run_dir / "config.resolved.json").read_text())
            self.assertIn("git_sha", saved)
            self.assertIn("inventory_version", saved)
            self.assertEqual(saved["seed"], 3)
            reloaded = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(reloaded["experiment"], "exp01_conditioning_predicts_recovery")


if __name__ == "__main__":
    unittest.main()
