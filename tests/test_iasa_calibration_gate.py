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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_iasa_sanity as sanity  # noqa: E402


class CalibrationGateTests(unittest.TestCase):
    def test_ks_uniform_statistic(self) -> None:
        # A perfectly uniform grid has a small KS statistic; a degenerate sample large.
        uniform = (np.arange(1, 101) - 0.5) / 100.0
        self.assertLess(sanity._ks_uniform_statistic(uniform), 0.02)
        self.assertGreater(sanity._ks_uniform_statistic(np.zeros(100)), 0.9)

    def test_calibration_gate_small(self) -> None:
        # Reduced-scale run: exercises the full study path (null rate, KS, power ramp,
        # determinism) cheaply. Tolerances are relaxed for the small trial count.
        result = sanity.run_calibration_gate(
            n_trials=60, n_replicates=60, alpha=0.1,
            amplitudes=(0.0, 0.4, 1.2), sigma_e=0.1,
            uniformity_threshold=0.25, power_target=0.8, seed=0,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["gate"], "calibration")
        # Null rejection rate near alpha (within the reported MC tolerance).
        self.assertTrue(result["rejection_rate_within_tolerance"])
        # Power ramps up and the largest omission is (almost) always rejected.
        rates = result["rejection_rates"]
        self.assertEqual(len(rates), 3)
        self.assertGreaterEqual(rates[-1], 0.8)
        self.assertTrue(result["power_increases_monotonically_with_omission_amplitude"])
        self.assertTrue(result["deterministic"])
        # Provenance recorded.
        self.assertEqual(result["n_trials"], 60)
        self.assertEqual(result["n_replicates"], 60)
        self.assertFalse(result["noise_model_provenance"]["estimated_from_fit_residual"])

    def test_calibration_gate_deterministic_across_calls(self) -> None:
        kw = dict(n_trials=40, n_replicates=50, alpha=0.1, amplitudes=(0.0, 1.2),
                  sigma_e=0.1, uniformity_threshold=0.3, power_target=0.7, seed=3)
        r1 = sanity.run_calibration_gate(**kw)
        r2 = sanity.run_calibration_gate(**kw)
        self.assertEqual(r1["rejection_rates"], r2["rejection_rates"])
        self.assertEqual(r1["p_value_ks_statistic"], r2["p_value_ks_statistic"])


if __name__ == "__main__":
    unittest.main()
