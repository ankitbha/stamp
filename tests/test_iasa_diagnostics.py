from __future__ import annotations

import json
import math
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
from model.iasa.diagnostics import (  # noqa: E402
    DiagnosticsConfig,
    diagnose_identifiability,
    diagnose_projection,
    summarize_wind_ensemble,
)
from model.iasa.projection import project_response_and_observations  # noqa: E402


def cols(source_names, basis_names=("c",)):
    out = []
    for k, s in enumerate(source_names):
        for b, name in enumerate(basis_names):
            out.append({"source_index": k, "source_name": s, "basis_index": b, "basis_name": name})
    return out


class DiagnosticsTests(unittest.TestCase):
    def test_duplicate_columns_rank_deficient(self) -> None:
        H = torch.tensor(
            [[1.0, 1.0, 0.2], [0.5, 0.5, 0.9], [0.2, 0.2, 0.1], [0.7, 0.7, 0.3]], dtype=torch.float64
        )
        d = diagnose_identifiability(H, cols(["a", "b", "c"]))
        self.assertLessEqual(d.sigma_J, 1e-8)
        self.assertEqual(d.condition_status, "infinite")
        self.assertIsNone(d.condition_number)
        self.assertLess(d.numerical_rank, 3)
        self.assertGreaterEqual(float(to_numpy(d.coherence)[0, 1]), 0.999)
        self.assertTrue(any("rank_deficient" in w for w in d.warnings))
        # J singular values, min is padded zero.
        self.assertEqual(d.singular_values.shape[0], 3)

    def test_orthogonal_columns_low_coherence_full_rank(self) -> None:
        H = torch.eye(5, 3, dtype=torch.float64)
        d = diagnose_identifiability(H, cols(["a", "b", "c"]))
        self.assertEqual(d.numerical_rank, 3)
        self.assertEqual(d.condition_status, "finite")
        offdiag = to_numpy(d.coherence)
        self.assertAlmostEqual(float(offdiag[0, 1]), 0.0, places=10)
        self.assertAlmostEqual(d.condition_number, 1.0, places=10)

    def test_near_zero_column_flagged_weak_with_null_pairs(self) -> None:
        H = torch.tensor(
            [[1.0, 1e-10, 0.2], [0.5, 1e-10, 0.9], [0.2, 1e-10, 0.1], [0.7, 1e-10, 0.3]], dtype=torch.float64
        )
        d = diagnose_identifiability(H, cols(["a", "weak", "c"]), config=DiagnosticsConfig(tau_v=1e-6))
        self.assertTrue(d.weak_flags[1])
        self.assertIn(1, d.weak_set)
        self.assertTrue(math.isnan(float(to_numpy(d.coherence)[0, 1])))
        self.assertTrue(math.isnan(float(to_numpy(d.coherence)[1, 2])))
        # eligible pair (0,2) is defined.
        self.assertFalse(math.isnan(float(to_numpy(d.coherence)[0, 2])))
        summary = d.to_json_summary()
        self.assertIsNone(summary["coherence"][0][1])
        self.assertIsNotNone(summary["coherence"][0][2])

    def test_wide_matrix_pads_singular_values_to_J(self) -> None:
        H = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)  # N=2, J=4
        d = diagnose_identifiability(H, cols(["a", "b", "c", "d"]))
        self.assertEqual(d.singular_values.shape[0], 4)
        self.assertLessEqual(d.sigma_J, 1e-12)
        self.assertLess(d.numerical_rank, 4)
        self.assertEqual(d.condition_status, "infinite")

    def test_fixed_zero_mask_reduces_and_preserves_mapping(self) -> None:
        H = torch.tensor(
            [[1.0, 5.0, 0.2, 0.3], [0.5, 5.0, 0.9, 0.1], [0.2, 5.0, 0.1, 0.4], [0.7, 5.0, 0.3, 0.8]],
            dtype=torch.float64,
        )
        column_index = cols(["a", "b", "c", "d"])
        full = diagnose_identifiability(H, column_index)
        masked = diagnose_identifiability(H, column_index, config=DiagnosticsConfig(fixed_zero_indices=(1,)))
        self.assertEqual(masked.reduced_to_original, [0, 2, 3])
        self.assertEqual(masked.original_to_reduced[1], -1)
        self.assertEqual(masked.original_to_reduced[2], 1)
        self.assertEqual(masked.singular_values.shape[0], 3)
        # Masking changes diagnostics only through the declared reduction: the
        # reduced diagnostics equal an independent diagnosis of the kept columns.
        independent = diagnose_identifiability(H[:, [0, 2, 3]], cols(["a", "c", "d"]))
        torch.testing.assert_close(masked.singular_values, independent.singular_values)
        # The full (unmasked) diagnosis is unaffected by any small-but-present column;
        # no fitted coefficient enters diagnose_identifiability at all.
        self.assertEqual(full.singular_values.shape[0], 4)

    def test_fixed_zero_validation(self) -> None:
        H = torch.eye(4, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "out of range"):
            diagnose_identifiability(H, cols(["a", "b", "c"]), config=DiagnosticsConfig(fixed_zero_indices=(5,)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            diagnose_identifiability(H, cols(["a", "b", "c"]), config=DiagnosticsConfig(fixed_zero_indices=(1, 1)))

    def test_absorption_from_projection(self) -> None:
        rows = [
            {"time_index": t, "sensor_index": i, "sensor_id": s}
            for t in range(3) for i, s in enumerate(("a", "b"))
        ]
        timestamps = torch.arange(3)  # unused count check only
        import numpy as np

        ts = np.datetime64("2026-06-01T00:00") + np.arange(3) * np.timedelta64(1, "h")
        H_lag = torch.arange(18, dtype=torch.float64).reshape(6, 3) + torch.eye(6, 3, dtype=torch.float64)
        Y = torch.zeros(6, dtype=torch.float64)
        background = build_background_basis(rows, ts)  # constant column
        projection = project_response_and_observations(
            H_lag, Y, background, rows, [{"c": i} for i in range(3)]
        )
        proj_cols = cols(["a", "b", "c"])
        d = diagnose_projection(projection, DiagnosticsConfig())
        self.assertIsNotNone(d.background_absorption)
        absorption = to_numpy(d.background_absorption)
        self.assertEqual(absorption.shape[0], 3)
        self.assertTrue(((absorption >= -1e-9) & (absorption <= 1.0 + 1e-9)).all())
        del timestamps

    def test_json_summary_serializable(self) -> None:
        H = torch.tensor([[1.0, 1e-12, 0.2], [0.5, 1e-12, 0.9], [0.2, 1e-12, 0.1]], dtype=torch.float64)
        d = diagnose_identifiability(H, cols(["a", "weak", "c"]), config=DiagnosticsConfig(tau_v=1e-6))
        text = json.dumps(d.to_json_summary())
        self.assertIn("sigma_J", text)
        self.assertNotIn("NaN", text)

    def test_device_and_dtype_preserved(self) -> None:
        H = torch.eye(5, 3, dtype=torch.float64)
        d = diagnose_identifiability(H, cols(["a", "b", "c"]))
        self.assertEqual(d.singular_values.dtype, torch.float64)
        self.assertEqual(d.singular_values.device.type, "cpu")
        self.assertEqual(d.metadata["device"], "cpu")
        self.assertEqual(d.metadata["dtype"], "float64")

    def test_wind_ensemble_deterministic(self) -> None:
        column_index = cols(["a", "b", "c"])
        results = []
        for scale in (1.0, 0.9, 1.1):
            H = torch.tensor(
                [[scale, scale, 0.2], [0.5, 0.5, 0.9], [0.2, 0.2, 0.1], [0.7, 0.7, 0.3]], dtype=torch.float64
            )
            results.append(diagnose_identifiability(H, column_index, config=DiagnosticsConfig(tau_rho=0.9)))
        first = summarize_wind_ensemble(results)
        second = summarize_wind_ensemble(results)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["n_windows"], 3)
        self.assertIn("q05", first["sigma_J_quantiles"])
        # a-b are duplicated in every window -> ambiguous every time.
        self.assertAlmostEqual(first["source_pair_ambiguity_probabilities"].get("a|b", 0.0), 1.0, places=9)

    def test_public_exports(self) -> None:
        for name in (
            "DiagnosticsConfig", "DiagnosticsResult", "diagnose_identifiability",
            "diagnose_projection", "summarize_wind_ensemble",
        ):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
