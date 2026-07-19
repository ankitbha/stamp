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
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_identifiability  # noqa: E402
from model.iasa.fit import FitConfig, fit_sources  # noqa: E402
from model.iasa.merge import MergeResult, recommend_merges  # noqa: E402

DTYPE = torch.float64


def cols(source_names, basis_names=("c",)):
    out = []
    for k, s in enumerate(source_names):
        for b, name in enumerate(basis_names):
            out.append({"source_index": k, "source_name": s, "basis_index": b, "basis_name": name})
    return out


def component_of(merge, k):
    for comp in merge.report_components:
        if k in comp["members"]:
            return tuple(comp["members"])
    raise AssertionError(f"source {k} missing")


DUP_H = torch.tensor(
    [[1.0, 1.0, 0.2], [0.5, 0.5, 0.9], [0.2, 0.2, 0.1], [0.0, 0.0, 0.7], [0.3, 0.3, 0.4], [0.8, 0.8, 0.6]],
    dtype=DTYPE,
)


class MergeTests(unittest.TestCase):
    def test_duplicate_merged_with_trigger(self) -> None:
        diag = diagnose_identifiability(DUP_H, cols(["dup_a", "dup_b", "other"]))
        merge = recommend_merges(diag)
        self.assertEqual(component_of(merge, 0), (0, 1))
        self.assertEqual(component_of(merge, 2), (2,))
        edge = next(e for e in merge.source_edges if e["sources"] == (0, 1))
        self.assertGreaterEqual(edge["max_coherence"], 0.999)
        self.assertIn("col_i", edge["trigger"])
        self.assertIn("source_i", edge["trigger"])

    def test_separated_not_merged(self) -> None:
        H = torch.eye(6, 3, dtype=DTYPE)
        diag = diagnose_identifiability(H, cols(["a", "b", "c"]))
        merge = recommend_merges(diag)
        self.assertTrue(all(len(c["members"]) == 1 for c in merge.report_components))
        self.assertEqual(merge.source_edges, [])

    def test_weak_flagged_no_edge(self) -> None:
        H = torch.eye(6, 3, dtype=DTYPE).clone()
        H[:, 1] = 1e-10
        diag = diagnose_identifiability(H, cols(["a", "weak", "c"]), config=DiagnosticsConfig(tau_v=1e-6))
        merge = recommend_merges(diag)
        self.assertIn(1, merge.weak_flags["weak_column_indices"])
        self.assertFalse(any(1 in e["sources"] for e in merge.source_edges))

    def test_chain_transitive_over_merge(self) -> None:
        e = 0.12
        H = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, e, e], [0.0, 0.0, e], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=DTYPE,
        )
        diag = diagnose_identifiability(H, cols(["A", "B", "C"]), config=DiagnosticsConfig(tau_rho=0.99))
        merge = recommend_merges(diag)
        self.assertEqual(component_of(merge, 0), (0, 1, 2))
        edges = {edge["sources"] for edge in merge.source_edges}
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)
        self.assertNotIn((0, 2), edges)  # A and C distinguishable, but chained via B

    def test_grouped_sums_no_refit(self) -> None:
        diag = diagnose_identifiability(DUP_H, cols(["dup_a", "dup_b", "other"]))
        c_true = torch.tensor([1.1, 1.1, 0.5], dtype=DTYPE)
        fit = fit_sources(DUP_H, DUP_H @ c_true, cols(["dup_a", "dup_b", "other"]), config=FitConfig())
        merge = recommend_merges(diag, fit=fit, H_tilde=DUP_H)
        grouped = next(g for g in merge.grouped_activity["groups"] if g["members"] == [0, 1])
        self.assertLessEqual(abs(grouped["total_contribution"] - float(c_true[0] + c_true[1])), 1e-4)
        sensor = next(g for g in merge.grouped_sensor_contribution if g["members"] == [0, 1])
        expected = (DUP_H[:, 0] * float(c_true[0] + c_true[1])).tolist()
        for a, b in zip(sensor["contribution"], expected):
            self.assertAlmostEqual(a, b, places=4)

    def test_deterministic(self) -> None:
        diag = diagnose_identifiability(DUP_H, cols(["dup_a", "dup_b", "other"]))
        a = recommend_merges(diag).to_json_summary()
        b = recommend_merges(diag).to_json_summary()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_conservative_no_finest_claim(self) -> None:
        diag = diagnose_identifiability(DUP_H, cols(["dup_a", "dup_b", "other"]))
        merge = recommend_merges(diag)
        self.assertTrue(all(c["is_conservative"] for c in merge.report_components))
        self.assertFalse(merge.resolution["finest_guarantee"])

    def test_global_unresolved_warning(self) -> None:
        # Rank deficient (duplicate) but tau_rho high enough that no edge forms.
        diag = diagnose_identifiability(DUP_H, cols(["dup_a", "dup_b", "other"]), config=DiagnosticsConfig(tau_rho=1.0))
        merge = recommend_merges(diag)
        self.assertEqual(merge.source_edges, [])
        self.assertTrue(merge.global_unresolved_warning)
        self.assertTrue(any("global_unresolved" in w for w in merge.warnings))

    def test_no_false_global_warning_when_full_rank(self) -> None:
        H = torch.eye(6, 3, dtype=DTYPE)
        diag = diagnose_identifiability(H, cols(["a", "b", "c"]))
        merge = recommend_merges(diag)
        self.assertFalse(merge.global_unresolved_warning)

    def test_public_exports(self) -> None:
        for name in ("MergeConfig", "MergeResult", "recommend_merges"):
            self.assertTrue(hasattr(iasa, name), name)


if __name__ == "__main__":
    unittest.main()
