"""Task 11 reporting-layer tests (pure-python; no torch)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from model.iasa import reporting

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "evaluation" / "iasa_pol" / "runs"


def _load(rel: str) -> dict:
    return json.loads((RUNS / rel / "result.json").read_text())


class TestControlledReports(unittest.TestCase):
    def test_every_controlled_experiment_reports_tables(self):
        for k in range(1, 11):
            rel = f"exp{k:02d}_seed0"
            if not (RUNS / rel / "result.json").exists():
                continue
            result = _load(rel)
            tables = reporting.report_result(result)
            self.assertTrue(tables, f"exp{k:02d} produced no tables")
            for t in tables:
                self.assertEqual(set(t["rows"][0].keys()) if t["rows"] else set(t["columns"]),
                                 set(t["columns"]) if t["rows"] else set(t["columns"]))

    def test_exp08_states_interpretation(self):
        if not (RUNS / "exp08_seed0" / "result.json").exists():
            self.skipTest("exp08 run not present")
        tables = reporting.report_result(_load("exp08_seed0"))
        notes = " ".join(tables[0]["notes"]).lower()
        self.assertIn("non-rejection cannot certify", notes)
        self.assertIn("without identifying its cause", notes)

    def test_weak_pair_nulls_preserved(self):
        """A null triggering_pair / ray_distance must serialize as JSON null, not NaN."""
        synthetic = {"experiment": "exp02_coherent_sources_grouped",
                     "rows": [{"offset": 8.0, "triggering_pair": None, "ray_distance": None,
                               "max_eligible_coherence": 0.04, "individual_relative_error": 0.0,
                               "grouped_relative_error": 0.0, "merged": False,
                               "numerical_rank": 2, "sigma_J": 21.7}]}
        tables = reporting.report_result(synthetic)
        row = tables[0]["rows"][0]
        self.assertIsNone(row["triggering_pair"])
        self.assertIsNone(row["ray_distance"])
        # round-trips through JSON as null.
        self.assertIn('"ray_distance": null', json.dumps(tables[0], indent=0).replace("\n", ""))

    def test_nan_becomes_null(self):
        synthetic = {"experiment": "exp01_conditioning_predicts_recovery",
                     "rows": [{"noise_frac": 0.0, "geometry": "grid", "sigma_J": float("nan"),
                               "numerical_rank": 4, "effective_rank": 4, "condition_number": 1.0,
                               "coefficient_relative_error": 0.0, "residual_norm": 0.0,
                               "min_visibility": 1.0}]}
        row = reporting.report_result(synthetic)[0]["rows"][0]
        self.assertIsNone(row["sigma_J"])


class TestMergeChain(unittest.TestCase):
    def test_abc_chain_retains_both_edges(self):
        result = {"source_edges": [{"sources": ["A", "B"], "max_coherence": 0.9},
                                   {"sources": ["B", "C"], "max_coherence": 0.85}],
                  "report_components": [["A", "B", "C"]]}
        table = reporting.report_merge_edges(result)
        pairs = [tuple(r["sources"]) for r in table["rows"]]
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("B", "C"), pairs)
        self.assertEqual(len(table["rows"]), 2)  # no dedup / no collapse

    def test_nonsingleton_component_triggers_grouped_note(self):
        result = {"experiment": "exp10_footprints_spatial_attribution",
                  "report_components": [[0, 1], [2]], "footprint_localization_error_cells": 0.5,
                  "footprint_mass_fraction_within_radius": 0.9, "localization_radius_cells": 2,
                  "n_active_cells": 5, "contribution_sum_error": 1e-9,
                  "footprints_nonnegative": True, "coefficient_relative_error": 0.1}
        notes = " ".join(reporting.report_result(result)[0]["notes"]).lower()
        self.assertIn("group", notes)


class TestObservedReports(unittest.TestCase):
    def _weeks(self):
        weeks = []
        for k in range(1, 5):
            rel = f"week{k}/observed_seed0"
            if (RUNS / rel / "result.json").exists():
                weeks.append(_load(rel))
        return weeks

    def test_observed_tables_present_and_uncalibrated(self):
        weeks = self._weeks()
        if not weeks:
            self.skipTest("observed week runs not present")
        tables = reporting.report_observed(weeks)
        labels = {t["label"] for t in tables}
        self.assertTrue({"observed_identifiability", "observed_apportionment",
                         "observed_residuals", "observed_per_monitor"} <= labels)
        resid = next(t for t in tables if t["label"] == "observed_residuals")
        # No adequacy pass presented; calibration status present and uncalibrated.
        for row in resid["rows"]:
            self.assertIn("uncalibrated", (row.get("calibration_status") or "").lower())

    def test_apportionment_is_sensor_signal_fraction(self):
        weeks = self._weeks()
        if not weeks:
            self.skipTest("observed week runs not present")
        tables = reporting.report_observed(weeks)
        appt = next(t for t in tables if t["label"] == "observed_apportionment")
        note = " ".join(appt["notes"]).lower()
        self.assertIn("fractions of fitted", note)
        self.assertIn("not physical", note)

    def test_unsupported_group_marked(self):
        """A group with no admissible temporal component is 'unsupported', not 0.0."""
        wk = {"experiment": "observed_new_delhi", "window_index": 1,
              "source_names": ["a", "b"], "admissible_components_per_group": [[0], []],
              "sensor_signal_contribution_shares": {"a": 1.0, "b": 0.0},
              "diagnostics": {}, "report_components": [[0], [1]],
              "per_monitor_group_contributions": {}}
        tables = reporting.report_observed([wk])
        appt = next(t for t in tables if t["label"] == "observed_apportionment")
        self.assertEqual(appt["rows"][0]["b"], "unsupported")
        self.assertEqual(appt["rows"][0]["a"], 1.0)

    def test_wind_imputation_unavailable_is_labeled(self):
        t = reporting.report_wind_imputation(None)
        self.assertIn("unavailable", json.dumps(t).lower())


if __name__ == "__main__":
    unittest.main()
