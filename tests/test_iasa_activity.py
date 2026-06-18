from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.activity import (  # noqa: E402
    TemporalBasis,
    build_default_activity_profile,
    build_theta_from_temporal_basis,
    combine_inventory_sources,
)
from sim.pol_sources import SOURCE_NAMES  # noqa: E402


class IasaActivityTests(unittest.TestCase):
    def test_zero_activity_produces_zero_source_terms(self) -> None:
        source_maps = np.ones((3, 4, 5), dtype=np.float32)
        theta = np.zeros((2, 3), dtype=np.float32)
        combined = combine_inventory_sources(source_maps, theta)
        self.assertEqual(combined.shape, (2, 4, 5))
        self.assertTrue(np.array_equal(combined, np.zeros_like(combined)))

    def test_one_hot_activity_activates_one_source_group(self) -> None:
        source_maps = np.stack(
            [
                np.full((2, 2), 1.0, dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
                np.full((2, 2), 9.0, dtype=np.float32),
            ],
            axis=0,
        )
        combined = combine_inventory_sources(source_maps, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(combined, source_maps[1])

    def test_negative_activity_raises_when_nonnegative(self) -> None:
        source_maps = np.ones((1, 2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            combine_inventory_sources(source_maps, np.asarray([-1.0], dtype=np.float32))

    def test_traffic_defaults_use_nearest_time_slice(self) -> None:
        timestamps = pd.date_range("2018-05-01 00:00:00+05:30", periods=24, freq="h")
        profile = build_default_activity_profile(SOURCE_NAMES, timestamps)
        by_name = {name: i for i, name in enumerate(profile.source_names)}
        for hour, source_name in ((0, "traffic_00"), (6, "traffic_06"), (12, "traffic_12"), (18, "traffic_18")):
            self.assertEqual(float(profile.theta[hour, by_name[source_name]]), 1.0)
        traffic_cols = [by_name[name] for name in SOURCE_NAMES if name.startswith("traffic_")]
        np.testing.assert_allclose(profile.theta[:, traffic_cols].sum(axis=1), np.ones(24, dtype=np.float32))

    def test_industry_default_is_day_heavy(self) -> None:
        timestamps = pd.date_range("2018-05-01 00:00:00+05:30", periods=24, freq="h")
        profile = build_default_activity_profile(SOURCE_NAMES, timestamps, industry_24h_fraction=0.25)
        industry = profile.theta[:, profile.source_names.index("industries")]
        day_mean = float(industry[[8, 12, 16]].mean())
        night_mean = float(industry[[1, 3, 23]].mean())
        self.assertGreater(day_mean, night_mean)
        self.assertEqual(profile.metadata["industries"]["spatial_sampling"], "metadata_only_v1")

    def test_population_default_has_cooking_peaks(self) -> None:
        timestamps = pd.date_range("2018-05-01 00:00:00+05:30", periods=24, freq="h")
        profile = build_default_activity_profile(SOURCE_NAMES, timestamps)
        population = profile.theta[:, profile.source_names.index("population_density")]
        for peak_hour in (7, 13, 19):
            self.assertGreater(float(population[peak_hour]), float(population[(peak_hour + 4) % 24]))
        self.assertEqual(profile.metadata["population_density"]["cooking_peak_hours"], [7, 13, 19])

    def test_one_basis_temporal_coefficient_produces_expected_theta(self) -> None:
        timestamps = pd.date_range("2018-05-01 00:00:00+05:30", periods=4, freq="h")
        basis = TemporalBasis(
            names=["constant"],
            values=np.ones((4, 1), dtype=np.float32),
            metadata={"kind": "test"},
        )
        coefficients = np.asarray([[0.5], [2.0]], dtype=np.float32)
        profile = build_theta_from_temporal_basis(["a", "b"], timestamps, basis, coefficients)
        expected = np.asarray(
            [
                [0.5, 2.0],
                [0.5, 2.0],
                [0.5, 2.0],
                [0.5, 2.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(profile.theta, expected)
        self.assertEqual(profile.metadata["basis_names"], ["constant"])


if __name__ == "__main__":
    unittest.main()
