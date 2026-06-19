from __future__ import annotations

import json
import unittest

import numpy as np

from model.iasa.background import BackgroundBasisConfig, build_background_basis


def rows(times: int = 3, sensors: tuple[str, ...] = ("a", "b")) -> list[dict[str, object]]:
    return [
        {"time_index": t, "sensor_index": i, "sensor_id": sensor}
        for t in range(times) for i, sensor in enumerate(sensors)
    ]


class BackgroundBasisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = np.datetime64("2026-06-01T00:00") + np.arange(50) * np.timedelta64(12, "h")

    def test_exact_time_major_columns_and_metadata(self) -> None:
        result = build_background_basis(
            rows(), self.timestamps,
            config=BackgroundBasisConfig(include_constant=True, temporal_polynomial_degree=1, daily_harmonics=1),
        )
        self.assertEqual(result.Q.shape, (6, 4))
        self.assertEqual(result.column_names, ["constant", "time_polynomial_1", "daily_sin_1", "daily_cos_1"])
        np.testing.assert_allclose(result.Q[:, 0], 1.0)
        np.testing.assert_allclose(result.Q[:, 1], np.repeat([-1.224744871, 0.0, 1.224744871], 2))
        np.testing.assert_allclose(result.Q[:, 2], np.repeat([0.0, 0.0, 0.0], 2), atol=1e-12)
        np.testing.assert_allclose(result.Q[:, 3], np.repeat([1.0, -1.0, 1.0], 2), atol=1e-12)
        self.assertEqual(result.metadata["row_ordering"], "time_major_sensor_minor")
        json.dumps(result.metadata)

    def test_reference_coded_days_sensors_and_regional_trends(self) -> None:
        timestamps = np.datetime64("2026-06-01T00:00") + np.arange(3) * np.timedelta64(24, "h")
        xy = np.asarray([[1.0, 10.0], [3.0, 14.0]])
        result = build_background_basis(
            rows(), timestamps, xy,
            BackgroundBasisConfig(
                include_constant=False, day_intercepts=True,
                regional_coordinate_trends=True, sensor_offsets=True,
            ),
        )
        self.assertEqual(
            result.column_names,
            ["day_2026-06-02", "day_2026-06-03", "regional_x_trend", "regional_y_trend", "sensor_offset_b"],
        )
        np.testing.assert_array_equal(result.Q[:, 0], [0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(result.Q[:, 1], [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result.Q[:, 2], [-1, 1, -1, 1, -1, 1])
        np.testing.assert_array_equal(result.Q[:, 4], [0, 1, 0, 1, 0, 1])

    def test_user_basis_empty_basis_and_rank_cap(self) -> None:
        empty = build_background_basis(rows(), self.timestamps, config=BackgroundBasisConfig(include_constant=False))
        self.assertEqual(empty.Q.shape, (6, 0))
        self.assertEqual(empty.metadata["effective_rank"], 0)
        user = np.arange(12, dtype=np.float64).reshape(6, 2)
        result = build_background_basis(
            rows(), self.timestamps, config=BackgroundBasisConfig(include_constant=False),
            user_basis=user, user_basis_names=["u", "v"],
        )
        np.testing.assert_array_equal(result.Q, user)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_background_basis(
                rows(), self.timestamps,
                config=BackgroundBasisConfig(include_constant=False, max_background_rank=1),
                user_basis=user, user_basis_names=["u", "v"],
            )
        stress = build_background_basis(
            rows(), self.timestamps,
            config=BackgroundBasisConfig(include_constant=False, max_background_rank=1, basis_mode="stress"),
            user_basis=user, user_basis_names=["u", "v"],
        )
        self.assertEqual(stress.metadata["basis_mode"], "stress")

    def test_validation_failures(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete time-major"):
            build_background_basis(rows()[:-1], self.timestamps)
        bad_order = rows()
        bad_order[3]["sensor_id"] = "a"
        with self.assertRaisesRegex(ValueError, "sensor IDs"):
            build_background_basis(bad_order, self.timestamps)
        with self.assertRaisesRegex(ValueError, "cover"):
            build_background_basis(rows(), self.timestamps[:2])
        with self.assertRaisesRegex(ValueError, "sensor_xy"):
            build_background_basis(
                rows(), self.timestamps,
                config=BackgroundBasisConfig(regional_coordinate_trends=True),
            )
        with self.assertRaisesRegex(ValueError, "user_basis_names length"):
            build_background_basis(rows(), self.timestamps, user_basis=np.ones((6, 2)), user_basis_names=["one"])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_background_basis(rows(), self.timestamps, user_basis=np.ones((6, 2)), user_basis_names=["x", "x"])
        with self.assertRaisesRegex(ValueError, "finite"):
            build_background_basis(rows(), self.timestamps, user_basis=np.full((6, 1), np.nan))


if __name__ == "__main__":
    unittest.main()
