from __future__ import annotations

import json
import unittest

import numpy as np
import torch

import model.iasa as iasa
from model.iasa.backend import to_numpy
from model.iasa.background import BackgroundBasisConfig, build_background_basis
from model.iasa.projection import ProjectionConfig, fit_background_projector, project_response_and_observations
from scripts.run_iasa_sanity import run_projection_gate


ROWS = [
    {"time_index": t, "sensor_index": i, "sensor_id": sensor}
    for t in range(3) for i, sensor in enumerate(("a", "b"))
]
TIMESTAMPS = np.datetime64("2026-06-01T00:00") + np.arange(3) * np.timedelta64(1, "h")


def basis(user: np.ndarray, names: list[str], *, mode: str = "normal", cap: int = 8):
    return build_background_basis(
        ROWS, TIMESTAMPS,
        config=BackgroundBasisConfig(include_constant=False, basis_mode=mode, max_background_rank=cap),
        user_basis=user, user_basis_names=names,
    )


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.H = np.arange(18, dtype=np.float64).reshape(6, 3) + np.eye(6, 3)
        self.y = np.arange(6, dtype=np.float64)
        self.columns = [{"column": i} for i in range(3)]

    def test_empty_and_constant_projection(self) -> None:
        empty = build_background_basis(ROWS, TIMESTAMPS, config=BackgroundBasisConfig(include_constant=False))
        empty_result = project_response_and_observations(self.H, self.y, empty, ROWS, self.columns)
        self.assertIsInstance(empty_result.H_tilde, torch.Tensor)
        self.assertEqual(empty_result.H_tilde.device.type, "cpu")
        np.testing.assert_array_equal(to_numpy(empty_result.H_tilde), self.H)
        np.testing.assert_array_equal(to_numpy(empty_result.Y_tilde), self.y)
        self.assertEqual(tuple(empty_result.U_r.shape), (6, 0))

        constant = build_background_basis(ROWS, TIMESTAMPS)
        result = project_response_and_observations(self.H, self.y, constant, ROWS, self.columns)
        np.testing.assert_allclose(to_numpy(result.H_tilde).mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(to_numpy(result.Y_tilde).mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(to_numpy(result.H_tilde) + to_numpy(result.H_removed), self.H)
        self.assertEqual(result.metadata["effective_rank"], 1)
        json.dumps(result.metadata)

    def test_redundancy_tolerance_idempotence_and_reconstruction(self) -> None:
        q = np.column_stack([np.ones(6), np.arange(6), np.ones(6)])
        redundant = basis(q, ["constant", "trend", "constant_copy"])
        projector = fit_background_projector(redundant)
        self.assertEqual(projector.effective_rank, 2)
        self.assertEqual(projector.metadata["dependent_column_names"], ["constant_copy"])
        once = to_numpy(projector.project(self.H))
        np.testing.assert_allclose(to_numpy(projector.project(once)), once, atol=1e-12)
        U_r = to_numpy(projector.U_r)
        np.testing.assert_allclose(once, self.H - U_r @ (U_r.T @ self.H), atol=1e-12)
        high_tolerance = fit_background_projector(redundant, ProjectionConfig(rank_tolerance=1e6))
        self.assertEqual(high_tolerance.effective_rank, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            fit_background_projector(redundant, ProjectionConfig(rank_tolerance=-1.0))

    def test_prefix_classification_uses_complete_prefix_at_explicit_tolerance(self) -> None:
        subthreshold = np.zeros((6, 2), dtype=np.float64)
        subthreshold[0, :] = 0.8
        duplicate = basis(subthreshold, ["first_subthreshold", "second_crosses_tolerance"])
        projector = fit_background_projector(duplicate, ProjectionConfig(rank_tolerance=1.0))
        self.assertEqual(projector.effective_rank, 1)
        self.assertEqual(projector.metadata["independent_column_names"], ["second_crosses_tolerance"])
        self.assertEqual(projector.metadata["dependent_column_names"], ["first_subthreshold"])

    def test_over_rank_requires_labeled_stress_opt_out(self) -> None:
        q = np.eye(6, 3)
        stress = basis(q, ["a", "b", "c"], mode="stress", cap=2)
        with self.assertRaisesRegex(ValueError, "only labeled stress"):
            fit_background_projector(stress)
        accepted = fit_background_projector(stress, ProjectionConfig(allow_over_rank=True))
        self.assertEqual(accepted.effective_rank, 3)

    def test_shape_alignment_and_nonfinite_errors(self) -> None:
        constant = build_background_basis(ROWS, TIMESTAMPS)
        with self.assertRaisesRegex(ValueError, "Y must"):
            project_response_and_observations(self.H, self.y[:, None], constant, ROWS, self.columns)
        with self.assertRaisesRegex(ValueError, "row counts"):
            project_response_and_observations(self.H[:-1], self.y[:-1], constant, ROWS, self.columns)
        with self.assertRaisesRegex(ValueError, "column_index"):
            project_response_and_observations(self.H, self.y, constant, ROWS, self.columns[:-1])
        mismatched = [dict(row) for row in ROWS]
        mismatched[0]["sensor_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "match exactly"):
            project_response_and_observations(self.H, self.y, constant, mismatched, self.columns)
        bad = self.H.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            project_response_and_observations(bad, self.y, constant, ROWS, self.columns)

    def test_zero_row_projectors_and_projections(self) -> None:
        for background in (
            build_background_basis([], []),
            build_background_basis([], [], config=BackgroundBasisConfig(include_constant=False)),
        ):
            projector = fit_background_projector(background)
            self.assertEqual(projector.U_r.shape, (0, 0))
            self.assertEqual(projector.effective_rank, 0)
            self.assertEqual(projector.singular_values.tolist(), [])
            self.assertEqual(projector.tolerance, 0.0)
            result = project_response_and_observations(
                np.empty((0, 2)), np.empty(0), background, [], [{"column": 0}, {"column": 1}],
            )
            self.assertEqual(result.H_tilde.shape, (0, 2))
            self.assertEqual(result.Y_tilde.shape, (0,))
            self.assertEqual(result.H_removed.shape, (0, 2))
            self.assertEqual(result.Y_removed.shape, (0,))
            json.dumps(result.metadata)

        zero_row_constant = build_background_basis([], [])
        explicit_tolerance = fit_background_projector(
            zero_row_constant, ProjectionConfig(rank_tolerance=1.0),
        )
        self.assertEqual(explicit_tolerance.Q.shape, (0, 1))
        self.assertEqual(explicit_tolerance.tolerance, 0.0)
        self.assertEqual(explicit_tolerance.metadata["rank_tolerance"], 0.0)

    def test_public_exports_and_gate(self) -> None:
        for name in (
            "BackgroundBasisConfig", "BackgroundBasisResult", "BackgroundProjector", "ProjectionConfig",
            "ProjectionResult", "build_background_basis", "fit_background_projector",
            "project_response_and_observations",
        ):
            self.assertTrue(hasattr(iasa, name), name)
        gate = run_projection_gate()
        self.assertEqual(gate["status"], "ok")
        self.assertEqual(gate["normal_effective_rank"], 4)


if __name__ == "__main__":
    unittest.main()
