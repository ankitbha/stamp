"""Tests for the signal-target grouping (model/iasa/grouping.py)."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.linalg import null_space, orth
from scipy.optimize import nnls

from model.iasa.grouping import (
    GroupingConfig,
    exact_components,
    group_report,
    is_exactly_identifiable,
    join_partitions,
    separation,
)
from model.iasa.grouping import _minimal_partitions_exhaustive  # noqa: PLC2701

import torch


def _partitions(items):
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def _canon(P):
    return sorted((sorted(g) for g in P), key=lambda g: g[0])


def _refines(P, Q):
    return all(any(set(g) <= set(h) for h in Q) for g in P)


def _identifiable_by_kernel(A, P):
    K = null_space(A, rcond=1e-9)
    return all(not K.size or np.linalg.norm(A[:, g] @ K[g, :]) < 1e-7 for g in P)


def _planted(rng, N, sizes, ranks):
    cols = [rng.standard_normal((N, r)) @ rng.standard_normal((r, k)) for k, r in zip(sizes, ranks)]
    A = np.concatenate(cols, axis=1)
    return A[:, rng.permutation(A.shape[1])]


class TestExactComponents(unittest.TestCase):
    def test_components_are_the_unique_finest_identifiable_partition(self):
        rng = np.random.default_rng(1)
        for _ in range(60):
            J = int(rng.integers(3, 7))
            sizes, left = [], J
            while left:
                k = int(rng.integers(1, min(3, left) + 1))
                sizes.append(k)
                left -= k
            ranks = [max(1, k - int(rng.integers(0, 2))) for k in sizes]
            A = _planted(rng, sum(ranks) + 2, sizes, ranks)
            if rng.random() < 0.3:
                i, j, k = rng.choice(J, 3, replace=False)
                A[:, k] = A[:, i] + 0.5 * A[:, j]
            ident = [P for P in _partitions(range(J)) if _identifiable_by_kernel(A, P)]
            for P in ident:
                self.assertTrue(is_exactly_identifiable(A, P))
            minimal = [P for P in ident if not any(Q != P and _refines(Q, P) for Q in ident)]
            self.assertEqual(len(minimal), 1)
            self.assertEqual(exact_components(A), _canon(minimal[0]))

    def test_zero_column_is_its_own_component(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((10, 3))
        A[:, 1] = 0.0
        A[:, 2] = A[:, 0] * 2.0
        self.assertEqual(exact_components(A), [[0, 2], [1]])

    def test_join_with_declared_partition(self):
        self.assertEqual(join_partitions([[0], [1], [2, 3]], [[0, 1], [2], [3]], 4), [[0, 1], [2, 3]])


class TestSeparation(unittest.TestCase):
    def test_two_columns_give_ray_distance(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            A = rng.standard_normal((15, 2))
            rho = abs(A[:, 0] @ A[:, 1]) / np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1])
            self.assertAlmostEqual(separation(A, [0]), math.sqrt(1 - rho**2), places=10)
            self.assertAlmostEqual(separation(A, [0]), separation(A, [1]), places=10)

    def test_whole_set_has_separation_one(self):
        A = np.eye(3)
        self.assertEqual(separation(A, [0, 1, 2]), 1.0)

    def test_least_squares_and_nnls_group_errors_respect_the_bound(self):
        rng = np.random.default_rng(4)
        for _ in range(40):
            A = _planted(rng, 30, [2, 1, 2], [1, 1, 2])
            A = A + 0.7 * rng.standard_normal((30, 1)) @ rng.standard_normal((1, A.shape[1]))
            P = exact_components(A)
            U = orth(A, rcond=1e-9)
            c = np.abs(rng.standard_normal(A.shape[1]))
            for _ in range(20):
                eps = 0.3 * rng.standard_normal(30)
                y = A @ c + eps
                c_ls = np.linalg.lstsq(A, y, rcond=None)[0]
                c_nn, _ = nnls(A, y, maxiter=5000)
                bound_norm = np.linalg.norm(U.T @ eps)
                for g in P:
                    s = separation(A, g)
                    for chat in (c_ls, c_nn):
                        err = np.linalg.norm(A[:, g] @ (chat[g] - c[g]))
                        self.assertLessEqual(err, bound_norm / s + 1e-8)


class TestReportablePartitions(unittest.TestCase):
    def test_two_incomparable_minimal_partitions(self):
        a = 0.3
        A = np.array([[1, math.cos(a), math.cos(a)], [0, math.sin(a), 0], [0, 0, math.sin(a)]])
        res = group_report(A, GroupingConfig(tau_theta=0.25))
        self.assertEqual(sorted(res.minimal_partitions), sorted([[[0, 1], [2]], [[0, 2], [1]]]))
        self.assertEqual(len(res.tied_alternatives), 1)

    def test_reportable_partitions_are_not_closed_under_coarsening(self):
        A = np.array([
            [0.248, -0.142, 0.165, 0.828],
            [0.899, -0.775, -0.917, 0.229],
            [-0.356, -0.381, 0.243, -0.507],
            [-0.063, -0.484, -0.270, 0.069],
        ])
        tau = 0.3
        self.assertTrue(all(separation(A, [j]) >= tau for j in range(4)))
        self.assertLess(separation(A, [0, 1]), tau)

    def test_exhaustive_matches_brute_force(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            J = int(rng.integers(3, 6))
            A = rng.standard_normal((10, J))
            i, j, k = rng.choice(J, 3, replace=False)
            A[:, k] = A[:, i] - 0.7 * A[:, j] + 0.1 * rng.standard_normal(10)
            tau = 0.25
            good = [P for P in _partitions(range(J)) if all(separation(A, g) >= tau for g in P)]
            minimal = [P for P in good if not any(Q != P and _refines(Q, P) for Q in good)]
            found, _ = _minimal_partitions_exhaustive(
                torch.as_tensor(A, dtype=torch.float64), [[j] for j in range(J)], tau, 1e-12)
            self.assertEqual(sorted(_canon(P) for P in found), sorted(_canon(P) for P in minimal))

    def test_greedy_returns_a_reportable_partition(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            A = rng.standard_normal((40, 16))
            A[:, 3] = A[:, 1] + 0.05 * rng.standard_normal(40)
            A[:, 9] = A[:, 5] - A[:, 7] + 0.05 * rng.standard_normal(40)
            res = group_report(A, GroupingConfig(tau_theta=0.2, max_exhaustive_atoms=8))
            self.assertEqual(res.method, "greedy")
            self.assertTrue(all(s >= 0.2 for s in res.separations) or len(res.reported) == 1)
            self.assertTrue(any({1, 3} <= set(g) for g in res.reported))

    def test_declared_partition_is_respected(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((20, 5))
        res = group_report(A, GroupingConfig(tau_theta=0.01), declared=[[0, 1], [2], [3, 4]])
        for g in res.reported:
            self.assertTrue(all(any(set(d) <= set(g) or not set(d) & set(g) for d in [[0, 1], [2], [3, 4]])
                                for _ in [0]))
        self.assertEqual(res.reported, [[0, 1], [2], [3, 4]])


if __name__ == "__main__":
    unittest.main()


class TestLatentGrouping(unittest.TestCase):
    """Section 4.4: recovering the latent grouping from an operator observed with error."""

    @staticmethod
    def _latent(rng, theta, N=10):
        e = np.linalg.qr(rng.standard_normal((N, 4)))[0]
        mu0 = e[:, 0]
        mu1 = math.cos(theta) * e[:, 0] + math.sin(theta) * e[:, 1]
        mu2 = 1.3 * e[:, 0]
        mu3 = 0.5 * e[:, 0] + e[:, 2]
        return np.stack([mu0, mu1, mu2, mu3], axis=1)

    def test_margin_condition_gives_the_latent_grouping(self):
        from model.iasa.grouping import latent_grouping
        rng = np.random.default_rng(21)
        Hs = self._latent(rng, 0.6)
        m = min(s for k in range(1, 5) for S in __import__("itertools").combinations(range(4), k)
                for s in np.linalg.svd(Hs[:, list(S)], compute_uv=False) if s > 1e-9)
        D = rng.standard_normal(Hs.shape)
        D *= 0.2 * m / np.linalg.norm(D, 2)
        res = latent_grouping(Hs + D, tau=0.5 * m)
        self.assertEqual(res.candidates, [[[0, 2], [1], [3]]])

    def test_tie_break_returns_latent_grouping_when_gap_exceeds_twice_the_error(self):
        from model.iasa.grouping import largest_discarded, latent_grouping
        rng = np.random.default_rng(22)
        hits = 0
        for _ in range(300):
            Hs = self._latent(rng, rng.uniform(0.02, 0.4))
            sf = 10 ** rng.uniform(-3.5, -1.3)
            H = Hs + sf * rng.standard_normal(Hs.shape)
            tau = sf * (math.sqrt(10) + 2 + rng.uniform(1, 6))
            res = latent_grouping(H, tau, sigma_f=sf)
            L = [[0, 2], [1], [3]]
            if L not in res.candidates or len(res.candidates) < 2:
                continue
            if any(np.linalg.matrix_rank(Hs[:, g], 1e-9) != int((np.linalg.svd(H[:, g], compute_uv=False) > tau).sum()) for g in L):
                continue
            err = np.linalg.norm(H - Hs, 2)
            self.assertLessEqual(largest_discarded(H, L, tau), err + 1e-12)
            for P in res.candidates:
                if P == L:
                    continue
                dP = max((np.linalg.svd(Hs[:, g], compute_uv=False)[int((np.linalg.svd(H[:, g], compute_uv=False) > tau).sum())]
                          if int((np.linalg.svd(H[:, g], compute_uv=False) > tau).sum()) < len(g) else 0.0) for g in P)
                self.assertGreater(dP, 0)
                self.assertGreaterEqual(largest_discarded(H, P, tau), dP - err - 1e-12)
            hits += 1
        self.assertGreater(hits, 5)

    def test_p_value_is_one_below_the_noise_scale(self):
        from model.iasa.grouping import gaussian_discard_p_value
        self.assertEqual(gaussian_discard_p_value(0.01, 0.01, 10, 4), 1.0)
        self.assertLess(gaussian_discard_p_value(1.0, 0.01, 10, 4), 1e-6)
