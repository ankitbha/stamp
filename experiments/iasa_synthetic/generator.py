"""Planted operators for the synthetic study.

Each planted block g spans its own subspace of dimension r_g.  The block
subspaces are drawn at random inside an ambient space of dimension
sum_g r_g + slack and embedded in R^N, so a small slack gives small angles
between blocks.  Inside a block the columns are either exactly dependent
(k columns in general position in a (k-1)-dimensional space, so no pair is
nearly parallel), nearly dependent in the same way (``near_circuit``: the k-th
coordinate has size ``omega``), or nearly dependent with spread ``omega``
around a common direction (``near``: every pair is nearly parallel).  In both
near modes the block's smallest singular value is proportional to omega.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantedProblem:
    A_raw: np.ndarray
    Q: np.ndarray
    A: np.ndarray
    blocks: list[list[int]]
    c: np.ndarray


def _orthonormal(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q[:, :k]


def _block_coordinates(rng: np.random.Generator, k: int, mode: str, omega: float) -> np.ndarray:
    """Coordinates of a block's k columns in its own subspace (columns of the result)."""
    if k == 1:
        return np.ones((1, 1))
    if mode == "exact":
        W = rng.standard_normal((k - 1, k))
    elif mode == "near_circuit":
        W = np.concatenate([rng.standard_normal((k - 1, k)), omega * rng.standard_normal((1, k))], axis=0)
    elif mode == "near":
        W = np.zeros((k, k))
        W[0, :] = 1.0
        W[1:, :] = omega * rng.standard_normal((k - 1, k))
    else:
        raise ValueError(f"unknown block mode {mode!r}")
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def planted_problem(
    rng: np.random.Generator,
    N: int,
    sizes: list[int],
    *,
    mode: str = "near",
    omega: float = 1e-2,
    slack: int = 0,
    nuisance_dim: int = 0,
    shuffle: bool = True,
) -> PlantedProblem:
    coords = [_block_coordinates(rng, k, mode, omega) for k in sizes]
    dims = [W.shape[0] for W in coords]
    D = sum(dims) + int(slack)
    if D > N:
        raise ValueError("ambient dimension exceeds N")
    embed = _orthonormal(rng, N, D)
    cols, start = [], 0
    for W in coords:
        U = _orthonormal(rng, D, W.shape[0])
        cols.append(embed @ U @ W)
    A_raw = np.concatenate(cols, axis=1)
    J = A_raw.shape[1]
    perm = rng.permutation(J) if shuffle else np.arange(J)
    A_raw = A_raw[:, perm]
    where = {int(old): new for new, old in enumerate(perm)}
    blocks, start = [], 0
    for k in sizes:
        blocks.append(sorted(where[j] for j in range(start, start + k)))
        start += k
    blocks.sort(key=lambda g: g[0])
    Q = rng.standard_normal((N, nuisance_dim)) if nuisance_dim else np.zeros((N, 0))
    A = project_out(A_raw, Q)
    c = rng.uniform(0.5, 1.5, size=J)
    return PlantedProblem(A_raw=A_raw, Q=Q, A=A, blocks=blocks, c=c)


def project_out(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[1] == 0:
        return M.copy()
    q, _ = np.linalg.qr(Q)
    return M - q @ (q.T @ M)


def observe(rng: np.random.Generator, problem: PlantedProblem, sigma: float) -> np.ndarray:
    """Projected observations P_Q^perp (A_raw c + Q gamma + eps)."""
    N = problem.A_raw.shape[0]
    gamma = rng.standard_normal(problem.Q.shape[1])
    y = problem.A_raw @ problem.c + problem.Q @ gamma + sigma * rng.standard_normal(N)
    return project_out(y[:, None], problem.Q)[:, 0]
