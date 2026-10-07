"""Identifiable groupings of the projected operator for the signal target.

For a partition P of the columns of the projected operator A, the reported
quantity of a block g is its signal A_g c_g.  P is exactly identifiable when
col(A) is the direct sum of the block column spaces, equivalently when
rank(A) = sum_g rank(A_g).  The finest such partition is unique: it is the set of
connected components of the column matroid of A, which ``exact_components``
computes from one basis and the coefficients of the remaining columns in it.

At a stated noise level the relevant quantity is the separation of a block,
s(g) = sin of the smallest principal angle between col(A_g) and the span of all
other columns.  The least-squares or NNLS error of A_g c_g is at most
||P_A eps|| / s(g), so a partition is reportable at threshold tau_theta when
every block has s(g) >= tau_theta.  s(g) depends only on g and its complement,
not on how the complement is split.  Minimal reportable partitions need not be
unique, so ``group_report`` returns every minimal one it finds and picks the one
with the most blocks, then the largest smallest separation.

All numerics run in PyTorch (float64) on the input device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

Partition = list[list[int]]


@dataclass(frozen=True)
class GroupingConfig:
    tau_theta: float = math.sqrt(1.0 - 0.99**2)
    tau_num: float | None = None
    coefficient_tolerance: float = 1e-8
    max_exhaustive_atoms: int = 12


@dataclass
class GroupingResult:
    exact_components: Partition
    base_partition: Partition
    reported: Partition
    separations: list[float]
    minimal_partitions: list[Partition]
    method: str
    tau_theta: float
    tau_num: float
    rank: int
    group_ranks: list[int] = field(default_factory=list)
    tied_alternatives: list[Partition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_components": self.exact_components,
            "base_partition": self.base_partition,
            "reported": self.reported,
            "separations": self.separations,
            "minimal_partitions": self.minimal_partitions,
            "method": self.method,
            "tau_theta": self.tau_theta,
            "tau_num": self.tau_num,
            "rank": self.rank,
            "group_ranks": self.group_ranks,
            "tied_alternatives": self.tied_alternatives,
        }


def _as_matrix(A: Any) -> torch.Tensor:
    M = torch.as_tensor(A, dtype=torch.float64)
    if M.dim() != 2:
        raise ValueError("operator must be a 2-D matrix")
    return M


def numerical_tolerance(A: Any) -> float:
    M = _as_matrix(A)
    if M.numel() == 0:
        return 0.0
    sigma_1 = float(torch.linalg.matrix_norm(M, ord=2))
    return max(M.shape) * float(torch.finfo(M.dtype).eps) * sigma_1


def _orthonormal_basis(M: torch.Tensor, tol: float) -> torch.Tensor:
    if M.numel() == 0 or M.shape[1] == 0:
        return M.new_zeros((M.shape[0], 0))
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, S > tol]


def _rank(M: torch.Tensor, tol: float) -> int:
    return int(_orthonormal_basis(M, tol).shape[1])


def separation(A: Any, block: Sequence[int], tol: float | None = None) -> float:
    """sin of the smallest principal angle between col(A_block) and the rest.

    Returns 1 when either span is {0}: a block with no visible signal has signal
    zero, and a block with nothing outside it has nothing to be confused with.
    """
    M = _as_matrix(A)
    t = numerical_tolerance(M) if tol is None else float(tol)
    inside = sorted(set(int(j) for j in block))
    outside = [j for j in range(M.shape[1]) if j not in set(inside)]
    Qg = _orthonormal_basis(M[:, inside], t)
    Qr = _orthonormal_basis(M[:, outside], t)
    if Qg.shape[1] == 0 or Qr.shape[1] == 0:
        return 1.0
    cos_max = float(torch.linalg.matrix_norm(Qg.T @ Qr, ord=2))
    return math.sqrt(max(0.0, 1.0 - min(1.0, cos_max) ** 2))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def blocks(self) -> Partition:
        out: dict[int, list[int]] = {}
        for j in range(len(self.parent)):
            out.setdefault(self.find(j), []).append(j)
        return _canonical(list(out.values()))


def _canonical(P: Sequence[Sequence[int]]) -> Partition:
    return sorted((sorted(int(j) for j in g) for g in P if len(g)), key=lambda g: g[0])


def exact_components(A: Any, tol: float | None = None, coefficient_tolerance: float = 1e-8) -> Partition:
    """Finest exactly identifiable partition: the column-matroid components of A.

    Columns are added greedily to a basis when they raise the rank.  Every
    non-basis column is joined to the basis columns that carry a nonzero
    coefficient when it is written in that basis; the connected components of
    this graph are the matroid components.  A zero column is its own component.
    """
    M = _as_matrix(A)
    t = numerical_tolerance(M) if tol is None else float(tol)
    J = M.shape[1]
    uf = _UnionFind(J)
    basis: list[int] = []
    for j in range(J):
        if _rank(M[:, basis + [j]], t) > len(basis):
            basis.append(j)
    if basis:
        B = M[:, basis]
        norms = torch.linalg.vector_norm(B, dim=0)
        for e in range(J):
            if e in basis:
                continue
            col = M[:, e]
            if float(torch.linalg.vector_norm(col)) <= t:
                continue
            coef = torch.linalg.lstsq(B, col.unsqueeze(1)).solution.reshape(-1)
            weight = coef.abs() * norms / max(float(torch.linalg.vector_norm(col)), t)
            for bi in torch.nonzero(weight > coefficient_tolerance).reshape(-1).tolist():
                uf.union(e, basis[bi])
    return uf.blocks()


def join_partitions(P: Sequence[Sequence[int]], D: Sequence[Sequence[int]], J: int) -> Partition:
    """Finest common coarsening of two partitions of range(J)."""
    uf = _UnionFind(J)
    for part in (P, D):
        for g in part:
            g = list(g)
            for j in g[1:]:
                uf.union(g[0], j)
    return uf.blocks()


def is_exactly_identifiable(A: Any, P: Sequence[Sequence[int]], tol: float | None = None) -> bool:
    M = _as_matrix(A)
    t = numerical_tolerance(M) if tol is None else float(tol)
    return _rank(M, t) == sum(_rank(M[:, list(g)], t) for g in P)


def _atom_union(atoms: Sequence[Sequence[int]], mask: int) -> list[int]:
    out: list[int] = []
    for i, a in enumerate(atoms):
        if mask >> i & 1:
            out.extend(a)
    return out


def _minimal_partitions_exhaustive(
    M: torch.Tensor, atoms: Partition, tau: float, tol: float
) -> tuple[list[Partition], dict[int, float]]:
    """All minimal reportable partitions whose blocks are unions of atoms.

    A partition is reportable when each block is a good set (separation >= tau).
    It is minimal exactly when each block is an atomic good set, one that cannot
    itself be partitioned into two or more good sets.
    """
    m = len(atoms)
    full = (1 << m) - 1
    sep: dict[int, float] = {}
    good = [False] * (1 << m)
    for mask in range(1, 1 << m):
        if mask == full:
            sep[mask] = 1.0
        elif (full ^ mask) in sep:
            sep[mask] = sep[full ^ mask]
        else:
            sep[mask] = separation(M, _atom_union(atoms, mask), tol)
        good[mask] = sep[mask] >= tau

    def lowest(mask: int) -> int:
        return (mask & -mask).bit_length() - 1

    def subsets_with_low(mask: int):
        low = 1 << lowest(mask)
        rest = mask ^ low
        sub = rest
        while True:
            yield sub | low
            if sub == 0:
                break
            sub = (sub - 1) & rest

    partitionable = [False] * (1 << m)
    partitionable[0] = True
    for mask in range(1, 1 << m):
        partitionable[mask] = any(good[h] and partitionable[mask ^ h] for h in subsets_with_low(mask))

    def splittable(mask: int) -> bool:
        return any(h != mask and good[h] and partitionable[mask ^ h] for h in subsets_with_low(mask))

    atomic = [good[mask] and not splittable(mask) for mask in range(1 << m)]
    found: list[Partition] = []

    def extend(remaining: int, chosen: list[int]) -> None:
        if remaining == 0:
            found.append(_canonical([_atom_union(atoms, h) for h in chosen]))
            return
        for h in subsets_with_low(remaining):
            if atomic[h] and partitionable[remaining ^ h]:
                extend(remaining ^ h, chosen + [h])

    extend(full, [])
    return found, sep


def _greedy_partition(M: torch.Tensor, atoms: Partition, tau: float, tol: float) -> Partition:
    """Merge the least separated block with the partner that maximizes the merged separation."""
    blocks = [list(a) for a in atoms]
    seps = [separation(M, g, tol) for g in blocks]
    while len(blocks) > 1 and min(seps) < tau:
        i = min(range(len(blocks)), key=lambda k: seps[k])
        best_k, best_s = None, -1.0
        for k in range(len(blocks)):
            if k == i:
                continue
            s = separation(M, blocks[i] + blocks[k], tol) if len(blocks) > 2 else 1.0
            if s > best_s:
                best_k, best_s = k, s
        merged = blocks[i] + blocks[best_k]
        blocks = [g for k, g in enumerate(blocks) if k not in (i, best_k)] + [merged]
        seps = [s for k, s in enumerate(seps) if k not in (i, best_k)] + [best_s]
    return _canonical(blocks)


def group_report(
    A: Any,
    config: GroupingConfig = GroupingConfig(),
    declared: Sequence[Sequence[int]] | None = None,
) -> GroupingResult:
    """Reported partition at separation threshold ``config.tau_theta``.

    ``declared`` is a partition the report must respect (for example one block
    per source when a source has several basis columns); reported blocks are
    unions of the blocks of the join of the exact components with it.
    """
    M = _as_matrix(A)
    J = M.shape[1]
    t = numerical_tolerance(M) if config.tau_num is None else float(config.tau_num)
    comps = exact_components(M, t, config.coefficient_tolerance)
    base = join_partitions(comps, declared, J) if declared is not None else comps
    tau = float(config.tau_theta)
    if len(base) <= config.max_exhaustive_atoms:
        minimal, _ = _minimal_partitions_exhaustive(M, base, tau, t)
        method = "exhaustive"

        def score(P: Partition) -> tuple[int, float]:
            return (len(P), min(separation(M, g, t) for g in P))

        scores = {tuple(map(tuple, P)): score(P) for P in minimal}
        reported = max(minimal, key=lambda P: scores[tuple(map(tuple, P))])
        best = scores[tuple(map(tuple, reported))]
        tied = [P for P in minimal if P != reported and scores[tuple(map(tuple, P))] == best]
    else:
        reported = _greedy_partition(M, base, tau, t)
        minimal = [reported]
        tied = []
        method = "greedy"
    seps = [separation(M, g, t) for g in reported]
    return GroupingResult(
        exact_components=comps,
        base_partition=base,
        reported=reported,
        separations=seps,
        minimal_partitions=minimal,
        method=method,
        tau_theta=tau,
        tau_num=t,
        rank=_rank(M, t),
        group_ranks=[_rank(M[:, g], t) for g in reported],
        tied_alternatives=tied,
    )


def group_signals(A: Any, c: Any, P: Sequence[Sequence[int]]) -> list[torch.Tensor]:
    """A_g c_g for every block of P."""
    M = _as_matrix(A)
    v = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
    return [M[:, list(g)] @ v[list(g)] for g in P]


def group_error_bound(sigma: float, group_rank: int, separation_value: float, delta: float = 0.05) -> float:
    """Least-squares bound sigma (sqrt(r_g) + sqrt(2 log(1/delta))) / s_g, valid with probability 1 - delta."""
    if separation_value <= 0:
        return math.inf
    return sigma * (math.sqrt(group_rank) + math.sqrt(2.0 * math.log(1.0 / delta))) / separation_value


# --------------------------------------------------------------------------- #
# Uncertain fingerprints: estimating the latent grouping (paper Section 4.4)   #
# --------------------------------------------------------------------------- #
@dataclass
class LatentGroupingResult:
    estimate: Partition
    candidates: list[Partition]
    discarded: list[float]
    p_values: list[float]
    tau: float
    sigma_f: float | None


def thresholded_rank(M: Any, tau: float) -> int:
    """Number of singular values of M strictly above tau."""
    S = torch.linalg.svdvals(_as_matrix(M))
    return int((S > float(tau)).sum())


def is_tau_identifiable(A: Any, P: Sequence[Sequence[int]], tau: float) -> bool:
    M = _as_matrix(A)
    return sum(thresholded_rank(M[:, list(g)], tau) for g in P) == thresholded_rank(M, tau)


def largest_discarded(A: Any, P: Sequence[Sequence[int]], tau: float) -> float:
    """D(P): the largest singular value that P discards, max_g sigma_{k_g+1}(A_g) with k_g = rank_tau(A_g)."""
    M = _as_matrix(A)
    out = 0.0
    for g in P:
        S = torch.linalg.svdvals(M[:, list(g)])
        k = int((S > float(tau)).sum())
        if k < S.numel():
            out = max(out, float(S[k]))
    return out


def gaussian_discard_p_value(D: float, sigma_f: float, N: int, J: int) -> float:
    """Upper bound on P(D(P) >= D) when P is an exact latent grouping and the operator error is
    P_Q^perp times an N x J matrix with independent N(0, sigma_f^2) entries."""
    z = D / sigma_f - math.sqrt(N) - math.sqrt(J)
    return 1.0 if z <= 0 else min(1.0, 2.0 * math.exp(-0.5 * z * z))


def _partitions_of(items: list[int]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions_of(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def latent_grouping(
    A: Any,
    tau: float,
    sigma_f: float | None = None,
    declared: Sequence[Sequence[int]] | None = None,
    max_atoms: int = 10,
) -> LatentGroupingResult:
    """Estimate the latent grouping of an operator observed with error.

    The candidates are the minimal tau-identifiable partitions whose blocks are unions of
    the declared blocks (one block per column if none are declared).  The estimate is the
    candidate with the smallest largest-discarded singular value D(P), which is the
    candidate with the largest Gaussian p-value when sigma_f is given.
    """
    M = _as_matrix(A)
    J = M.shape[1]
    atoms = _canonical(declared) if declared is not None else [[j] for j in range(J)]
    if len(atoms) > max_atoms:
        raise ValueError(f"{len(atoms)} atoms exceed max_atoms={max_atoms}; exhaustive search only")
    passing = []
    for Pa in _partitions_of(list(range(len(atoms)))):
        P = _canonical([[j for a in block for j in atoms[a]] for block in Pa])
        if is_tau_identifiable(M, P, tau):
            passing.append(P)

    def refines(P: Partition, Q: Partition) -> bool:
        return all(any(set(g) <= set(h) for h in Q) for g in P)

    minimal = [P for P in passing if not any(Q != P and refines(Q, P) for Q in passing)]
    discarded = [largest_discarded(M, P, tau) for P in minimal]
    N = M.shape[0]
    pvals = ([gaussian_discard_p_value(d, sigma_f, N, J) for d in discarded]
             if sigma_f else [float("nan")] * len(minimal))
    best = min(range(len(minimal)), key=lambda i: discarded[i])
    return LatentGroupingResult(estimate=minimal[best], candidates=minimal, discarded=discarded,
                                p_values=pvals, tau=float(tau), sigma_f=sigma_f)


__all__ = [
    "GroupingConfig",
    "LatentGroupingResult",
    "gaussian_discard_p_value",
    "is_tau_identifiable",
    "largest_discarded",
    "latent_grouping",
    "thresholded_rank",
    "GroupingResult",
    "exact_components",
    "group_error_bound",
    "group_report",
    "group_signals",
    "is_exactly_identifiable",
    "join_partitions",
    "numerical_tolerance",
    "separation",
]
