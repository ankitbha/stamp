"""Identifiability diagnostics for the projected source--basis response (Task 7).

All numerics run in PyTorch on the input device/dtype. NumPy appears only at the
JSON/serialization boundary. Diagnostics are computed on the reduced admissible
column set (after an optional predeclared fixed-zero mask); guarantees are uniform
over that set. There is no tangent-cone or fitted-active-set path: a coefficient
that turns out small after fitting never changes the mask or any diagnostic.

Symbols follow paper/5.theory.tex: padded minimum singular value ``sigma_J``,
numerical tolerance/rank ``tau_num``/``r_num``, condition number ``kappa``,
noise-dependent effective rank ``r_eff(tau_sigma)``, visibility ``v_j``, weak set
``W``, pairwise coherence ``rho_ij``, ray distance ``sqrt(1 - rho^2)``, background
absorption ``a_j``, and the eligible ambiguity set ``A``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from model.iasa.backend import dtype_name, runtime_provenance, to_numpy


@dataclass(frozen=True)
class DiagnosticsConfig:
    fixed_zero_indices: tuple[int, ...] = ()
    tau_sigma: float | None = None
    tau_v: float = 0.0
    tau_rho: float = 0.99


@dataclass
class DiagnosticsResult:
    singular_values: torch.Tensor
    sigma_1: float
    sigma_J: float
    sigma_min_positive: float | None
    numerical_rank: int
    numerical_tolerance: float
    condition_number: float | None
    condition_status: str
    effective_rank: int
    effective_rank_threshold: float
    noise_threshold_provided: bool
    visibility: torch.Tensor
    weak_flags: list[bool]
    weak_set: list[int]
    coherence: torch.Tensor
    ray_distance: torch.Tensor
    background_absorption: torch.Tensor | None
    ambiguous_pairs: list[dict[str, Any]]
    cross_source_max_coherence: dict[str, Any] | None
    cross_source_min_ray_distance: dict[str, Any] | None
    per_source_weak: dict[str, Any]
    perturbation_sensitivity: dict[str, Any]
    original_to_reduced: dict[int, int]
    reduced_to_original: list[int]
    column_index: list[dict[str, Any]]
    config: dict[str, Any]
    warnings: list[str]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "singular_values": _to_list(self.singular_values),
            "sigma_1": self.sigma_1,
            "sigma_J": self.sigma_J,
            "sigma_min_positive": self.sigma_min_positive,
            "numerical_rank": self.numerical_rank,
            "numerical_tolerance": self.numerical_tolerance,
            "condition_number": self.condition_number,
            "condition_status": self.condition_status,
            "effective_rank": self.effective_rank,
            "effective_rank_threshold": self.effective_rank_threshold,
            "noise_threshold_provided": self.noise_threshold_provided,
            "visibility": _to_list(self.visibility),
            "weak_flags": list(self.weak_flags),
            "weak_set": list(self.weak_set),
            "coherence": _matrix_with_nulls(self.coherence),
            "ray_distance": _matrix_with_nulls(self.ray_distance),
            "background_absorption": (
                None if self.background_absorption is None else _vector_with_nulls(self.background_absorption)
            ),
            "ambiguous_pairs": self.ambiguous_pairs,
            "cross_source_max_coherence": self.cross_source_max_coherence,
            "cross_source_min_ray_distance": self.cross_source_min_ray_distance,
            "per_source_weak": self.per_source_weak,
            "perturbation_sensitivity": self.perturbation_sensitivity,
            "original_to_reduced": {str(k): v for k, v in self.original_to_reduced.items()},
            "reduced_to_original": list(self.reduced_to_original),
            "column_index": self.column_index,
            "config": self.config,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }

    def human_readable_table(self) -> str:
        lines = [
            f"identifiability: sigma_J={self.sigma_J:.6g} sigma_1={self.sigma_1:.6g} "
            f"cond={self.condition_status}({self.condition_number}) "
            f"r_num={self.numerical_rank}/{len(self.reduced_to_original)} r_eff={self.effective_rank}",
            "col  source                  basis                 visibility  absorption  weak",
        ]
        absorption = None if self.background_absorption is None else _vector_with_nulls(self.background_absorption)
        vis = _to_list(self.visibility)
        for reduced_idx, orig in enumerate(self.reduced_to_original):
            col = self.column_index[orig]
            a = "n/a" if absorption is None or absorption[reduced_idx] is None else f"{absorption[reduced_idx]:.4f}"
            lines.append(
                f"{orig:<4d} {str(col.get('source_name')):<22.22} {str(col.get('basis_name')):<20.20} "
                f"{vis[reduced_idx]:<11.4g} {a:<11} {self.weak_flags[reduced_idx]}"
            )
        return "\n".join(lines)


def _to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in to_numpy(tensor).reshape(-1).tolist()]


def _vector_with_nulls(tensor: torch.Tensor) -> list[float | None]:
    return [None if math.isnan(float(v)) else float(v) for v in to_numpy(tensor).reshape(-1).tolist()]


def _matrix_with_nulls(tensor: torch.Tensor) -> list[list[float | None]]:
    rows = to_numpy(tensor).tolist()
    return [[None if math.isnan(float(v)) else float(v) for v in row] for row in rows]


def _validate_fixed_zero(fixed_zero_indices: Sequence[int], J_full: int) -> list[int]:
    seen: set[int] = set()
    for idx in fixed_zero_indices:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError("fixed_zero_indices must be integers")
        if idx < 0 or idx >= J_full:
            raise ValueError(f"fixed_zero index {idx} out of range for {J_full} columns")
        if idx in seen:
            raise ValueError(f"duplicate fixed_zero index {idx}")
        seen.add(idx)
    return sorted(i for i in range(J_full) if i not in seen)


def diagnose_identifiability(
    H_tilde: torch.Tensor,
    column_index: Sequence[dict[str, Any]],
    *,
    H_removed: torch.Tensor | None = None,
    H_lag: torch.Tensor | None = None,
    config: DiagnosticsConfig | None = None,
) -> DiagnosticsResult:
    cfg = config or DiagnosticsConfig()
    if not isinstance(H_tilde, torch.Tensor) or H_tilde.ndim != 2:
        raise ValueError("H_tilde must be a 2-D torch.Tensor [N, J]")
    if not torch.isfinite(H_tilde).all():
        raise ValueError("H_tilde must contain only finite values")
    J_full = H_tilde.shape[1]
    if len(column_index) != J_full:
        raise ValueError("column_index length must match H_tilde columns")
    device = H_tilde.device
    dtype = H_tilde.dtype
    if cfg.tau_v < 0 or cfg.tau_rho < 0 or cfg.tau_rho > 1:
        raise ValueError("tau_v must be >= 0 and tau_rho in [0, 1]")
    if cfg.tau_sigma is not None and cfg.tau_sigma < 0:
        raise ValueError("tau_sigma must be nonnegative when provided")

    kept = _validate_fixed_zero(cfg.fixed_zero_indices, J_full)
    reduced_to_original = list(kept)
    original_to_reduced = {orig: reduced for reduced, orig in enumerate(kept)}
    for orig in range(J_full):
        original_to_reduced.setdefault(orig, -1)
    reduced_column_index = [dict(column_index[orig]) for orig in kept]

    index = torch.tensor(kept, dtype=torch.long, device=device)
    Ht = H_tilde.index_select(1, index) if kept else H_tilde[:, :0]
    N = Ht.shape[0]
    J = Ht.shape[1]
    warnings: list[str] = []

    eps = float(torch.finfo(dtype).eps)
    if J == 0:
        gram = None
        singular_values = torch.zeros(0, dtype=dtype, device=device)
        sigma_1 = 0.0
        sigma_J = 0.0
    else:
        # Gram (H_tilde^T H_tilde) supplies the visibility and coherence entries.
        gram = Ht.transpose(0, 1) @ Ht
        # The J singular values are those of H_tilde (equivalently the square roots
        # of the Gram eigenvalues), computed with a backward-stable SVD whose
        # numerical floor is ~eps*sigma_1 -- so tau_num = max(N,J)*eps*sigma_1
        # correctly detects rank deficiency. Forming sqrt(eigvalsh(Gram)) instead
        # would raise that floor to ~sqrt(eps)*sigma_1 and mask deficiency. When
        # J > N the spectrum is zero-padded to exactly J entries.
        sv = torch.linalg.svdvals(Ht)  # descending, length min(N, J)
        if sv.shape[0] < J:
            singular_values = torch.cat(
                [sv, torch.zeros(J - sv.shape[0], dtype=dtype, device=device)]
            )
        else:
            singular_values = sv
        sigma_1 = float(singular_values[0])
        sigma_J = float(singular_values[-1])

    tau_num = float(max(N, J) * eps * sigma_1) if J else 0.0
    numerical_rank = int(torch.count_nonzero(singular_values > tau_num)) if J else 0
    positive = singular_values[singular_values > 0] if J else singular_values
    sigma_min_positive = float(positive.min()) if positive.numel() else None

    if J and numerical_rank == J and sigma_J > 0:
        condition_number: float | None = sigma_1 / sigma_J
        condition_status = "finite"
    else:
        condition_number = None
        condition_status = "infinite"

    if cfg.tau_sigma is None:
        effective_rank_threshold = tau_num
        noise_threshold_provided = False
    else:
        effective_rank_threshold = float(cfg.tau_sigma)
        noise_threshold_provided = True
    effective_rank = int(torch.count_nonzero(singular_values > effective_rank_threshold)) if J else 0

    if J:
        visibility = torch.sqrt(torch.clamp(torch.diagonal(gram), min=0.0))
    else:
        visibility = torch.zeros(0, dtype=dtype, device=device)
    weak_mask = visibility <= cfg.tau_v
    weak_flags = [bool(x) for x in weak_mask.tolist()]
    weak_set = [reduced_to_original[j] for j in range(J) if weak_flags[j]]

    # Pairwise coherence and ray distance; ineligible entries are NaN sentinels.
    coherence = torch.full((J, J), float("nan"), dtype=dtype, device=device)
    ray_distance = torch.full((J, J), float("nan"), dtype=dtype, device=device)
    if J:
        eligible = (~weak_mask) & (visibility > 0)
        eligible_pair = eligible[:, None] & eligible[None, :]
        denom = visibility[:, None] * visibility[None, :]
        safe = eligible_pair & (denom > 0)
        rho = torch.where(safe, torch.abs(gram) / torch.where(denom > 0, denom, torch.ones_like(denom)), coherence)
        rho = torch.where(safe, torch.clamp(rho, 0.0, 1.0), coherence)
        coherence = rho
        ray_distance = torch.where(safe, torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0)), ray_distance)

    # Background absorption a_j = ||H_removed_j|| / ||H_lag_j|| on kept columns.
    background_absorption: torch.Tensor | None = None
    if H_lag is not None and H_removed is not None and J:
        H_lag_kept = H_lag.index_select(1, index).to(device=device, dtype=dtype)
        H_removed_kept = H_removed.index_select(1, index).to(device=device, dtype=dtype)
        lag_norm = torch.linalg.vector_norm(H_lag_kept, dim=0)
        removed_norm = torch.linalg.vector_norm(H_removed_kept, dim=0)
        abs_tol = max(tau_num, eps)
        background_absorption = torch.where(
            lag_norm > abs_tol,
            removed_norm / torch.where(lag_norm > abs_tol, lag_norm, torch.ones_like(lag_norm)),
            torch.full_like(lag_norm, float("nan")),
        )

    # Eligible ambiguity set and cross-source summaries.
    ambiguous_pairs: list[dict[str, Any]] = []
    cross_source_max_coherence: dict[str, Any] | None = None
    cross_source_min_ray_distance: dict[str, Any] | None = None
    best_coh = -1.0
    best_ray = math.inf
    for i in range(J):
        for j in range(i + 1, J):
            rho_ij = float(coherence[i, j])
            if math.isnan(rho_ij):
                continue
            ray_ij = float(ray_distance[i, j])
            oi = reduced_to_original[i]
            oj = reduced_to_original[j]
            ci = reduced_column_index[i]
            cj = reduced_column_index[j]
            record = {
                "i": oi,
                "j": oj,
                "coherence": rho_ij,
                "ray_distance": ray_ij,
                "source_i": ci.get("source_name"),
                "basis_i": ci.get("basis_name"),
                "source_j": cj.get("source_name"),
                "basis_j": cj.get("basis_name"),
            }
            if rho_ij > cfg.tau_rho:
                ambiguous_pairs.append(record)
            if ci.get("source_index") != cj.get("source_index"):
                if rho_ij > best_coh:
                    best_coh = rho_ij
                    cross_source_max_coherence = record
                if ray_ij < best_ray:
                    best_ray = ray_ij
                    cross_source_min_ray_distance = record

    # Per-source weak-basis flags.
    per_source_weak: dict[str, Any] = {}
    for j in range(J):
        col = reduced_column_index[j]
        name = str(col.get("source_name"))
        entry = per_source_weak.setdefault(name, {"weak_bases": [], "all_weak": True})
        if weak_flags[j]:
            entry["weak_bases"].append(col.get("basis_name"))
        else:
            entry["all_weak"] = False

    inv_sigma_J = math.inf if sigma_J <= 0 else 1.0 / sigma_J
    perturbation_sensitivity = {
        "inverse_sigma_J": {
            "status": "infinite" if math.isinf(inv_sigma_J) else "finite",
            "value": None if math.isinf(inv_sigma_J) else inv_sigma_J,
        },
        "condition_number": {
            "status": condition_status,
            "value": condition_number,
        },
    }

    if J and numerical_rank < J:
        warnings.append("rank_deficient: sigma_J == 0; at least one coefficient direction is unresolved")
    if weak_set:
        warnings.append(f"weak_coefficients: {len(weak_set)} coefficient fingerprint(s) below tau_v")
    if background_absorption is None:
        warnings.append("background_absorption_unavailable: H_lag/H_removed not supplied")

    metadata = {
        **runtime_provenance(device, dtype),
        "N": int(N),
        "J": int(J),
        "J_full": int(J_full),
        "response_dtype": dtype_name(dtype),
        "uniform_over_admissible_set": True,
    }
    config_record = {
        "fixed_zero_indices": list(cfg.fixed_zero_indices),
        "tau_sigma": cfg.tau_sigma,
        "tau_v": cfg.tau_v,
        "tau_rho": cfg.tau_rho,
    }
    return DiagnosticsResult(
        singular_values=singular_values,
        sigma_1=sigma_1,
        sigma_J=sigma_J,
        sigma_min_positive=sigma_min_positive,
        numerical_rank=numerical_rank,
        numerical_tolerance=tau_num,
        condition_number=condition_number,
        condition_status=condition_status,
        effective_rank=effective_rank,
        effective_rank_threshold=effective_rank_threshold,
        noise_threshold_provided=noise_threshold_provided,
        visibility=visibility,
        weak_flags=weak_flags,
        weak_set=weak_set,
        coherence=coherence,
        ray_distance=ray_distance,
        background_absorption=background_absorption,
        ambiguous_pairs=ambiguous_pairs,
        cross_source_max_coherence=cross_source_max_coherence,
        cross_source_min_ray_distance=cross_source_min_ray_distance,
        per_source_weak=per_source_weak,
        perturbation_sensitivity=perturbation_sensitivity,
        original_to_reduced=original_to_reduced,
        reduced_to_original=reduced_to_original,
        column_index=[dict(col) for col in column_index],
        config=config_record,
        warnings=warnings,
        metadata=metadata,
    )


def diagnose_projection(projection_result: Any, config: DiagnosticsConfig | None = None) -> DiagnosticsResult:
    H_tilde = projection_result.H_tilde
    H_removed = getattr(projection_result, "H_removed", None)
    H_lag = None
    if H_removed is not None:
        H_lag = H_tilde + H_removed
    return diagnose_identifiability(
        H_tilde, projection_result.column_index, H_removed=H_removed, H_lag=H_lag, config=config
    )


def _connected_components(num_sources: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(num_sources))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[min(ra, rb)] = max(ra, rb)  # deterministic
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for node in range(num_sources):
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(members) for members in groups.values()), key=lambda g: g[0])


def summarize_wind_ensemble(
    results: Sequence[DiagnosticsResult],
    *,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    window_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("summarize_wind_ensemble requires at least one DiagnosticsResult")
    reference = results[0].reduced_to_original
    ref_columns = results[0].column_index
    for r in results:
        if r.reduced_to_original != reference:
            raise ValueError("all ensemble windows must share the same reduced column mapping")
        if r.column_index != ref_columns:
            raise ValueError("all ensemble windows must share the same column_index")
    n_windows = len(results)
    device = results[0].singular_values.device if results[0].singular_values.numel() else torch.device("cpu")

    sigma_J_values = torch.tensor([r.sigma_J for r in results], dtype=torch.float64, device=device)
    q = torch.tensor(list(quantiles), dtype=torch.float64, device=device)
    sigma_J_quantiles = {
        f"q{int(round(quant * 100)):02d}": float(val)
        for quant, val in zip(quantiles, torch.quantile(sigma_J_values, q).tolist())
    }

    J = len(reference)
    full_num = sum(1 for r in results if r.numerical_rank == J)
    full_eff = sum(1 for r in results if r.effective_rank == J)

    weak_counts = [0] * J
    for r in results:
        for reduced_idx in range(J):
            if r.weak_flags[reduced_idx]:
                weak_counts[reduced_idx] += 1
    weak_probabilities = {
        str(reference[reduced_idx]): weak_counts[reduced_idx] / n_windows for reduced_idx in range(J)
    }

    source_names = [str(col.get("source_name")) for col in ref_columns]
    source_order: list[str] = []
    for name in source_names:
        if name not in source_order:
            source_order.append(name)
    source_to_node = {name: node for node, name in enumerate(source_order)}

    pair_counts: dict[tuple[str, str], int] = {}
    component_counts: dict[tuple[str, ...], int] = {}
    for r in results:
        edges: set[tuple[int, int]] = set()
        window_pairs: set[tuple[str, str]] = set()
        for pair in r.ambiguous_pairs:
            si, sj = str(pair["source_i"]), str(pair["source_j"])
            if si == sj:
                continue
            key = tuple(sorted((si, sj)))
            window_pairs.add(key)
            edges.add(tuple(sorted((source_to_node[si], source_to_node[sj]))))
        for key in window_pairs:
            pair_counts[key] = pair_counts.get(key, 0) + 1
        components = _connected_components(len(source_order), sorted(edges))
        for comp in components:
            comp_key = tuple(source_order[node] for node in comp)
            component_counts[comp_key] = component_counts.get(comp_key, 0) + 1

    return {
        "n_windows": n_windows,
        "window_ids": list(window_ids) if window_ids is not None else list(range(n_windows)),
        "quantiles": list(quantiles),
        "sigma_J_quantiles": sigma_J_quantiles,
        "prob_full_numerical_rank": full_num / n_windows,
        "prob_full_effective_rank": full_eff / n_windows,
        "coefficient_weak_probabilities": weak_probabilities,
        "source_pair_ambiguity_probabilities": {
            f"{a}|{b}": count / n_windows for (a, b), count in sorted(pair_counts.items())
        },
        "report_component_frequencies": {
            "|".join(comp): count / n_windows for comp, count in sorted(component_counts.items())
        },
        "reduced_to_original": list(reference),
    }


__all__ = [
    "DiagnosticsConfig",
    "DiagnosticsResult",
    "diagnose_identifiability",
    "diagnose_projection",
    "summarize_wind_ensemble",
]
