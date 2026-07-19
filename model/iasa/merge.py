"""Merge recommendation system for identifiable source resolution (Task 9).

Consumes the coefficient-level identifiability diagnostics (Task 7) and turns the
eligible cross-source coefficient-pair ambiguities into a source ambiguity graph,
whose deterministically ordered connected components are the recommended
conservative report groups. The fine-resolution coefficient fit (Task 8) is never
replaced by a grouped refit: group activity and fitted sensor contributions are
pure sums of member sources.

Symbols follow paper/5.theory.tex subsec:source_merging: the source ambiguity
graph ``E_src`` (eq. source_ambiguity_graph), connected report groups ``G``, and
the grouped reports ``theta_{G_a}(t) = sum_{k in G_a} theta_k(t)`` and
``Y_{G_a} = sum_{k in G_a} sum_b h_kb^lag c_kb`` (eq. grouped_reports). Components
are conservative reporting units, never a claim of the globally finest
identifiable partition; an A-B-C edge chain yields one component while both
trigger edges are retained.

All numerics run in PyTorch; NumPy appears only at the JSON/serialization
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from model.iasa.backend import runtime_provenance, to_numpy
from model.iasa.fit import summarize_report_groups


@dataclass(frozen=True)
class MergeConfig:
    # Reserved for the optional constrained-refinement acceptance threshold
    # (tau_rho_ref, paper eq. refinement_coherence_check). Task 9 consumes the
    # ambiguity set exactly as Task 7 produced it, so no threshold is re-applied.
    tau_rho_ref: float | None = None


@dataclass
class MergeResult:
    report_components: list[dict[str, Any]]
    source_edges: list[dict[str, Any]]
    weak_flags: dict[str, Any]
    global_unresolved_warning: bool
    source_level_activity_summaries: dict[str, Any] | None
    grouped_activity: dict[str, Any] | None
    grouped_sensor_contribution: list[dict[str, Any]] | None
    resolution: dict[str, Any]
    warnings: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "report_components": self.report_components,
            "source_edges": self.source_edges,
            "weak_flags": self.weak_flags,
            "global_unresolved_warning": self.global_unresolved_warning,
            "source_level_activity_summaries": self.source_level_activity_summaries,
            "grouped_activity": self.grouped_activity,
            "grouped_sensor_contribution": self.grouped_sensor_contribution,
            "resolution": self.resolution,
            "warnings": list(self.warnings),
            "config": self.config,
            "metadata": self.metadata,
        }


def _connected_components(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Deterministic connected components over the given nodes and edges.

    Nodes are the original source indices; components are ordered by their
    smallest member and members are sorted ascending.
    """

    parent = {n: n for n in nodes}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            hi, lo = max(ra, rb), min(ra, rb)
            parent[hi] = lo  # attach to the smaller root for determinism
    groups: dict[int, list[int]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(members) for members in groups.values()), key=lambda g: g[0])


def recommend_merges(
    diagnostics: Any,
    *,
    fit: Any = None,
    H_tilde: torch.Tensor | None = None,
    config: MergeConfig | None = None,
) -> MergeResult:
    cfg = config or MergeConfig()
    column_index = diagnostics.column_index
    if not column_index:
        raise ValueError("diagnostics.column_index must be non-empty")

    # Vertices: distinct source indices in original order.
    source_name_by_index: dict[int, str] = {}
    for col in column_index:
        k = int(col["source_index"])
        source_name_by_index.setdefault(k, str(col.get("source_name")))
    nodes = sorted(source_name_by_index)

    # Build the source ambiguity graph from eligible cross-source coefficient
    # pairs. ambiguous_pairs are already eligible (non-weak) with rho > tau_rho.
    edge_records: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in diagnostics.ambiguous_pairs:
        oi, oj = int(pair["i"]), int(pair["j"])
        ki = int(column_index[oi]["source_index"])
        kj = int(column_index[oj]["source_index"])
        if ki == kj:
            continue  # same source: within-source, not a source edge
        key = (min(ki, kj), max(ki, kj))
        coherence = float(pair["coherence"])
        ray = float(pair["ray_distance"])
        current = edge_records.get(key)
        trigger = {
            "col_i": oi,
            "col_j": oj,
            "source_i": pair.get("source_i"),
            "basis_i": pair.get("basis_i"),
            "source_j": pair.get("source_j"),
            "basis_j": pair.get("basis_j"),
            "coherence": coherence,
            "ray_distance": ray,
        }
        # Trigger = the coefficient pair attaining the max coherence (== min ray
        # distance). Ties break by (col_i, col_j) order for determinism.
        if current is None or coherence > current["max_coherence"] or (
            coherence == current["max_coherence"] and (oi, oj) < (current["trigger"]["col_i"], current["trigger"]["col_j"])
        ):
            edge_records[key] = {
                "sources": key,
                "source_names": (source_name_by_index[key[0]], source_name_by_index[key[1]]),
                "max_coherence": coherence,
                "min_ray_distance": ray,
                "trigger": trigger,
            }
        else:
            current["min_ray_distance"] = min(current["min_ray_distance"], ray)

    source_edges = [edge_records[key] for key in sorted(edge_records)]
    edges = [key for key in sorted(edge_records)]
    components = _connected_components(nodes, edges)

    report_components = [
        {
            "members": members,
            "member_names": [source_name_by_index[k] for k in members],
            "is_conservative": True,
        }
        for members in components
    ]

    # Weak flags per source (weak coefficients never create edges).
    weak_flags = {
        "weak_column_indices": list(diagnostics.weak_set),
        "per_source": diagnostics.per_source_weak,
    }

    rank_deficient = int(diagnostics.numerical_rank) < len(diagnostics.reduced_to_original)
    global_unresolved = bool(rank_deficient and not source_edges)

    warnings: list[str] = []
    if global_unresolved:
        warnings.append(
            "global_unresolved: response is rank deficient but no eligible coefficient "
            "pair defines a source edge; not inventing a merge"
        )
    if diagnostics.weak_set:
        warnings.append(f"weak_sources: {len(diagnostics.weak_set)} weak coefficient fingerprint(s) flagged")

    # Grouped reporting (sums of members; no refit).
    source_level_activity_summaries = None
    grouped_activity = None
    grouped_sensor_contribution = None
    if fit is not None:
        source_level_activity_summaries = fit.source_contribution_summaries
        grouped_activity = summarize_report_groups(fit, components)
        if H_tilde is not None:
            device, dtype = H_tilde.device, H_tilde.dtype
            c_hat = fit.c_hat.to(device=device, dtype=dtype)
            metadata = fit.source_basis_metadata
            grouped_sensor_contribution = []
            for members in components:
                member_set = set(members)
                cols = [orig for orig, col in enumerate(metadata) if int(col["source_index"]) in member_set]
                if cols:
                    idx = torch.tensor(cols, dtype=torch.long, device=device)
                    contribution = H_tilde.index_select(1, idx) @ c_hat.index_select(0, idx)
                else:
                    contribution = torch.zeros(H_tilde.shape[0], dtype=dtype, device=device)
                grouped_sensor_contribution.append({
                    "members": members,
                    "member_names": [source_name_by_index[k] for k in members],
                    "contribution": [float(v) for v in to_numpy(contribution).reshape(-1).tolist()],
                })

    resolution = {
        "kind": "conservative_connected_components",
        "finest_guarantee": False,
        "n_components": len(components),
        "n_sources": len(nodes),
    }
    metadata = {
        "n_source_edges": len(source_edges),
        "rank_deficient": bool(rank_deficient),
        "numerical_rank": int(diagnostics.numerical_rank),
        "reduced_source_count": len(diagnostics.reduced_to_original),
    }
    device = getattr(diagnostics.singular_values, "device", torch.device("cpu"))
    dtype = getattr(diagnostics.singular_values, "dtype", torch.float64)
    metadata.update(runtime_provenance(device, dtype))
    config_record = {"tau_rho_ref": cfg.tau_rho_ref}
    return MergeResult(
        report_components=report_components,
        source_edges=source_edges,
        weak_flags=weak_flags,
        global_unresolved_warning=global_unresolved,
        source_level_activity_summaries=source_level_activity_summaries,
        grouped_activity=grouped_activity,
        grouped_sensor_contribution=grouped_sensor_contribution,
        resolution=resolution,
        warnings=warnings,
        config=config_record,
        metadata=metadata,
    )


__all__ = ["MergeConfig", "MergeResult", "recommend_merges"]
