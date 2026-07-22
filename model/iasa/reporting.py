#!/usr/bin/env python3
"""Task 11 reporting layer: turn saved IASA run results into paper-matching tables.

Pure functions -- each takes a parsed ``result.json`` dict and returns a list of
``ReportTable`` (title, label, ordered columns, rows, notes). No file I/O; the
CLI (``evaluation/eval_pol_iasa.py``) does loading/writing.

Column sets follow paper/8.evaluation.tex one subsection at a time. Cross-cutting
rules enforced here:
  * undefined weak-pair metrics stay ``None`` (serialize as JSON null, never NaN),
  * grouped metrics are emitted whenever a non-singleton report component exists,
  * merge edges are never deduplicated (an A-B-C chain keeps both trigger edges),
  * percentages are fractions of fitted inventory-attributed sensor signal only,
  * an uncalibrated run never reports an adequacy pass -- only its status.
"""

from __future__ import annotations

from typing import Any

# A report table is a plain dict so it round-trips to JSON unchanged.
#   {"label","title","columns":[...],"rows":[{col:val}],"notes":[...]}
ReportTable = dict[str, Any]


def _round(x: Any, nd: int = 6) -> Any:
    """Round floats for display; pass through None (weak-pair nulls) and non-floats."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        return None if x != x else round(x, nd)  # NaN -> None, never "nan"
    return x


def _table(label: str, title: str, columns: list[str], rows: list[dict[str, Any]],
           notes: list[str] | None = None) -> ReportTable:
    clean = [{c: _round(r.get(c)) for c in columns} for r in rows]
    return {"label": label, "title": title, "columns": columns,
            "rows": clean, "notes": notes or []}


def _has_nonsingleton(components: Any) -> bool:
    return bool(components) and any(isinstance(c, list) and len(c) > 1 for c in components)


# --------------------------------------------------------------------------- #
# Controlled experiments
# --------------------------------------------------------------------------- #
def report_exp01(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["noise_frac", "geometry", "sigma_J", "numerical_rank", "effective_rank",
            "condition_number", "coefficient_relative_error", "residual_norm",
            "min_visibility"]
    return [_table("exp01_conditioning_recovery",
                   "Experiment 1: Conditioning and Recovery", cols,
                   result.get("rows", []),
                   ["Recovery error vs sigma_J and condition number."])]


def report_exp02(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["offset", "triggering_pair", "max_eligible_coherence", "ray_distance",
            "individual_relative_error", "grouped_relative_error", "merged",
            "numerical_rank", "sigma_J"]
    rows = result.get("rows", [])
    notes = ["Undefined coherence/ray-distance for weak or ineligible pairs are null.",
             "Grouped error reported wherever a non-singleton component merges sources."]
    if any(r.get("merged") for r in rows):
        notes.append("At least one scenario conservatively merged pairwise-distinguishable "
                     "sources into one connected component.")
    return [_table("exp02_coherence_grouping",
                   "Experiment 2: Coherence and Grouped Reporting", cols, rows, notes)]


def report_exp03(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["background_mode", "declared_before_fit", "min_visibility", "max_absorption",
            "sigma_J", "coefficient_relative_error", "residual_norm"]
    return [_table("exp03_background_stress",
                   "Experiment 3: Background Stress", cols, result.get("rows", []),
                   ["Predeclared background basis; declared_before_fit records independence "
                    "from Y and recovery results."])]


def report_exp04(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["wind_provider", "wind", "layout", "sigma_J", "numerical_rank",
            "max_eligible_coherence", "coefficient_relative_error"]
    tables = [_table("exp04_wind_geometry",
                     "Experiment 4: Wind Diversity and Sensor Geometry", cols,
                     result.get("rows", []),
                     ["Wind comparisons use identical source-basis columns."])]
    wwe = result.get("wind_window_ensemble") or {}
    ens_rows = []
    for family, blk in wwe.items():
        if not isinstance(blk, dict):
            continue
        ens_rows.append({
            "ensemble": family, "provider": blk.get("provider"),
            "n_members": blk.get("n_members"), "n_windows": blk.get("n_windows"),
            "sigma_J_quantiles": blk.get("sigma_J_quantiles"),
            "prob_full_numerical_rank": blk.get("prob_full_numerical_rank"),
            "prob_full_effective_rank": blk.get("prob_full_effective_rank"),
            "coefficient_weak_probabilities": blk.get("coefficient_weak_probabilities"),
            "source_pair_ambiguity_probabilities": blk.get("source_pair_ambiguity_probabilities"),
            "report_component_frequencies": blk.get("report_component_frequencies"),
        })
    if ens_rows:
        ecols = ["ensemble", "provider", "n_members", "n_windows", "sigma_J_quantiles",
                 "prob_full_numerical_rank", "prob_full_effective_rank",
                 "coefficient_weak_probabilities", "source_pair_ambiguity_probabilities",
                 "report_component_frequencies"]
        tables.append(_table("exp04_wind_window_ensemble",
                             "Experiment 4: Wind-Window Ensemble Distribution", ecols, ens_rows,
                             ["Historical windows are contiguous real-record slices; "
                              "simulated windows are AR(1). Quantiles are [0.05, 0.5, 0.95]."]))
    return tables


def report_exp05(result: dict[str, Any]) -> list[ReportTable]:
    par = result.get("parametric") or {}
    cols = ["perturbation_kind", "perturbation_value", "operator_error_norm",
            "sigma_J", "coefficient_relative_error", "residual_norm"]
    tables = [_table("exp05_transport_error",
                     "Experiment 5: Transport Error (Parametric)", cols,
                     par.get("rows", []),
                     [f"Transport ensemble kind: {par.get('transport_ensemble_kind')}."])]
    st = result.get("structural") or {}
    if st:
        srow = [{"generator": st.get("generator"),
                 "operator_mismatch_norm": st.get("operator_mismatch_norm"),
                 "coefficient_relative_error_mean": st.get("coefficient_relative_error_mean"),
                 "adequacy_rejection_rate": st.get("adequacy_rejection_rate"),
                 "n_trials": st.get("n_trials")}]
        scols = ["generator", "operator_mismatch_norm", "coefficient_relative_error_mean",
                 "adequacy_rejection_rate", "n_trials"]
        tables.append(_table("exp05_structural_mismatch",
                             "Experiment 5: Structural (Edge-Hold PDE) Mismatch", scols, srow,
                             ["Structural generator differs in shape from the puff family; "
                              "a global scale cannot hide the mismatch."]))
    return tables


def report_exp06(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["scenario", "sigma_J", "coefficient_relative_error", "c_hat"]
    return [_table("exp06_inventory_robustness",
                   "Experiment 6: Inventory Robustness", cols, result.get("rows", []),
                   ["Rows are robustness scenarios, NOT confidence-interval draws.",
                    f"Transport/inventory pooling rejected: "
                    f"{result.get('transport_inventory_pooling_rejected')}."])]


def report_exp07(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["lag_window_steps", "sigma_J", "numerical_rank", "condition_number",
            "n_report_components", "coefficient_relative_error"]
    notes = [f"Selected lag: {result.get('selected_lag_window_steps')} "
             f"(tau_L={result.get('tau_L')}, criterion={result.get('selection_criterion')}).",
             f"Row count fixed across lag: {result.get('row_count_fixed')}; "
             f"coefficients used for selection: {result.get('coefficients_used_for_selection')}."]
    return [_table("exp07_lag_selection",
                   "Experiment 7: Lag-Window Selection", cols,
                   result.get("grid_rows", []), notes)]


def report_exp08(result: dict[str, Any]) -> list[ReportTable]:
    diag = result.get("diagnostics", {}) or {}
    rows = [
        {"case": "null", "rejection_rate": result.get("null_rejection_rate")},
        {"case": "residual_visible", "rejection_rate": result.get("residual_visible_power")},
        {"case": "aligned (in-span)",
         "rejection_rate": result.get("aligned_negative_control_rejection_rate")},
    ]
    cols = ["case", "rejection_rate"]
    notes = [
        f"alpha={result.get('alpha')}, n_replicates={result.get('n_replicates')}, "
        f"n_trials={result.get('n_trials')}, omission_amplitude={result.get('omission_amplitude')}.",
        f"Omitted-source out-of-span fraction={_round(diag.get('omitted_source_out_of_span_fraction'))}, "
        f"background rank={diag.get('background_rank')} (non-empty Q on the platform).",
        "Rejection diagnoses model inadequacy without identifying its cause; "
        "non-rejection cannot certify inventory completeness.",
    ]
    return [_table("exp08_missing_source_adequacy",
                   "Experiment 8: Missing-Source Adequacy", cols, rows, notes)]


def report_exp09(result: dict[str, Any]) -> list[ReportTable]:
    cols = ["noise_frac", "coefficient_relative_error", "activity_relative_error", "sigma_J"]
    return [_table("exp09_temporal_basis",
                   "Experiment 9: Temporal-Basis Recovery", cols, result.get("rows", []),
                   [f"Basis names: {result.get('basis_names')}."])]


def report_exp10(result: dict[str, Any]) -> list[ReportTable]:
    row = [{
        "footprint_localization_error_cells": result.get("footprint_localization_error_cells"),
        "footprint_mass_fraction_within_radius": result.get("footprint_mass_fraction_within_radius"),
        "localization_radius_cells": result.get("localization_radius_cells"),
        "n_active_cells": result.get("n_active_cells"),
        "contribution_sum_error": result.get("contribution_sum_error"),
        "footprints_nonnegative": result.get("footprints_nonnegative"),
        "coefficient_relative_error": result.get("coefficient_relative_error"),
    }]
    cols = ["footprint_localization_error_cells", "footprint_mass_fraction_within_radius",
            "localization_radius_cells", "n_active_cells", "contribution_sum_error",
            "footprints_nonnegative", "coefficient_relative_error"]
    notes = ["Footprint localization error vs known source origins; contributions sum "
             "to the fitted sensor signal (contribution_sum_error ~ 0)."]
    if _has_nonsingleton(result.get("report_components")):
        notes.append("Non-singleton report component present; footprints reported per group.")
    return [_table("exp10_footprints", "Experiment 10: Per-Sensor Footprints", cols, row, notes)]


# --------------------------------------------------------------------------- #
# Observed New Delhi (windowed, Tier 0)
# --------------------------------------------------------------------------- #
def _observed_week_row(result: dict[str, Any]) -> dict[str, Any]:
    return {"result": result, "week": result.get("window_index"),
            "window_start": result.get("window_start"),
            "window_end": result.get("window_end")}


def report_observed(results: list[dict[str, Any]]) -> list[ReportTable]:
    """Windowed observed study: one row per week per table (Tier 0, no aggregation)."""
    weeks = sorted(results, key=lambda r: (r.get("window_index") is None, r.get("window_index")))
    tables: list[ReportTable] = []

    # 1. Identifiability & report groups.
    idr = []
    for r in weeks:
        d = r.get("diagnostics", {}) or {}
        idr.append({"week": r.get("window_index"),
                    "window_start": r.get("window_start"), "window_end": r.get("window_end"),
                    "sigma_J": d.get("sigma_J"), "numerical_rank": d.get("numerical_rank"),
                    "effective_rank": d.get("effective_rank"),
                    "condition_status": d.get("condition_status"),
                    "max_eligible_coherence": d.get("max_eligible_coherence"),
                    "weak_set": d.get("weak_set"),
                    "report_components": r.get("report_components")})
    idcols = ["week", "window_start", "window_end", "sigma_J", "numerical_rank",
              "effective_rank", "condition_status", "max_eligible_coherence",
              "weak_set", "report_components"]
    idnotes = ["No source-activity ground truth: geometry/residuals/groups only."]
    if any(_has_nonsingleton(r.get("report_components")) for r in weeks):
        idnotes.append("A non-singleton report component appears; grouped contributions apply.")
    tables.append(_table("observed_identifiability",
                         "New Delhi Identifiability and Report Groups (weeks 1-4)",
                         idcols, idr, idnotes))

    # 2. Proxy apportionment (sensor-signal shares; non-admissible groups unsupported).
    groups: list[str] = []
    for r in weeks:
        for g in (r.get("source_names") or []):
            if g not in groups:
                groups.append(g)
    aprows = []
    for r in weeks:
        shares = r.get("sensor_signal_contribution_shares") or {}
        admissible = r.get("admissible_components_per_group")
        row = {"week": r.get("window_index")}
        for gi, g in enumerate(groups):
            # A group with no admissible component is unsupported, not zero-share.
            unsupported = isinstance(admissible, list) and gi < len(admissible) and not admissible[gi]
            row[g] = "unsupported" if unsupported else shares.get(g)
        aprows.append(row)
    apcols = ["week"] + groups
    tables.append(_table("observed_apportionment",
                         "New Delhi Proxy Apportionment (fraction of fitted sensor signal)",
                         apcols, aprows,
                         ["Shares are fractions of fitted inventory-attributed sensor signal, "
                          "NOT physical-emission shares.",
                          "Denominator: sum over groups of L1 fitted per-group sensor-signal magnitude.",
                          "Groups with no admissible temporal component are marked unsupported."]))

    # 3. Sensor fit & residual diagnostics.
    rdr = []
    for r in weeks:
        rdr.append({"week": r.get("window_index"),
                    "n_observed_rows": r.get("n_observed_rows"),
                    "n_total_rows": r.get("n_total_rows"),
                    "observed_mask_fraction": r.get("observed_mask_fraction"),
                    "residual_norm": r.get("residual_norm"),
                    "projected_residual_norm": r.get("projected_residual_norm"),
                    "kriged_baseline_subtracted": r.get("kriged_baseline_subtracted"),
                    "pm25_imputed": r.get("pm25_imputed"),
                    "wind_provider": r.get("wind_provider"),
                    "calibration_status": r.get("calibration_status")})
    rdcols = ["week", "n_observed_rows", "n_total_rows", "observed_mask_fraction",
              "residual_norm", "projected_residual_norm", "kriged_baseline_subtracted",
              "pm25_imputed", "wind_provider", "calibration_status"]
    tables.append(_table("observed_residuals",
                         "New Delhi Sensor Fit and Residual Diagnostics (weeks 1-4)",
                         rdcols, rdr,
                         ["PM2.5 is never imputed. Uncalibrated: no adequacy pass is presented."]))

    # 4. Per-monitor group contributions (one row per monitor per week).
    pmr = []
    for r in weeks:
        per = r.get("per_monitor_group_contributions") or {}
        for monitor, contribs in per.items():
            row = {"week": r.get("window_index"), "monitor": monitor}
            row.update({g: contribs.get(g) for g in groups})
            pmr.append(row)
    pmcols = ["week", "monitor"] + groups
    tables.append(_table("observed_per_monitor",
                         "New Delhi Per-Monitor Fitted Group Contributions (projected sensor signal)",
                         pmcols, pmr,
                         ["Projected (P_Q^perp) per-monitor contributions can be signed; "
                          "the reported apportionment shares use L1 magnitudes."]))
    return tables


def report_wind_imputation(checkpoint_report: dict[str, Any] | None) -> ReportTable:
    """Held-out wind-imputation validation from a checkpoint report json, if present."""
    if not checkpoint_report:
        return _table("observed_wind_imputation",
                      "New Delhi Wind-Imputation Validation", ["metric", "value"],
                      [{"metric": "status",
                        "value": "unavailable -- paper-facing wind is the kernel coordinate-query "
                                 "imputer (FieldFormer evaluated but not adopted; see reconciliation)"}],
                      ["Dense real wind truth is unavailable; gridded-field accuracy is assessed "
                       "in controlled synthetic-wind experiments (Experiment 4)."])
    rows = [{"metric": k, "value": v} for k, v in checkpoint_report.items()
            if isinstance(v, (int, float, str, bool, type(None)))]
    return _table("observed_wind_imputation",
                  "New Delhi Wind-Imputation Validation", ["metric", "value"], rows)


# --------------------------------------------------------------------------- #
# Dispatch + aggregation
# --------------------------------------------------------------------------- #
_CONTROLLED = {
    "exp01_conditioning_predicts_recovery": report_exp01,
    "exp02_coherent_sources_grouped": report_exp02,
    "exp03_background_help_or_hurt": report_exp03,
    "exp04_wind_diversity_geometry": report_exp04,
    "exp05_transport_error": report_exp05,
    "exp06_inventory_error": report_exp06,
    "exp07_lag_window_sensitivity": report_exp07,
    "exp08_missing_source_adequacy": report_exp08,
    "exp09_temporal_basis_recovery": report_exp09,
    "exp10_footprints_spatial_attribution": report_exp10,
}


def report_result(result: dict[str, Any]) -> list[ReportTable]:
    """Single controlled result -> its report tables (observed handled separately)."""
    fn = _CONTROLLED.get(result.get("experiment", ""))
    return fn(result) if fn else []


def report_merge_edges(result: dict[str, Any]) -> ReportTable:
    """Render source_edges + connected components WITHOUT deduplication, so an
    A-B-C chain retains both trigger edges. Used where a result carries a merge."""
    edges = result.get("source_edges") or []
    rows = [{"sources": e.get("sources"), "max_coherence": e.get("max_coherence")}
            for e in edges]
    return _table("merge_edges", "Conservative Merge: Retained Trigger Edges",
                  ["sources", "max_coherence"], rows,
                  [f"Report components: {result.get('report_components')}",
                   "Every eligible trigger edge is retained (chains keep all edges)."])
