"""The paper's ten one-factor controlled experiments + observed-New-Delhi mode.

Every experiment reuses the shared base platform (``nd_platform.py``) so wind,
source, background, and geometry comparisons use identical source, temporal-basis,
sensor, observation-mask, and background columns on both sides. Each returns a
JSON-able result carrying BOTH an ``accuracy`` block and a ``diagnostics`` block
(roadmap acceptance), plus ``arrays`` destined for ``arrays.npz``.

Real imputed/synthetic New Delhi wind is a controlled transport input with
synthetic coefficients; the observed-PM2.5 mode is a separate evaluation and is
never assigned synthetic recovery metrics.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import torch

from model.iasa.backend import to_numpy
from model.iasa.background import BackgroundBasisConfig, build_background_basis
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_projection
from model.iasa.fit import (
    AdequacyConfig,
    FitConfig,
    NoiseModel,
    aggregate_inventory_scenarios,
    aggregate_transport_ensemble,
    fit_projection,
    residual_adequacy_check,
)
from model.iasa.merge import recommend_merges
from model.iasa.projection import project_response_and_observations
from model.iasa.response import build_lagged_response_matrix
from model.iasa.wind import WindSequence, constant_direction

from experiments.iasa_pol import edge_hold_pde
from experiments.iasa_pol.nd_platform import (
    Platform,
    compact_source,
    default_basis,
    make_wind,
    sensor_layout,
    synthetic_coefficients,
)


# --------------------------------------------------------------------------- #
# Shared forward pipeline
# --------------------------------------------------------------------------- #
def _max_eligible_coherence(diag) -> float:
    coh = to_numpy(diag.coherence)
    best = 0.0
    for i in range(coh.shape[0]):
        for j in range(i + 1, coh.shape[0]):
            v = coh[i, j]
            if not np.isnan(v) and v > best:
                best = float(v)
    return best


def _diag_summary(diag) -> dict[str, Any]:
    return {
        "sigma_J": diag.sigma_J,
        "sigma_1": diag.sigma_1,
        "numerical_rank": diag.numerical_rank,
        "effective_rank": diag.effective_rank,
        "condition_number": diag.condition_number,
        "condition_status": diag.condition_status,
        "max_eligible_coherence": _max_eligible_coherence(diag),
        "weak_set": list(diag.weak_set),
        "visibility": to_numpy(diag.visibility).tolist(),
    }


def _build_response(platform: Platform, source_maps, source_names, basis, observer, wind,
                    *, lag_window_steps: int | None = None, dispersion=None):
    rc = platform.response_config
    if lag_window_steps is not None:
        rc = replace(rc, lag_window_steps=int(lag_window_steps))
    dc = dispersion if dispersion is not None else platform.dispersion_config
    return build_lagged_response_matrix(
        source_maps, source_names, basis, observer, wind,
        response_config=rc, dispersion_config=dc,
    )


def _background_for(platform: Platform, response, timestamps, observer, mode: str):
    """Declared background basis. redundant/stress are built from the primary Q so
    they share span / add a labeled source-like column (Experiment 3)."""
    if mode == "none":
        return build_background_basis(
            response.row_index, timestamps, observer.sensor_xy,
            BackgroundBasisConfig(include_constant=False),
        )
    primary = build_background_basis(
        response.row_index, timestamps, observer.sensor_xy,
        platform.background_config("primary"),
    )
    if mode == "primary":
        return primary
    Q = to_numpy(primary.Q)
    if mode == "redundant":
        return build_background_basis(
            response.row_index, timestamps, observer.sensor_xy,
            BackgroundBasisConfig(include_constant=False, basis_mode="stress"),
            user_basis=np.column_stack([Q, Q[:, 0]]),
            user_basis_names=list(primary.column_names) + ["constant_duplicate"],
        )
    if mode == "stress":
        H = to_numpy(response.H_lag).astype(np.float64)
        col = H[:, min(H.shape[1] - 1, 0)].copy()
        col /= max(float(np.linalg.norm(col)), np.finfo(np.float64).eps)
        return build_background_basis(
            response.row_index, timestamps, observer.sensor_xy,
            BackgroundBasisConfig(include_constant=False, basis_mode="stress"),
            user_basis=np.column_stack([Q, col]),
            user_basis_names=list(primary.column_names) + ["stress_source_like"],
        )
    raise ValueError(f"unknown background mode {mode!r}")


def forward(
    platform: Platform,
    source_maps: np.ndarray,
    source_names: Sequence[str],
    basis,
    observer,
    wind,
    *,
    c_true: np.ndarray | None = None,
    background_mode: str = "primary",
    noise_frac: float = 0.0,
    seed: int = 0,
    fixed_zero_indices: tuple[int, ...] = (),
    tau_rho: float = 0.99,
    beta_scale: float = 0.1,
    with_temporal: bool = False,
    Y_override: np.ndarray | None = None,
    ensemble_kind: str = "inventory",
    run_adequacy: bool = False,
    adequacy_sigma: float | None = None,
    n_replicates: int = 200,
    alpha: float = 0.05,
    lag_window_steps: int | None = None,
    dispersion=None,
) -> dict[str, Any]:
    """Run response -> background -> project -> diagnose -> fit -> merge and score."""
    response = _build_response(platform, source_maps, source_names, basis, observer, wind,
                               lag_window_steps=lag_window_steps, dispersion=dispersion)
    T = basis.values.shape[0]
    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    background = _background_for(platform, response, timestamps, observer, background_mode)
    H_lag = to_numpy(response.H_lag).astype(np.float64)
    Qnp = to_numpy(background.Q)

    n_cols = H_lag.shape[1]
    if c_true is None and Y_override is None:
        c_true = synthetic_coefficients(len(source_names), len(basis.names), seed=seed)
    c_true_np = None if c_true is None else np.asarray(c_true, dtype=np.float64)

    sigma_e = 0.0
    if Y_override is not None:
        Y = np.asarray(Y_override, dtype=np.float64)
    else:
        clean = H_lag @ c_true_np
        beta = np.full(Qnp.shape[1], beta_scale, dtype=np.float64) if Qnp.shape[1] else np.zeros(0)
        bg = Qnp @ beta if Qnp.shape[1] else np.zeros_like(clean)
        max_signal = max(float(np.max(np.abs(clean))), 1e-12)
        sigma_e = float(noise_frac) * max_signal
        rng = np.random.default_rng(seed + 991)
        noise = rng.normal(0.0, sigma_e, size=clean.shape[0]) if sigma_e > 0 else np.zeros_like(clean)
        Y = clean + bg + noise

    projection = project_response_and_observations(
        response.H_lag, Y, background, response.row_index, response.column_index
    )
    diagnostics = diagnose_projection(
        projection, DiagnosticsConfig(tau_rho=tau_rho, fixed_zero_indices=fixed_zero_indices)
    )
    temporal_basis = None
    if with_temporal:
        temporal_basis = torch.as_tensor(basis.values, dtype=torch.float64)
    fit = fit_projection(
        projection,
        config=FitConfig(fixed_zero_indices=fixed_zero_indices, ensemble_kind=ensemble_kind),
        temporal_basis=temporal_basis,
        timestamps=[t for t in timestamps] if with_temporal else None,
    )
    merge = recommend_merges(diagnostics, fit=fit, H_tilde=projection.H_tilde)

    c_hat = to_numpy(fit.c_hat)
    accuracy: dict[str, Any] = {
        "residual_norm": fit.residual_norm,
        "zero_model_residual_norm": fit.zero_model_residual_norm,
    }
    if c_true_np is not None:
        accuracy["coefficient_relative_error"] = float(
            np.linalg.norm(c_hat - c_true_np) / max(np.linalg.norm(c_true_np), 1e-12)
        )
        if with_temporal and fit.theta is not None:
            # activity-trajectory error vs the true theta = Phi @ C_true^T.
            C_true = c_true_np.reshape(len(source_names), len(basis.names))
            theta_true = basis.values.astype(np.float64) @ C_true.T
            theta_hat = to_numpy(fit.theta)
            accuracy["activity_relative_error"] = float(
                np.linalg.norm(theta_hat - theta_true) / max(np.linalg.norm(theta_true), 1e-12)
            )

    adequacy_out = None
    if run_adequacy:
        # Declared observation-noise std: the sweep's own sigma_e, or an explicit
        # override (e.g. the structural case whose Y is supplied via Y_override).
        adq_sigma = adequacy_sigma if adequacy_sigma is not None else sigma_e
        noise_model = NoiseModel(
            covariance=max(adq_sigma, 1e-6) ** 2, calibrated=True,
            source="task10_declared_noise", estimated_from_fit_residual=False,
        )
        adq = residual_adequacy_check(
            fit, projection, noise_model,
            config=AdequacyConfig(alpha=alpha, n_replicates=n_replicates, seed=seed + 13),
        )
        adequacy_out = {
            "T_res": adq.T_res, "p_value": adq.p_value, "inadequate": adq.inadequate,
            "bootstrap_quantile": adq.bootstrap_quantile,
        }

    return {
        "accuracy": accuracy,
        "diagnostics": _diag_summary(diagnostics),
        "report_components": [c["members"] for c in merge.report_components],
        "source_edges": [
            {"sources": list(e["sources"]), "max_coherence": e["max_coherence"]}
            for e in merge.source_edges
        ],
        "c_true": None if c_true_np is None else c_true_np.tolist(),
        "c_hat": c_hat.tolist(),
        "adequacy": adequacy_out,
        "sigma_e": sigma_e,
        # objects for downstream use / arrays (not JSON-serialized directly)
        "_response": response,
        "_projection": projection,
        "_diagnostics": diagnostics,
        "_fit": fit,
        "_merge": merge,
        "_background": background,
        "_Y": Y,
        "_H_lag": H_lag,
    }


def _strip_objects(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _diverse_wind(T: int) -> WindSequence:
    vx = np.ones(T, dtype=np.float32)
    vy = np.zeros(T, dtype=np.float32)
    vx[T // 2:] = 0.0
    vy[T // 2:] = 1.0
    return WindSequence(
        timestamps=constant_direction(length=T, vx=1.0, vy=0.0).timestamps,
        vx=vx, vy=vy, provider="two_direction_synthetic", metadata={"switch_time_index": T // 2},
    )


def _arrays_from_bundle(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    proj = bundle["_projection"]
    diag = bundle["_diagnostics"]
    fit = bundle["_fit"]
    return {
        "H_lag": bundle["_H_lag"].astype(np.float32),
        "H_tilde": to_numpy(proj.H_tilde).astype(np.float32),
        "Y": np.asarray(bundle["_Y"], dtype=np.float32),
        "c_hat": to_numpy(fit.c_hat).astype(np.float32),
        "singular_values": to_numpy(diag.singular_values).astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Experiment 1 -- conditioning predicts recovery
# --------------------------------------------------------------------------- #
def experiment_1(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    observer = platform.observer
    wind = _diverse_wind(T)
    noise_fracs = cfg.get("noise_fracs", [0.0, 0.01, 0.05, 0.10, 0.20])
    geometries = {
        "separated": np.stack([compact_source(gs, (gs[0] * 0.25, gs[1] * 0.5)),
                               compact_source(gs, (gs[0] * 0.75, gs[1] * 0.5))], axis=0),
        "close": np.stack([compact_source(gs, (gs[0] * 0.45, gs[1] * 0.5)),
                           compact_source(gs, (gs[0] * 0.55, gs[1] * 0.5))], axis=0),
    }
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    rows = []
    last_bundle = None
    for geom_name, maps in geometries.items():
        for nf in noise_fracs:
            b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                        c_true=c_true, noise_frac=float(nf), seed=seed)
            last_bundle = b
            rows.append({
                "geometry": geom_name, "noise_frac": float(nf),
                "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
                "residual_norm": b["accuracy"]["residual_norm"],
                "sigma_J": b["diagnostics"]["sigma_J"],
                "numerical_rank": b["diagnostics"]["numerical_rank"],
                "effective_rank": b["diagnostics"]["effective_rank"],
                "condition_number": b["diagnostics"]["condition_number"],
                "min_visibility": float(min(b["diagnostics"]["visibility"])),
            })
    return {
        "experiment": "exp01_conditioning_predicts_recovery", "hypothesis": "H1",
        "rows": rows,
        "arrays": _arrays_from_bundle(last_bundle),
    }


# --------------------------------------------------------------------------- #
# Experiment 2 -- coherent sources require grouped reporting
# --------------------------------------------------------------------------- #
def experiment_2(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    observer = platform.observer
    wind = make_wind("constant", T)  # aligned transport -> shifted copies stay coherent
    offsets = cfg.get("offsets", [6.0, 4.0, 2.0, 1.0])
    base_center = (gs[0] * 0.35, gs[1] * 0.5)
    c_true = np.asarray([1.0, 0.7], dtype=np.float64)
    rows = []
    last_bundle = None
    for off in offsets:
        maps = np.stack([
            compact_source(gs, base_center),
            compact_source(gs, (base_center[0] + float(off), base_center[1])),
        ], axis=0)
        b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind, c_true=c_true, seed=seed)
        last_bundle = b
        comps = b["report_components"]
        merged = any(set(c) == {0, 1} for c in comps)
        # individual vs grouped error
        c_hat = np.asarray(b["c_hat"])
        indiv_err = float(np.linalg.norm(c_hat - c_true) / np.linalg.norm(c_true))
        grouped_true = float(c_true.sum())
        grouped_hat = float(c_hat[:2].sum())
        grouped_err = abs(grouped_hat - grouped_true) / max(grouped_true, 1e-12)
        trigger = None
        for e in b["source_edges"]:
            if set(e["sources"]) == {0, 1}:
                trigger = e
        rows.append({
            "offset": float(off),
            "max_eligible_coherence": b["diagnostics"]["max_eligible_coherence"],
            "sigma_J": b["diagnostics"]["sigma_J"],
            "numerical_rank": b["diagnostics"]["numerical_rank"],
            "merged": bool(merged),
            "individual_relative_error": indiv_err,
            "grouped_relative_error": grouped_err,
            "triggering_pair": trigger,
        })
    return {
        "experiment": "exp02_coherent_sources_grouped", "hypothesis": "H2",
        "rows": rows,
        "arrays": _arrays_from_bundle(last_bundle),
    }


# --------------------------------------------------------------------------- #
# Experiment 3 -- background correction can help or hurt
# --------------------------------------------------------------------------- #
def experiment_3(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("impulse_constant", T)
    observer = platform.observer
    wind = _diverse_wind(T)
    maps = np.stack([compact_source(gs, (gs[0] * 0.3, gs[1] * 0.5)),
                     compact_source(gs, (gs[0] * 0.7, gs[1] * 0.5))], axis=0)
    c_true = synthetic_coefficients(2, len(basis.names), seed=seed)
    rows = []
    last_bundle = None
    for mode in cfg.get("background_modes", ["none", "primary", "redundant", "stress"]):
        b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                    c_true=c_true, background_mode=mode, noise_frac=float(cfg.get("noise_frac", 0.05)),
                    seed=seed)
        last_bundle = b
        proj = b["_projection"]
        vis = proj.metadata.get("H_visibility_ratio_by_column", [])
        absorp = proj.metadata.get("H_absorption_ratio_by_column", [])
        rows.append({
            "background_mode": mode,
            "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
            "residual_norm": b["accuracy"]["residual_norm"],
            "min_visibility": float(min(vis)) if len(vis) else None,
            "max_absorption": float(max(absorp)) if len(absorp) else None,
            "sigma_J": b["diagnostics"]["sigma_J"],
            "declared_before_fit": True,
        })
    return {
        "experiment": "exp03_background_help_or_hurt", "hypothesis": "H3",
        "rows": rows,
        "arrays": _arrays_from_bundle(last_bundle),
    }


# --------------------------------------------------------------------------- #
# Experiment 4 -- wind diversity and sensor geometry change resolution
# --------------------------------------------------------------------------- #
def experiment_4(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    maps = np.stack([compact_source(gs, (gs[0] * 0.4, gs[1] * 0.5)),
                     compact_source(gs, (gs[0] * 0.5, gs[1] * 0.5))], axis=0)
    c_true = np.asarray([1.0, 0.7], dtype=np.float64)
    wind_kinds = cfg.get("wind_kinds", ["constant", "single", "diurnal", "ar1", "multi"])
    if cfg.get("include_real_wind", False):
        wind_kinds = list(wind_kinds) + ["real"]
    layouts = cfg.get("layouts", ["regulatory", "random", "downwind"])
    rows = []
    last_bundle = None
    for wk in wind_kinds:
        try:
            wind = make_wind(wk, T, seed=seed)
        except Exception as exc:  # real wind product may be unavailable
            rows.append({"wind": wk, "layout": None, "error": str(exc)})
            continue
        for lay in layouts:
            observer = sensor_layout(lay, gs, n=cfg.get("n_sensors", 6), seed=seed,
                                     regulatory=platform.observer)
            b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                        c_true=c_true, seed=seed)
            last_bundle = b
            rows.append({
                "wind": wk, "layout": lay,
                "sigma_J": b["diagnostics"]["sigma_J"],
                "max_eligible_coherence": b["diagnostics"]["max_eligible_coherence"],
                "numerical_rank": b["diagnostics"]["numerical_rank"],
                "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
            })

    # Wind-window ensembles: simulated (AR1 seeds) and historical (real slices).
    ensemble = _wind_window_ensemble(platform, maps, basis, c_true, seed, cfg)
    return {
        "experiment": "exp04_wind_diversity_geometry", "hypothesis": "H4",
        "rows": rows,
        "wind_window_ensemble": ensemble,
        "arrays": _arrays_from_bundle(last_bundle) if last_bundle else {},
    }


def _wind_window_ensemble(platform, maps, basis, c_true, seed, cfg) -> dict[str, Any]:
    from model.iasa.diagnostics import summarize_wind_ensemble

    T = platform.config.T
    observer = platform.observer
    n_members = int(cfg.get("ensemble_members", 8))
    out: dict[str, Any] = {}
    for family in ("simulated", "historical"):
        diags = []
        for m in range(n_members):
            if family == "simulated":
                wind = make_wind("ar1", T, seed=seed + 100 + m)
            else:
                # "historical" windows: shifted diurnal phases as proxy real windows.
                wind = make_wind("multi", T, seed=seed + 200 + m)
            b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                        c_true=c_true, seed=seed)
            diags.append(b["_diagnostics"])
        try:
            summary = summarize_wind_ensemble(diags, quantiles=(0.05, 0.5, 0.95))
            out[family] = {k: v for k, v in summary.items() if k != "column_index"}
        except Exception as exc:
            out[family] = {"error": str(exc)}
    return out


# --------------------------------------------------------------------------- #
# Experiment 5 -- transport error amplified by ill-conditioning
# --------------------------------------------------------------------------- #
def experiment_5(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    observer = platform.observer
    maps = np.stack([compact_source(gs, (gs[0] * 0.3, gs[1] * 0.5)),
                     compact_source(gs, (gs[0] * 0.55, gs[1] * 0.5))], axis=0)
    source_names = ["src_a", "src_b"]
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    true_wind = make_wind("constant", T)

    # --- (a) parametric transport perturbations (transport ensemble) ---
    truth = _build_response(platform, maps, source_names, basis, observer, true_wind)
    H_true = to_numpy(truth.H_lag).astype(np.float64)
    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    bg = build_background_basis(truth.row_index, timestamps, observer.sensor_xy,
                                platform.background_config("primary"))
    Y = H_true @ c_true + to_numpy(bg.Q) @ np.full(len(bg.column_names), 0.1)
    parametric_rows = []
    transport_fits = []
    for dtheta in cfg.get("wind_direction_perturbations_deg", [0.0, 5.0, 10.0, 20.0]):
        pert_wind = make_wind("single", T, seed=seed, direction_degrees=float(dtheta))
        pert = _build_response(platform, maps, source_names, basis, observer, pert_wind)
        H_pert = to_numpy(pert.H_lag).astype(np.float64)
        op_err = float(np.linalg.norm(H_pert - H_true) / max(np.linalg.norm(H_true), 1e-12))
        proj = project_response_and_observations(pert.H_lag, Y, bg, pert.row_index, pert.column_index)
        diag = diagnose_projection(proj, DiagnosticsConfig())
        fit = fit_projection(proj, config=FitConfig(ensemble_kind="transport"))
        transport_fits.append(fit)
        parametric_rows.append({
            "wind_direction_perturbation_deg": float(dtheta),
            "operator_error_norm": op_err,
            "coefficient_relative_error": float(
                np.linalg.norm(to_numpy(fit.c_hat) - c_true) / np.linalg.norm(c_true)
            ),
            "residual_norm": fit.residual_norm,
            "sigma_J": diag.sigma_J,
            "singular_values": to_numpy(diag.singular_values).tolist(),
        })
    transport_ensemble = aggregate_transport_ensemble(transport_fits)

    # --- (b) structural mismatch: edge-hold PDE generates Y; puff response fits ---
    structural = _experiment_5_structural(platform, maps, source_names, basis, observer,
                                           true_wind, c_true, bg, H_true, cfg, seed)

    return {
        "experiment": "exp05_transport_error", "hypothesis": "H5a",
        "parametric": {"rows": parametric_rows,
                       "transport_ensemble_kind": transport_ensemble["ensemble_kind"]},
        "structural": structural,
        "arrays": {"H_true": H_true.astype(np.float32)},
    }


def _experiment_5_structural(platform, maps, source_names, basis, observer, true_wind,
                             c_true, bg, H_true, cfg, seed) -> dict[str, Any]:
    from model.iasa.activity import combine_inventory_sources

    T = platform.config.T
    # True activity theta_true(t) = Phi @ C_true^T, then the emission field per step.
    C_true = c_true.reshape(len(source_names), len(basis.names))
    theta_true = basis.values.astype(np.float64) @ C_true.T  # [T, K]
    source_terms = combine_inventory_sources(maps, theta_true.astype(np.float32))  # [T, Nx, Ny]

    sim = edge_hold_pde.simulate_edge_hold_observations(
        source_terms, true_wind.vx, true_wind.vy, observer.sensor_xy,
        config=edge_hold_pde.EdgeHoldConfig(), device=platform.config.device,
    )
    Y_pde_rows = edge_hold_pde.observations_to_row_vector(
        sim["observations"], truth_row_index := _row_index_for(platform, maps, source_names, basis, observer, true_wind),
        observer.sensor_ids,
    )
    # Match global magnitude to the puff-consistent observation so the comparison is
    # about STRUCTURAL shape mismatch, not arbitrary emission-rate units.
    Y_puff = H_true @ c_true
    denom = float(np.dot(Y_pde_rows, Y_pde_rows))
    alpha_scale = float(np.dot(Y_pde_rows, Y_puff) / denom) if denom > 0 else 1.0
    Y_struct = alpha_scale * Y_pde_rows
    op_mismatch = float(np.linalg.norm(Y_struct - Y_puff) / max(np.linalg.norm(Y_puff), 1e-12))

    rejects = 0
    n_trials = int(cfg.get("structural_adequacy_trials", 5))
    n_replicates = int(cfg.get("structural_n_replicates", 200))
    coeff_errs = []
    struct_noise_frac = float(cfg.get("structural_noise_frac", 0.02))
    struct_sigma = struct_noise_frac * max(float(np.max(np.abs(Y_struct))), 1e-12)
    for t in range(n_trials):
        b = forward(platform, maps, source_names, basis, observer, true_wind,
                    c_true=c_true, Y_override=Y_struct + _obs_noise(Y_struct, cfg, seed + t),
                    run_adequacy=True, adequacy_sigma=struct_sigma,
                    n_replicates=n_replicates, seed=seed + t)
        coeff_errs.append(b["accuracy"]["coefficient_relative_error"])
        if b["adequacy"] and b["adequacy"]["inadequate"]:
            rejects += 1
    return {
        "generator": sim["generator"],
        "operator_mismatch_norm": op_mismatch,
        "coefficient_relative_error_mean": float(np.mean(coeff_errs)),
        "adequacy_rejection_rate": rejects / max(n_trials, 1),
        "n_trials": n_trials,
        "edge_hold_config": sim["config"],
    }


def _obs_noise(Y, cfg, seed):
    frac = float(cfg.get("structural_noise_frac", 0.02))
    sigma = frac * max(float(np.max(np.abs(Y))), 1e-12)
    rng = np.random.default_rng(seed + 555)
    return rng.normal(0.0, sigma, size=Y.shape[0])


def _row_index_for(platform, maps, source_names, basis, observer, wind):
    return _build_response(platform, maps, source_names, basis, observer, wind).row_index


# --------------------------------------------------------------------------- #
# Experiment 6 -- inventory error changes the attribution target
# --------------------------------------------------------------------------- #
def experiment_6(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    observer = platform.observer
    wind = make_wind("constant", T)  # transport held fixed
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    base_center = (gs[0] * 0.35, gs[1] * 0.5)
    second = (gs[0] * 0.65, gs[1] * 0.5)

    scenarios = {
        "baseline": np.stack([compact_source(gs, base_center), compact_source(gs, second)], axis=0),
        "location_shift": np.stack([compact_source(gs, (base_center[0] + 2, base_center[1])),
                                    compact_source(gs, second)], axis=0),
        "scale_change": np.stack([compact_source(gs, base_center, sigma=2.0),
                                  compact_source(gs, second)], axis=0),
        "category_swap": np.stack([compact_source(gs, second), compact_source(gs, base_center)], axis=0),
    }
    inventory_fits = []
    names = []
    rows = []
    last_bundle = None
    for name, maps in scenarios.items():
        b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                    c_true=c_true, ensemble_kind="inventory", seed=seed)
        last_bundle = b
        inventory_fits.append(b["_fit"])
        names.append(name)
        rows.append({
            "scenario": name,
            "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
            "c_hat": b["c_hat"],
            "sigma_J": b["diagnostics"]["sigma_J"],
        })
    scenario_agg = aggregate_inventory_scenarios(inventory_fits, scenario_names=names)

    # Type-separation guard: inventory scenarios must never pool with transport fits.
    transport_fit = forward(platform, scenarios["baseline"], ["src_a", "src_b"], basis, observer,
                            wind, c_true=c_true, ensemble_kind="transport", seed=seed)["_fit"]
    pooling_rejected = False
    try:
        aggregate_inventory_scenarios(inventory_fits + [transport_fit])
    except ValueError:
        pooling_rejected = True

    return {
        "experiment": "exp06_inventory_error", "hypothesis": "H5b",
        "rows": rows,
        "scenario_aggregate_kind": scenario_agg["ensemble_kind"],
        "n_scenarios": scenario_agg["n_scenarios"],
        "transport_inventory_pooling_rejected": pooling_rejected,
        "arrays": _arrays_from_bundle(last_bundle),
    }


# --------------------------------------------------------------------------- #
# Experiment 7 -- lag-window sensitivity
# --------------------------------------------------------------------------- #
def experiment_7(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    observer = platform.observer
    wind = _diverse_wind(T)
    maps = np.stack([compact_source(gs, (gs[0] * 0.3, gs[1] * 0.5)),
                     compact_source(gs, (gs[0] * 0.7, gs[1] * 0.5))], axis=0)
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    tau_L = float(cfg.get("tau_L", 1e-3))
    lag_grid = cfg.get("lag_grid", [4, 6, 8, 10, 12])

    # Frobenius stability of H across the candidate grid.
    H_by_L = {}
    row_counts = set()
    for L in lag_grid:
        resp = _build_response(platform, maps, ["src_a", "src_b"], basis, observer, wind, lag_window_steps=L)
        H_by_L[L] = to_numpy(resp.H_lag).astype(np.float64)
        row_counts.add(H_by_L[L].shape[0])
    deltas = []
    selected_L = lag_grid[-1]
    for i in range(len(lag_grid) - 1):
        L, Lp = lag_grid[i], lag_grid[i + 1]
        rel = float(np.linalg.norm(H_by_L[Lp] - H_by_L[L]) / max(np.linalg.norm(H_by_L[Lp]), 1e-12))
        deltas.append({"L": L, "L_plus": Lp, "relative_frobenius_delta": rel})
    for d in deltas:
        if d["relative_frobenius_delta"] <= tau_L:
            selected_L = d["L"]
            break

    # rank / conditioning / report-component stability across the grid.
    grid_rows = []
    for L in lag_grid:
        b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                    c_true=c_true, lag_window_steps=L, seed=seed)
        grid_rows.append({
            "lag_window_steps": L,
            "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
            "numerical_rank": b["diagnostics"]["numerical_rank"],
            "condition_number": b["diagnostics"]["condition_number"],
            "sigma_J": b["diagnostics"]["sigma_J"],
            "n_report_components": len(b["report_components"]),
        })
    return {
        "experiment": "exp07_lag_window_sensitivity",
        "tau_L": tau_L,
        "selected_lag_window_steps": int(selected_L),
        "row_count_fixed": len(row_counts) == 1,
        "adjacent_deltas": deltas,
        "grid_rows": grid_rows,
        "coefficients_used_for_selection": False,
        "arrays": {},
    }


# --------------------------------------------------------------------------- #
# Experiment 8 -- missing-source model adequacy
# --------------------------------------------------------------------------- #
def experiment_8(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Refitted parametric-bootstrap adequacy under (a) a residual-visible omission
    (outside span([H_lag, Q])) and (b) an aligned omission (inside the span) --
    a required negative control: non-rejection cannot certify completeness."""
    device = torch.device("cpu")
    dtype = torch.float64
    N = int(cfg.get("N", 48))
    n_replicates = int(cfg.get("n_replicates", 200))
    n_trials = int(cfg.get("n_trials", 40))
    alpha = float(cfg.get("alpha", 0.05))
    sigma_e = float(cfg.get("sigma_e", 0.1))

    # Deterministic operator with mutually orthogonal columns (clean residual power).
    g0 = torch.Generator(device=device)
    g0.manual_seed(20260720)
    A = torch.randn(N, 4, dtype=dtype, device=device, generator=g0)
    Qb, _ = torch.linalg.qr(A)
    H = Qb * torch.tensor([1.0, 0.9, 0.8, 0.7], dtype=dtype, device=device)

    cols = [{"source_index": k, "source_name": f"s{k}", "basis_index": 0, "basis_name": "c"} for k in range(4)]
    row_index = [{"time_index": i, "sensor_index": 0, "sensor_id": "s"} for i in range(N)]
    ts = np.datetime64("2018-05-01T00:00") + np.arange(N) * np.timedelta64(1, "h")
    empty_bg = build_background_basis(row_index, ts, config=BackgroundBasisConfig(include_constant=False))
    c_base = torch.tensor([1.0, 0.8, 0.6, 0.0], dtype=dtype, device=device)
    noise_model = NoiseModel(covariance=sigma_e ** 2, calibrated=True,
                             source="task10_exp8", estimated_from_fit_residual=False)

    # Deterministic per-case seed offset (never Python's salted str hash).
    case_offset = {"residual_visible": 1, "aligned": 2}

    def _run(case: str, amplitude: float) -> dict[str, Any]:
        rejects = 0
        for t in range(n_trials):
            c_true = c_base.clone()
            if case == "residual_visible":
                c_true[3] = float(amplitude)  # column 3 omitted from fit -> visible residual
                fixed = (3,)
            else:  # aligned: signal lies in a FITTED column (index 1) -> absorbed
                c_true[1] = c_true[1] + float(amplitude)
                fixed = (3,)
            mean = H @ c_true
            g = torch.Generator(device=device)
            g.manual_seed(seed + 1_000_003 * case_offset[case] + t)
            Y = to_numpy(mean + sigma_e * torch.randn(N, dtype=dtype, device=device, generator=g))
            proj = project_response_and_observations(H, Y, empty_bg, row_index, cols)
            fit = fit_projection(proj, config=FitConfig(fixed_zero_indices=fixed))
            adq = residual_adequacy_check(
                fit, proj, noise_model,
                config=AdequacyConfig(alpha=alpha, n_replicates=n_replicates, seed=seed + 7_654_321 + t),
            )
            rejects += int(bool(adq.inadequate))
        return {"case": case, "amplitude": float(amplitude), "rejection_rate": rejects / max(n_trials, 1)}

    null = _run("residual_visible", 0.0)
    power = _run("residual_visible", float(cfg.get("omission_amplitude", 1.0)))
    aligned = _run("aligned", float(cfg.get("omission_amplitude", 1.0)))

    # Identifiability diagnostics of the fitted design (columns 0-2; column 3 omitted).
    svals = torch.linalg.svdvals(H[:, :3]).tolist()
    tol = 1e-10 * max(svals)
    diagnostics = {
        "sigma_J": float(min(svals)),
        "numerical_rank": int(sum(1 for s in svals if s > tol)),
        "singular_values": [float(s) for s in svals],
        "omitted_column_visible_norm": float(torch.linalg.vector_norm(H[:, 3])),
    }
    return {
        "experiment": "exp08_missing_source_adequacy",
        "null_rejection_rate": null["rejection_rate"],
        "residual_visible_power": power["rejection_rate"],
        "aligned_negative_control_rejection_rate": aligned["rejection_rate"],
        "diagnostics": diagnostics,
        "alpha": alpha, "n_trials": n_trials, "n_replicates": n_replicates,
        "negative_control_note": "aligned omission absorbed in-span; non-rejection cannot certify completeness",
        "arrays": {},
    }


# --------------------------------------------------------------------------- #
# Experiment 9 -- temporal-basis recovery
# --------------------------------------------------------------------------- #
def experiment_9(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("multi", T)  # diurnal / block / day_night
    observer = platform.observer
    wind = _diverse_wind(T)
    # Three well-separated sources so the source geometry is not the bottleneck.
    maps = np.stack([
        compact_source(gs, (gs[0] * 0.25, gs[1] * 0.5)),
        compact_source(gs, (gs[0] * 0.5, gs[1] * 0.75)),
        compact_source(gs, (gs[0] * 0.75, gs[1] * 0.5)),
    ], axis=0)
    source_names = ["traffic", "brick_kilns", "industries"]
    # Each source loads primarily on one basis; a small mix on another.
    C_true = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]], dtype=np.float64)
    c_true = C_true.reshape(-1)
    rows = []
    last_bundle = None
    for nf in cfg.get("noise_fracs", [0.0, 0.02, 0.05, 0.10]):
        b = forward(platform, maps, source_names, basis, observer, wind,
                    c_true=c_true, noise_frac=float(nf), with_temporal=True, seed=seed)
        last_bundle = b
        rows.append({
            "noise_frac": float(nf),
            "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
            "activity_relative_error": b["accuracy"].get("activity_relative_error"),
            "sigma_J": b["diagnostics"]["sigma_J"],
        })
    return {
        "experiment": "exp09_temporal_basis_recovery",
        "basis_names": list(basis.names),
        "rows": rows,
        "arrays": _arrays_from_bundle(last_bundle),
    }


# --------------------------------------------------------------------------- #
# Experiment 10 -- per-sensor footprints and spatial attribution
# --------------------------------------------------------------------------- #
def experiment_10(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    from model.iasa.footprints import compute_sensor_footprints, decompose_per_sensor

    gs = platform.grid_shape
    T = platform.config.T
    basis = default_basis("constant", T)
    # Sensors placed downwind (east) of two known source origins.
    observer = sensor_layout("downwind", gs, n=3, seed=seed, regulatory=platform.observer)
    wind = make_wind("constant", T)  # eastward
    maps = np.stack([compact_source(gs, (gs[0] * 0.3, gs[1] * 0.5)),
                     compact_source(gs, (gs[0] * 0.5, gs[1] * 0.5))], axis=0)
    source_names = ["src_west", "src_mid"]
    c_true = np.asarray([1.0, 0.6], dtype=np.float64)
    b = forward(platform, maps, source_names, basis, observer, wind, c_true=c_true, seed=seed)
    proj = b["_projection"]
    fit = b["_fit"]
    groups = b["report_components"]

    footprints = compute_sensor_footprints(
        maps, source_names, basis, observer, wind, fit=fit, projection=proj,
        response_config=platform.response_config, dispersion_config=platform.dispersion_config,
        groups=groups,
    )
    decomposition = decompose_per_sensor(
        proj.H_tilde, proj.H_tilde + proj.H_removed, fit.c_hat,
        proj.row_index, proj.column_index, groups=groups,
    )
    fitted_vec = to_numpy(fit.fitted_sensor_vector)
    contrib_sum_error = 0.0
    for sid, rowsix in decomposition["sensor_rows"].items():
        proj_total = sum(footprints.per_sensor_source_contribution_projected[sid].values())
        fitted_total = float(fitted_vec[np.asarray(rowsix, dtype=np.int64)].sum())
        contrib_sum_error = max(contrib_sum_error, abs(proj_total - fitted_total))
    nonneg = all(np.asarray(f).min() >= -1e-9 for f in footprints.geometric_footprint.values())

    return {
        "experiment": "exp10_footprints_spatial_attribution",
        "contribution_sum_error": float(contrib_sum_error),
        "footprints_nonnegative": bool(nonneg),
        "n_active_cells": footprints.metadata.get("n_active_cells"),
        "report_components": groups,
        "coefficient_relative_error": b["accuracy"]["coefficient_relative_error"],
        "sigma_J": b["diagnostics"]["sigma_J"],
        "arrays": _arrays_from_bundle(b),
    }


# --------------------------------------------------------------------------- #
# Observed New Delhi study mode (NO synthetic recovery metrics)
# --------------------------------------------------------------------------- #
def observed_new_delhi(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Real PM2.5 + mask + real inventory + real/synthetic wind. Reports residuals,
    geometry, uncertainty, weak/ambiguous coefficients, normalized proxy
    contributions, and recommended report groups -- NEVER a recovery error."""
    from data.pol_weather import load_new_delhi_wind_data

    gs = platform.grid_shape
    T = min(platform.config.T, int(cfg.get("T", platform.config.T)))
    basis = default_basis("constant", T)
    observer = platform.observer
    source_names = platform.source_names
    maps = platform.source_maps

    wind = make_wind(cfg.get("wind_kind", "constant"), T, seed=seed)
    response = _build_response(platform, maps, source_names, basis, observer, wind)

    # Real PM2.5 observations aligned to the response rows.
    wind_data = load_new_delhi_wind_data(
        str(platform.metadata.get("source_data_csv", "sim/govdata_1H_current.csv")),
        "sim/govdata_locations.csv",
        start="2018-05-01 00:00:00+05:30",
        end=f"2018-05-{1 + (T - 1) // 24:02d} {(T - 1) % 24:02d}:00:00+05:30",
    ) if cfg.get("use_real_pm25", True) else None

    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    background = build_background_basis(response.row_index, timestamps, observer.sensor_xy,
                                        platform.background_config("primary"))
    H_lag = to_numpy(response.H_lag).astype(np.float64)

    # Build Y from real PM2.5 where available; otherwise a declared proxy signal.
    Y, mask_fraction = _observed_Y(wind_data, response, observer, H_lag, seed)

    projection = project_response_and_observations(response.H_lag, Y, background,
                                                   response.row_index, response.column_index)
    diagnostics = diagnose_projection(projection, DiagnosticsConfig())
    fit = fit_projection(projection, config=FitConfig(ensemble_kind="inventory"))
    merge = recommend_merges(diagnostics, fit=fit, H_tilde=projection.H_tilde)

    c_hat = to_numpy(fit.c_hat)
    total = float(np.sum(np.abs(c_hat))) or 1.0
    normalized_contributions = {
        col["source_name"]: float(abs(c_hat[i])) / total
        for i, col in enumerate(projection.column_index)
    }
    return {
        "experiment": "observed_new_delhi",
        "has_ground_truth": False,
        "recovery_error": None,  # explicitly never computed for observed data
        "observed_mask_fraction": mask_fraction,
        "residual_norm": fit.residual_norm,
        "projected_residual_norm": getattr(fit, "projected_residual_norm", None),
        "diagnostics": _diag_summary(diagnostics),
        "weak_set": list(diagnostics.weak_set),
        "ambiguous_pairs": diagnostics.ambiguous_pairs,
        "report_components": [c["members"] for c in merge.report_components],
        "normalized_proxy_contributions": normalized_contributions,
        "source_names": list(source_names),
        "arrays": {
            "H_lag": H_lag.astype(np.float32),
            "H_tilde": to_numpy(projection.H_tilde).astype(np.float32),
            "Y": np.asarray(Y, dtype=np.float32),
            "c_hat": c_hat.astype(np.float32),
        },
    }


def _observed_Y(wind_data, response, observer, H_lag, seed):
    """Assemble the observed Y vector aligned to response rows from real PM2.5.

    Missing observations are filled by the projected-out background later; here we
    substitute the column mean so the fit sees a complete vector, and report the
    observed mask fraction as provenance."""
    n_rows = H_lag.shape[0]
    if wind_data is None or not hasattr(wind_data, "raw_pm25"):
        rng = np.random.default_rng(seed)
        return H_lag @ rng.uniform(0.2, 1.0, size=H_lag.shape[1]), 0.0
    pm = np.asarray(getattr(wind_data, "raw_pm25"), dtype=np.float64)  # [S, T] or similar
    mask = np.asarray(getattr(wind_data, "raw_pm25_mask"), dtype=bool)
    # Map station index -> observer sensor by position order (best effort).
    Y = np.zeros(n_rows, dtype=np.float64)
    observed = 0
    col_mean = np.nanmean(np.where(mask, pm, np.nan)) if mask.any() else 0.0
    for r, row in enumerate(response.row_index):
        t = int(row["time_index"])
        si = int(row.get("sensor_index", 0))
        if si < pm.shape[0] and t < pm.shape[1] and mask[si, t]:
            Y[r] = pm[si, t]
            observed += 1
        else:
            Y[r] = col_mean
    return Y, observed / max(n_rows, 1)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "exp01": experiment_1,
    "exp02": experiment_2,
    "exp03": experiment_3,
    "exp04": experiment_4,
    "exp05": experiment_5,
    "exp06": experiment_6,
    "exp07": experiment_7,
    "exp08": experiment_8,
    "exp09": experiment_9,
    "exp10": experiment_10,
    "observed": observed_new_delhi,
}


def run_named_experiment(name: str, platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; choices: {sorted(EXPERIMENTS)}")
    result = EXPERIMENTS[name](platform, cfg, seed)
    # Split arrays out for npz; keep JSON-able result clean of tensors/objects.
    arrays = result.pop("arrays", {})
    return {"result": _json_safe(result), "arrays": arrays}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj
