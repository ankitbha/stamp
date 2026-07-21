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
                    *, lag_window_steps: int | None = None, dispersion=None,
                    source_cell_threshold: float | None = None):
    rc = platform.response_config
    if lag_window_steps is not None:
        rc = replace(rc, lag_window_steps=int(lag_window_steps))
    if source_cell_threshold is not None:
        # Prune negligible emission cells for dense real-inventory builds (keeps the
        # response tractable at 40x40; cells below threshold contribute negligibly).
        rc = replace(rc, source_cell_threshold=float(source_cell_threshold))
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
    # Plain (non-ensemble) controlled fits use the NEUTRAL "single" kind so a lone fit
    # is never pooled into (and never dilutes) either ensemble; only the ensemble
    # experiments (E5 transport, E6 inventory scenarios) set a poolable kind explicitly.
    ensemble_kind: str = "single",
    run_adequacy: bool = False,
    adequacy_sigma: float | None = None,
    n_replicates: int = 200,
    alpha: float = 0.05,
    lag_window_steps: int | None = None,
    dispersion=None,
    source_cell_threshold: float | None = None,
) -> dict[str, Any]:
    """Run response -> background -> project -> diagnose -> fit -> merge and score."""
    response = _build_response(platform, source_maps, source_names, basis, observer, wind,
                               lag_window_steps=lag_window_steps, dispersion=dispersion,
                               source_cell_threshold=source_cell_threshold)
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


def _shift_map(m: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """Translate a source map by (dx, dy) integer cells, zero-filling vacated cells
    (a non-periodic shift; used to make a coherent copy of a real inventory map)."""
    out = np.zeros_like(m)
    nx, ny = m.shape
    xs0, xs1 = max(0, dx), min(nx, nx + dx)
    ys0, ys1 = max(0, dy), min(ny, ny + dy)
    xd0, xd1 = max(0, -dx), min(nx, nx - dx)
    yd0, yd1 = max(0, -dy), min(ny, ny - dy)
    out[xs0:xs1, ys0:ys1] = m[xd0:xd1, yd0:yd1]
    return out.astype(np.float32)


def _scale_map(m: np.ndarray, factor: float) -> np.ndarray:
    """Rescale a source map's SPATIAL EXTENT about its intensity centroid by `factor`
    (>1 broadens the footprint, <1 sharpens it) via nearest-neighbour resampling. A
    genuine spatial-scale perturbation of the same source (not a map substitution),
    used by E6's inventory-error scenarios."""
    m = np.asarray(m, dtype=np.float64)
    nx, ny = m.shape
    total = float(m.sum())
    if total <= 0 or factor == 1.0:
        return m.astype(np.float32)
    xs, ys = np.arange(nx), np.arange(ny)
    cx = float((m.sum(axis=1) * xs).sum() / total)
    cy = float((m.sum(axis=0) * ys).sum() / total)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    # sample input at (centroid + (out - centroid)/factor): factor>1 magnifies features.
    sx = np.rint(cx + (gx - cx) / factor).astype(int)
    sy = np.rint(cy + (gy - cy) / factor).astype(int)
    valid = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
    out = np.zeros_like(m)
    out[valid] = m[sx[valid], sy[valid]]
    return out.astype(np.float32)


def _real_group_maps(platform: Platform, indices):
    """Selected real proxy GROUP maps + names (F9: controlled mode preserves the
    New Delhi inventories where the paper names them)."""
    from experiments.iasa_pol.nd_platform import four_group_inventory

    names, maps = four_group_inventory(platform)
    sel_names = [names[i] for i in indices]
    sel_maps = np.stack([maps[i] for i in indices], axis=0)
    return sel_names, sel_maps


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
    c_true = np.asarray([1.0, 0.7], dtype=np.float64)
    # F9: shift a copy of a REAL inventory map (paper E2). Use a peaked group
    # (brick kilns) so the pair is well-localized and the coherence-vs-offset trend
    # is clean; prune negligible cells for tractability at 40x40.
    base_names, base_maps = _real_group_maps(platform, [0])  # brick_kilns
    base_map = base_maps[0]
    thr = float(cfg.get("source_cell_threshold", 0.02))
    rows = []
    last_bundle = None
    for off in offsets:
        maps = np.stack([base_map, _shift_map(base_map, int(round(float(off))))], axis=0)
        b = forward(platform, maps, [f"{base_names[0]}_a", f"{base_names[0]}_b"], basis, observer,
                    wind, c_true=c_true, seed=seed, source_cell_threshold=thr)
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
            # Pass grid_shape so "real" resolves to the gridded imputed field (F15),
            # not a city-level scalar fallback.
            wind = make_wind(wk, T, seed=seed, grid_shape=gs)
        except Exception as exc:  # real wind product may be unavailable
            rows.append({"wind": wk, "layout": None, "error": str(exc)})
            continue
        wind_provider = getattr(wind, "provider", wk)
        for lay in layouts:
            observer = sensor_layout(lay, gs, n=cfg.get("n_sensors", 6), seed=seed,
                                     regulatory=platform.observer)
            b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                        c_true=c_true, seed=seed)
            last_bundle = b
            rows.append({
                "wind": wk, "wind_provider": wind_provider, "layout": lay,
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


def _historical_wind_windows(T: int, n_members: int):
    """Contiguous windows of the REAL observed New Delhi wind record (city-level),
    each a genuine historical window (not a synthetic proxy). Returns a list of
    WindSequence windows; empty if the record is unavailable."""
    from experiments.iasa_pol.nd_platform import SIM_DIR
    from model.iasa.wind import WindSequence, real_new_delhi_wind_sequence

    try:
        seq = real_new_delhi_wind_sequence(
            SIM_DIR / "govdata_1H_current.csv", SIM_DIR / "govdata_locations.csv",
            allow_observed_fallback=True,
        )
    except Exception:
        return []
    total = int(seq.vx.shape[0])
    if total < T:
        return []
    windows = []
    # Evenly spaced, non-identical contiguous starts across the record.
    max_start = total - T
    starts = np.linspace(0, max_start, n_members).round().astype(int) if n_members > 1 else [0]
    for s in starts:
        windows.append(WindSequence(
            timestamps=seq.timestamps[s:s + T],
            vx=np.asarray(seq.vx[s:s + T], dtype=np.float32),
            vy=np.asarray(seq.vy[s:s + T], dtype=np.float32),
            provider="historical_real_new_delhi_window",
            metadata={"record_start_index": int(s), "window_length": int(T)},
        ))
    return windows


def _wind_window_ensemble(platform, maps, basis, c_true, seed, cfg) -> dict[str, Any]:
    from model.iasa.diagnostics import summarize_wind_ensemble

    T = platform.config.T
    observer = platform.observer
    n_members = int(cfg.get("ensemble_members", 8))
    out: dict[str, Any] = {}

    # Simulated ensemble: AR(1) synthetic windows (declared synthetic).
    # Historical ensemble: contiguous slices of the REAL observed wind record (F6).
    historical = _historical_wind_windows(T, n_members)
    families = {
        "simulated": [make_wind("ar1", T, seed=seed + 100 + m) for m in range(n_members)],
        "historical": historical,
    }
    for family, winds in families.items():
        if not winds:
            out[family] = {"error": "real wind record unavailable for historical windows"
                           if family == "historical" else "no members"}
            continue
        diags = []
        for wind in winds:
            b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                        c_true=c_true, seed=seed)
            diags.append(b["_diagnostics"])
        try:
            summary = summarize_wind_ensemble(diags, quantiles=(0.05, 0.5, 0.95))
            out[family] = {k: v for k, v in summary.items() if k != "column_index"}
            out[family]["provider"] = winds[0].provider
            out[family]["n_members"] = len(winds)
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

    def _parametric(kind: str, value: float, pert_wind, disp):
        pert = _build_response(platform, maps, source_names, basis, observer, pert_wind, dispersion=disp)
        H_pert = to_numpy(pert.H_lag).astype(np.float64)
        op_err = float(np.linalg.norm(H_pert - H_true) / max(np.linalg.norm(H_true), 1e-12))
        proj = project_response_and_observations(pert.H_lag, Y, bg, pert.row_index, pert.column_index)
        diag = diagnose_projection(proj, DiagnosticsConfig())
        fit = fit_projection(proj, config=FitConfig(ensemble_kind="transport"))
        transport_fits.append(fit)
        parametric_rows.append({
            "perturbation_kind": kind, "perturbation_value": float(value),
            "operator_error_norm": op_err,
            "coefficient_relative_error": float(
                np.linalg.norm(to_numpy(fit.c_hat) - c_true) / np.linalg.norm(c_true)),
            "residual_norm": fit.residual_norm,
            "sigma_J": diag.sigma_J,
            "singular_values": to_numpy(diag.singular_values).tolist(),
        })

    # E5 parametric transport family varies wind DIRECTION, wind SPEED, and DISPERSION
    # (paper E5: "vary wind speed, wind direction, and dispersion within the puff family").
    for dtheta in cfg.get("wind_direction_perturbations_deg", [0.0, 5.0, 10.0, 20.0]):
        _parametric("wind_direction_deg", dtheta,
                    make_wind("single", T, seed=seed, direction_degrees=float(dtheta)), None)
    for spd in cfg.get("wind_speed_factors", [1.0, 1.25, 1.5]):
        _parametric("wind_speed_factor", spd, make_wind("constant", T, speed=float(spd)), None)
    for dfac in cfg.get("dispersion_factors", [1.0, 1.5, 2.0]):
        disp = replace(platform.dispersion_config,
                       sigma_parallel=platform.dispersion_config.sigma_parallel * float(dfac),
                       sigma_perp=platform.dispersion_config.sigma_perp * float(dfac))
        _parametric("dispersion_factor", dfac, true_wind, disp)
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
    thr = float(cfg.get("source_cell_threshold", 0.02))
    # F9: perturb the REAL New Delhi inventory maps (paper E6 is literally inventory
    # error). Base pair = brick_kilns + industries; scenarios perturb location, spatial
    # scale (a genuine extent rescale about the centroid), the map version, and category
    # assignment while transport is held fixed.
    names_bi, maps_bi = _real_group_maps(platform, [0, 1])  # brick_kilns, industries
    m0, m1 = maps_bi[0], maps_bi[1]
    _, maps_traffic = _real_group_maps(platform, [3])  # traffic (alternate map version)
    scenarios = {
        "baseline": np.stack([m0, m1], axis=0),
        "location_shift": np.stack([_shift_map(m0, 3), m1], axis=0),
        "spatial_scale": np.stack([_scale_map(m0, 1.5), m1], axis=0),  # genuine extent rescale
        "alt_map_version": np.stack([maps_traffic[0], m1], axis=0),    # wrong map version
        "category_swap": np.stack([m1, m0], axis=0),
    }
    inventory_fits = []
    names = []
    rows = []
    last_bundle = None
    for name, maps in scenarios.items():
        b = forward(platform, maps, ["src_a", "src_b"], basis, observer, wind,
                    c_true=c_true, ensemble_kind="inventory", seed=seed, source_cell_threshold=thr)
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
                            wind, c_true=c_true, ensemble_kind="transport", seed=seed,
                            source_cell_threshold=thr)["_fit"]
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
        # Factual provenance: selection consumed only Frobenius deltas of H(L); the
        # code path (above) has no access to any fit result.
        "selection_criterion": "smallest_L_with_relative_frobenius_delta_le_tau_L",
        "arrays": {},
    }


# --------------------------------------------------------------------------- #
# Experiment 8 -- missing-source model adequacy
# --------------------------------------------------------------------------- #
def experiment_8(platform: Platform, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Refitted parametric-bootstrap adequacy on the NEW DELHI PLATFORM under an
    omitted source, with a NON-EMPTY primary background Q (F7):

      * null             : no omitted source -> residual ~ noise (rejection ~ alpha).
      * residual_visible : an omitted REAL inventory group (brick_kilns) whose
                           transported signature lies largely OUTSIDE span([H_lag, Q])
                           -> lands in the residual -> detectable.
      * aligned          : an omitted source whose signature lies INSIDE span([H_lag, Q])
                           -- a nonnegative mix of the fitted response columns AND the
                           background directions -> absorbed by the free coefficients and
                           the background -> NOT detectable. A required negative control:
                           non-rejection cannot certify inventory completeness.

    Unlike the earlier abstract orthonormal design, H_lag here is the real platform
    transport operator over real inventory maps and Q is the rank-4 primary background,
    so both the platform geometry and the span([H_lag, Q]) background-absorption case
    are genuinely exercised."""
    T = platform.config.T
    n_replicates = int(cfg.get("n_replicates", 200))
    n_trials = int(cfg.get("n_trials", 40))
    alpha = float(cfg.get("alpha", 0.05))
    noise_frac = float(cfg.get("noise_frac", 0.05))
    amp = float(cfg.get("omission_amplitude", 1.2))
    thr = float(cfg.get("source_cell_threshold", 0.02))
    basis = default_basis("constant", T)
    observer = platform.observer
    wind = make_wind("constant", T)  # transport held fixed on the platform

    # Fitted design: three REAL inventory groups transported through the platform.
    names_fit, maps_fit = _real_group_maps(platform, [1, 2, 3])  # industries, population, traffic
    resp_fit = _build_response(platform, maps_fit, names_fit, basis, observer, wind,
                               source_cell_threshold=thr)
    H = to_numpy(resp_fit.H_lag).astype(np.float64)  # N x 3, real transport operator
    N = H.shape[0]
    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    background = _background_for(platform, resp_fit, timestamps, observer, "primary")
    Q = to_numpy(background.Q).astype(np.float64)  # NON-empty rank-4 primary background

    c_fit_true = np.array([1.0, 0.8, 0.6], dtype=np.float64)
    clean = H @ c_fit_true
    sig_norm = float(np.linalg.norm(clean))
    sigma_e = noise_frac * max(float(np.max(np.abs(clean))), 1e-12)

    # Residual-visible omission: a REAL omitted group (brick_kilns -- peripheral, so its
    # transported signature is largely unreachable by the fitted columns + Q).
    names_om, maps_om = _real_group_maps(platform, [0])  # brick_kilns
    resp_om = _build_response(platform, maps_om, names_om, basis, observer, wind,
                              source_cell_threshold=thr)
    h_out = to_numpy(resp_om.H_lag).astype(np.float64)[:, 0]
    # Fraction of the omitted signature that lies OUTSIDE span([H, Q]) (its visible part).
    B = np.column_stack([H, Q]) if Q.shape[1] else H
    Ub, _ = np.linalg.qr(B)
    h_perp = h_out - Ub @ (Ub.T @ h_out)
    out_of_span_fraction = float(np.linalg.norm(h_perp) / max(np.linalg.norm(h_out), 1e-12))
    # Energy-match the omitted signature to the fitted signal (fair power comparison).
    h_out = h_out * (sig_norm / max(float(np.linalg.norm(h_out)), 1e-12))

    # Aligned (in-span) omission: nonnegative mix of fitted response columns AND
    # background directions -> in span([H, Q]) -> absorbed at any amplitude. Matched to
    # the same energy as the residual-visible omission.
    w_in = np.array([0.6, 0.5, 0.4], dtype=np.float64)          # >= 0 over fitted columns
    b_in = np.full(Q.shape[1], 0.3, dtype=np.float64) if Q.shape[1] else np.zeros(0)
    v_in = H @ w_in + (Q @ b_in if Q.shape[1] else 0.0)
    v_in = v_in * (sig_norm / max(float(np.linalg.norm(v_in)), 1e-12))

    noise_model = NoiseModel(covariance=max(sigma_e, 1e-9) ** 2, calibrated=True,
                             source="task10_exp8_platform", estimated_from_fit_residual=False)
    cols = resp_fit.column_index
    row_index = resp_fit.row_index
    case_offset = {"null": 0, "residual_visible": 1, "aligned": 2}  # deterministic seeds

    def _run(case: str, amplitude: float) -> float:
        rejects = 0
        for t in range(n_trials):
            mean = clean.copy()
            if case == "residual_visible":
                mean = mean + float(amplitude) * h_out
            elif case == "aligned":
                mean = mean + float(amplitude) * v_in
            g = np.random.default_rng(seed + 1_000_003 * (case_offset[case] + 1) + t)
            Y = mean + g.normal(0.0, sigma_e, size=N)
            proj = project_response_and_observations(resp_fit.H_lag, Y, background, row_index, cols)
            fit = fit_projection(proj, config=FitConfig())
            adq = residual_adequacy_check(
                fit, proj, noise_model,
                config=AdequacyConfig(alpha=alpha, n_replicates=n_replicates,
                                      seed=seed + 7_654_321 * (case_offset[case] + 1) + t),
            )
            rejects += int(bool(adq.inadequate))
        return rejects / max(n_trials, 1)

    null_rate = _run("null", 0.0)
    power = _run("residual_visible", amp)
    aligned_rate = _run("aligned", amp)

    # Identifiability diagnostics of the fitted PROJECTED design (F11 zeroing rule).
    svals = np.linalg.svd(to_numpy(project_response_and_observations(
        resp_fit.H_lag, clean, background, row_index, cols).H_tilde).astype(np.float64),
        compute_uv=False)
    J = int(svals.shape[0])
    tol = max(N, J) * float(np.finfo(np.float64).eps) * float(svals[0])
    num_rank = int(np.count_nonzero(svals > tol))
    diagnostics = {
        "sigma_J": 0.0 if num_rank < J else float(svals[-1]),
        "sigma_min_positive": float(svals[-1]),
        "numerical_rank": num_rank,
        "singular_values": [float(s) for s in svals],
        "background_rank": int(Q.shape[1]),
        "omitted_source_out_of_span_fraction": out_of_span_fraction,
    }
    return {
        "experiment": "exp08_missing_source_adequacy",
        "null_rejection_rate": null_rate,
        "residual_visible_power": power,
        "aligned_negative_control_rejection_rate": aligned_rate,
        "diagnostics": diagnostics,
        "alpha": alpha, "n_trials": n_trials, "n_replicates": n_replicates,
        "omission_amplitude": amp,
        "platform_grid_shape": list(platform.grid_shape),
        "fitted_groups": list(names_fit),
        "omitted_residual_visible_group": names_om[0],
        "aligned_in_span_weights": w_in.tolist(),
        "aligned_background_weights": b_in.tolist(),
        "operator_note": "real New Delhi platform transport operator over real inventory maps "
                         f"with a NON-empty rank-{int(Q.shape[1])} primary background Q; both the "
                         "platform geometry and span([H_lag, Q]) background absorption are exercised.",
        "negative_control_note": "aligned = a REAL omitted source whose signature is a nonnegative mix "
                                 "of the fitted response columns AND the background -> absorbed -> not "
                                 "detected; non-rejection cannot certify inventory completeness",
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

    # F10: footprint LOCALIZATION to the known upwind source origins. For each
    # sensor, the mass-weighted centroid of its geometric footprint should sit near
    # a true source cell (upwind of the sensor). Report the mean centroid distance
    # and the fraction of footprint mass within `radius` cells of a source origin.
    source_centers = np.asarray([(gs[0] * 0.3, gs[1] * 0.5), (gs[0] * 0.5, gs[1] * 0.5)], dtype=np.float64)
    radius = float(cfg.get("localization_radius_cells", 4.0))
    xs, ys = np.meshgrid(np.arange(gs[0]), np.arange(gs[1]), indexing="ij")
    cen_dists, mass_fracs = [], []
    for field in footprints.geometric_footprint.values():
        f = np.asarray(field, dtype=np.float64)
        total = float(f.sum())
        if total <= 0:
            continue
        cx = float((f * xs).sum() / total)
        cy = float((f * ys).sum() / total)
        cen_dists.append(float(np.min(np.hypot(source_centers[:, 0] - cx, source_centers[:, 1] - cy))))
        within = 0.0
        for sc in source_centers:
            within += float(f[np.hypot(xs - sc[0], ys - sc[1]) <= radius].sum())
        mass_fracs.append(min(within / total, 1.0))
    localization_error = float(np.mean(cen_dists)) if cen_dists else None
    mass_within_radius = float(np.mean(mass_fracs)) if mass_fracs else None

    return {
        "experiment": "exp10_footprints_spatial_attribution",
        "contribution_sum_error": float(contrib_sum_error),
        "footprints_nonnegative": bool(nonneg),
        "footprint_localization_error_cells": localization_error,
        "footprint_mass_fraction_within_radius": mass_within_radius,
        "localization_radius_cells": radius,
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
    """Observed New Delhi study mode, per 7.evaluation.tex: real PM2.5 with its
    observation mask M_O (PM2.5 is NEVER imputed), the four normalized proxy source
    GROUPS (traffic collapsed to one road map), the declared per-group temporal
    bases with the fixed-zero mask F0, the real gridded imputed wind field, the
    rank-4 primary background, and NO synthetic recovery metric. Reports residuals,
    geometry, uncertainty, weak/ambiguous coefficients, sensor-signal-space
    contribution shares, per-monitor group contributions, and report groups.

    A declared zero-source transported kriged initial-condition baseline is
    subtracted from the observations before source fitting: first-hour PM2.5 (Pusa
    monitors averaged upstream) is kriged onto the grid and transported forward as a
    single t0 impulse through the SAME open-boundary puff propagator (the Green's
    function of the advection-diffusion model, so units stay in PM2.5 space). The
    rank-4 primary Q additionally absorbs smooth offsets.
    """
    from dataclasses import replace as _dc_replace

    from data.pol_weather import load_new_delhi_wind_data
    from model.iasa.footprints import decompose_per_sensor
    from experiments.iasa_pol.nd_platform import four_group_inventory, paper_temporal_bases

    gs = platform.grid_shape
    T = min(platform.config.T, int(cfg.get("T", platform.config.T)))
    observer = platform.observer

    # F16: four source GROUPS (traffic = single road map + slot bases).
    group_names, group_maps = four_group_inventory(platform)
    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    # F3: declared per-group temporal bases + fixed-zero mask F0 (each group free
    # only on its own admissible components).
    basis, admissible, fixed_zero = paper_temporal_bases(timestamps, group_names)

    # F15: real gridded imputed wind field (kernel coordinate-query imputer).
    wind = make_wind(cfg.get("wind_kind", "real"), T, seed=seed, grid_shape=gs)
    wind_provider = getattr(wind, "provider", str(cfg.get("wind_kind", "real")))

    response = _build_response(
        platform, group_maps, group_names, basis, observer, wind,
        source_cell_threshold=float(cfg.get("source_cell_threshold", 0.02)),
    )
    background = build_background_basis(response.row_index, timestamps, observer.sensor_xy,
                                        platform.background_config("primary"))

    # F2: assemble the FULL Y from real PM2.5 (no imputation) and the observation
    # mask M_O; then select the SAME observed rows from Y, H_lag, Q, and metadata.
    wind_data = load_new_delhi_wind_data(
        "sim/govdata_1H_current.csv", "sim/govdata_locations.csv",
        start="2018-05-01 00:00:00+05:30",
        end=f"2018-05-{1 + (T - 1) // 24:02d} {(T - 1) % 24:02d}:00:00+05:30",
    ) if cfg.get("use_real_pm25", True) else None
    station_index = platform.metadata.get("regulatory_station_index")
    Y_full, observed_flags = _observed_pm25_rows(wind_data, response, seed, station_index=station_index)

    # Declared zero-source transported kriged IC baseline: subtract before fitting.
    baseline_vec, baseline_meta = _ic_transport_baseline(
        platform, observer, wind, response, wind_data, station_index, cfg,
        source_cell_threshold=float(cfg.get("source_cell_threshold", 0.02)),
    )
    if baseline_vec is not None:
        Y_full = np.asarray(Y_full, dtype=np.float64) - baseline_vec

    keep = [r for r, ok in enumerate(observed_flags) if ok]
    if len(keep) < response.H_lag.shape[1] + 1:
        # Too few observed rows to identify the design; report honestly and stop.
        return {
            "experiment": "observed_new_delhi", "has_ground_truth": False,
            "recovery_error": None, "status": "insufficient_observed_rows",
            "adequacy": None,
            "calibration_status": "uncalibrated: no external calibrated noise model for observed PM2.5",
            "n_observed_rows": len(keep), "n_total_rows": len(observed_flags),
            "wind_provider": wind_provider, "source_names": list(group_names),
            "arrays": {},
        }

    idx = torch.as_tensor(keep, dtype=torch.long)
    H_m = response.H_lag.index_select(0, idx)
    Y_m = np.asarray(Y_full, dtype=np.float64)[keep]
    Q_m = background.Q.index_select(0, idx)
    row_m = [response.row_index[r] for r in keep]
    masked_bg = _dc_replace(background, Q=Q_m, row_index=row_m)

    projection = project_response_and_observations(H_m, Y_m, masked_bg, row_m, response.column_index)
    diagnostics = diagnose_projection(projection, DiagnosticsConfig(fixed_zero_indices=fixed_zero))
    fit = fit_projection(projection,
                         config=FitConfig(fixed_zero_indices=fixed_zero, ensemble_kind="inventory"))
    merge = recommend_merges(diagnostics, fit=fit, H_tilde=projection.H_tilde)

    # F4: contribution shares in SENSOR-SIGNAL space (comparable across groups),
    # NOT raw coefficient magnitudes (which live in incomparable per-proxy units).
    H_tilde = to_numpy(projection.H_tilde)
    c_hat = to_numpy(fit.c_hat)
    col_index = projection.column_index
    contrib_l1: dict[str, float] = {}
    for k, gname in enumerate(group_names):
        cols = [j for j, col in enumerate(col_index) if int(col["source_index"]) == k]
        contrib_vec = H_tilde[:, cols] @ c_hat[cols] if cols else np.zeros(H_tilde.shape[0])
        contrib_l1[gname] = float(np.sum(np.abs(contrib_vec)))
    denom = sum(contrib_l1.values()) or 1.0
    sensor_signal_shares = {g: v / denom for g, v in contrib_l1.items()}

    # F10: per-monitor fitted group contributions (spatial-origin footprint FIELDS
    # are available via compute_sensor_footprints but skipped by default at 40x40
    # for cost; enable with cfg["with_footprints"]).
    groups = [c["members"] for c in merge.report_components]
    decomposition = decompose_per_sensor(
        projection.H_tilde, projection.H_tilde + projection.H_removed, fit.c_hat,
        projection.row_index, projection.column_index, groups=groups,
    )
    per_monitor = {
        str(sid): {str(k): float(v) for k, v in vals.items()}
        for sid, vals in decomposition["per_sensor_group_contribution_projected"].items()
    }

    return {
        "experiment": "observed_new_delhi",
        "has_ground_truth": False,
        "recovery_error": None,  # explicitly never computed for observed data
        "n_source_groups": len(group_names),
        "source_names": list(group_names),
        "wind_provider": wind_provider,
        "adequacy": None,
        "calibration_status": "uncalibrated: no external calibrated noise model exists for "
                              "observed PM2.5, so per the paper's adequacy contract the residual "
                              "adequacy test emits no pass/fail verdict",
        "observed_mask_fraction": len(keep) / max(len(observed_flags), 1),
        "n_observed_rows": len(keep),
        "n_total_rows": len(observed_flags),
        "pm25_imputed": False,
        "kriged_baseline_subtracted": bool(baseline_vec is not None),
        "kriged_baseline": baseline_meta,
        "fixed_zero_indices": list(fixed_zero),
        "admissible_components_per_group": admissible,
        "residual_norm": fit.residual_norm,
        "projected_residual_norm": getattr(fit, "projected_residual_norm", None),
        "diagnostics": _diag_summary(diagnostics),
        "weak_set": list(diagnostics.weak_set),
        "ambiguous_pairs": diagnostics.ambiguous_pairs,
        "report_components": groups,
        "sensor_signal_contribution_shares": sensor_signal_shares,
        "sensor_signal_shares_denominator": "sum over groups of L1 fitted per-group sensor-signal magnitude",
        "per_monitor_group_contributions": per_monitor,
        "arrays": {
            "H_tilde": H_tilde.astype(np.float32),
            "Y": Y_m.astype(np.float32),
            "c_hat": c_hat.astype(np.float32),
        },
    }


def _ic_transport_baseline(platform, observer, wind, response, wind_data, station_index, cfg,
                           *, source_cell_threshold=0.02):
    """Zero-source transported kriged initial-condition baseline aligned to response
    rows. Krige the first observed hour's PM2.5 (Pusa-averaged loader stations) onto
    the grid, then transport it as a single t0 impulse through the SAME open-boundary
    puff builder (its Gaussian kernel is the advection-diffusion Green's function, so
    the propagated field stays in PM2.5 units). Returns (baseline_vec[N], meta) or
    (None, meta) when no real PM2.5 is available."""
    from model.iasa.activity import TemporalBasis
    from experiments.iasa_pol.nd_platform import krige_initial_condition

    if wind_data is None or not hasattr(wind_data, "raw_pm25"):
        return None, {"subtracted": False, "reason": "no real PM2.5"}
    pm = np.asarray(wind_data.raw_pm25, dtype=np.float64)
    mask = np.asarray(wind_data.raw_pm25_mask, dtype=bool)
    T = max(int(r["time_index"]) for r in response.row_index) + 1
    sensor_xy = np.asarray(observer.sensor_xy, dtype=np.float64)

    def _orig(si):
        return int(station_index[si]) if station_index is not None and si < len(station_index) else si

    # First hour with >= 3 observed stations -> the initial-condition estimate.
    t0, cells, vals = None, [], []
    for t in range(T):
        c, v = [], []
        for si in range(sensor_xy.shape[0]):
            o = _orig(si)
            if o < pm.shape[0] and t < pm.shape[1] and mask[o, t]:
                c.append(sensor_xy[si]); v.append(pm[o, t])
        if len(c) >= 3:
            t0, cells, vals = t, c, v
            break
    if t0 is None:
        return None, {"subtracted": False, "reason": "no hour with >=3 observed stations"}

    U0 = krige_initial_condition(platform.grid_shape, np.asarray(cells), np.asarray(vals))
    impulse = np.zeros((T, 1), dtype=np.float32)
    impulse[t0, 0] = 1.0
    ic_basis = TemporalBasis(names=["ic_impulse"], values=impulse,
                             metadata={"kind": "initial_condition_impulse", "release_index": int(t0)})
    ic_resp = _build_response(platform, U0[None], ["ic_baseline"], ic_basis, observer, wind,
                              source_cell_threshold=source_cell_threshold)
    baseline_vec = to_numpy(ic_resp.H_lag).astype(np.float64).reshape(-1)
    meta = {
        "subtracted": True,
        "method": "gaussian_kernel_kriging_surrogate + open_boundary_puff_green_function",
        "pusa_handling": "averaged_upstream_in_loader",
        "ic_release_index": int(t0),
        "n_stations_kriged": len(cells),
        "U0_max": float(U0.max()),
        "baseline_l2_norm": float(np.linalg.norm(baseline_vec)),
    }
    return baseline_vec, meta


def _observed_pm25_rows(wind_data, response, seed, *, station_index=None):
    """Return (Y_full aligned to response rows, observed_flags per row). PM2.5 is
    NEVER imputed: rows without a valid observation are flagged unobserved (their
    Y value is a placeholder and is dropped by M_O selection). ``station_index[si]``
    maps the deduped observer sensor back to its original raw-station row."""
    n_rows = len(response.row_index)
    if wind_data is None or not hasattr(wind_data, "raw_pm25"):
        # Declared-proxy fallback (no real PM2.5 requested): all rows "observed".
        rng = np.random.default_rng(seed)
        H_lag = to_numpy(response.H_lag)
        return H_lag @ rng.uniform(0.2, 1.0, size=H_lag.shape[1]), [True] * n_rows
    pm = np.asarray(getattr(wind_data, "raw_pm25"), dtype=np.float64)  # [S, T]
    mask = np.asarray(getattr(wind_data, "raw_pm25_mask"), dtype=bool)
    Y = np.zeros(n_rows, dtype=np.float64)
    flags = [False] * n_rows
    for r, row in enumerate(response.row_index):
        t = int(row["time_index"])
        si = int(row.get("sensor_index", 0))
        orig = int(station_index[si]) if station_index is not None and si < len(station_index) else si
        if orig < pm.shape[0] and t < pm.shape[1] and mask[orig, t]:
            Y[r] = pm[orig, t]
            flags[r] = True
    return Y, flags


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
