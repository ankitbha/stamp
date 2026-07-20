"""Constrained end-to-end refinement (Task 9C, optional stage).

Jointly corrects the wind field ``phi``, dispersion parameters ``psi``, source
coefficients ``c >= 0``, and background coefficients ``gamma`` -- but only within
physical limits and only when identifiability is preserved. The fixed-response
IASA solution stays the default reported estimate; refinement is a constrained
local correction reported with its own diagnostics (paper 4.method.tex
subsec:constrained_refinement, 6.algorithm.tex subsec:refinement_acceptance).

Objective (eq. refinement_objective)

    L_refine = ||Y - H_lag(phi,psi) c - Q gamma||^2 + lambda_theta R(c)
             + lambda_w ||w_phi - w_phi0||^2 + lambda_psi ||psi - psi0||^2
             + lambda_sm R_sm(w_phi)

subject to (eq. refinement_constraints)

    c >= 0,   psi in Psi_phys,   ||w_phi - w_phi0||_inf <= eps_w.

For fixed (phi, psi, c), min_gamma ||Y - H_lag c - Q gamma||^2 =
||Y_tilde - H_tilde c||^2 (background projection identity), so the (c, gamma)
block is exactly ``fit_projection`` against the same background basis and
``L_refine`` reduces to a low-dimensional function of (phi, psi). The wind field
is a smooth additive correction w_phi(t) = w0(t) + Phi_corr[t] @ A over a small
correction basis; (phi, psi) are optimized by a deterministic coordinate pattern
search (no RNG), each evaluation rebuilding H_lag, re-projecting, and refitting.

Acceptance (eqs. refinement_smin_check / refinement_coherence_check): accept only
if sigma_J(H_tilde_ref) >= (1 - eta_id) sigma_J(H_tilde_0) and the maximum
eligible pairwise coherence rho_ref <= tau_rho^ref, with sigma_J taken as zero
when a response has fewer than J numerically nonzero singular values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from model.iasa.backend import runtime_provenance, to_numpy
from model.iasa.diagnostics import DiagnosticsConfig, diagnose_projection
from model.iasa.fit import FitConfig, FitResult, fit_projection, projected_data_objective


# --------------------------------------------------------------------------- #
# Configuration and result                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RefineConfig:
    # Objective weights.
    lambda_theta: float = 0.0        # R(c) ridge weight (inner FitConfig.lambda_reg)
    lambda_w: float = 1.0            # ||w_phi - w_phi0||^2 anchor
    lambda_psi: float = 1.0          # ||psi - psi0||^2 anchor
    lambda_sm: float = 0.0           # R_sm(w_phi) smoothness
    # Constraints.
    eps_w: float = 0.25              # L-inf bound on the wind-field correction
    psi_lower: tuple[float, float, float] = (0.1, 0.05, 0.05)   # Psi_phys lower box
    psi_upper: tuple[float, float, float] = (3.0, 2.0, 5.0)     # Psi_phys upper box
    # Acceptance thresholds.
    eta_id: float = 0.05             # allowed relative sigma_J drop
    tau_rho_ref: float | None = None  # default -> DiagnosticsConfig.tau_rho
    require_fit_improvement: bool = False  # also require data residual to not increase
    # Wind correction parameterization.
    refine_wind: bool = True
    refine_dispersion: bool = True
    correction_basis: str = "constant_linear"  # "constant" | "constant_linear"
    # Deterministic pattern-search optimizer.
    max_outer_iters: int = 40
    init_wind_step: float = 0.05
    init_psi_step: float = 0.05
    shrink: float = 0.5
    tol_step: float = 1e-4
    tol_obj: float = 1e-10
    # Passthrough configs.
    fit_config: FitConfig | None = None
    diagnostics_config: DiagnosticsConfig | None = None
    rank_atol_scale: float = 1.0     # numerical-rank tolerance follows diagnostics


@dataclass
class RefineResult:
    accepted: bool
    reason: str
    wind_correction: list[list[float]] | None      # A [P, 2] (None if wind fixed)
    wind_correction_field: list[list[float]] | None  # delta [T, 2] at integer times
    refined_psi: list[float]
    baseline_psi: list[float]
    refined_fit: FitResult
    fixed_response_fit: FitResult
    sigma_J_baseline_eff: float
    sigma_J_refined_eff: float
    sigma_J_threshold: float
    rho_baseline: float | None
    rho_refined: float | None
    tau_rho_ref: float
    rank_baseline: int
    rank_refined: int
    objective_start: dict[str, float]
    objective_end: dict[str, float]
    constraint_satisfaction: dict[str, Any]
    optimizer_trace: dict[str, Any]
    warnings: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "wind_correction": self.wind_correction,
            "wind_correction_field": self.wind_correction_field,
            "refined_psi": self.refined_psi,
            "baseline_psi": self.baseline_psi,
            "refined_fit": self.refined_fit.to_json_summary(),
            "fixed_response_fit": self.fixed_response_fit.to_json_summary(),
            "sigma_J_baseline_eff": self.sigma_J_baseline_eff,
            "sigma_J_refined_eff": self.sigma_J_refined_eff,
            "sigma_J_threshold": self.sigma_J_threshold,
            "rho_baseline": self.rho_baseline,
            "rho_refined": self.rho_refined,
            "tau_rho_ref": self.tau_rho_ref,
            "rank_baseline": self.rank_baseline,
            "rank_refined": self.rank_refined,
            "objective_start": self.objective_start,
            "objective_end": self.objective_end,
            "constraint_satisfaction": self.constraint_satisfaction,
            "optimizer_trace": self.optimizer_trace,
            "warnings": list(self.warnings),
            "config": self.config,
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Wind correction: smooth additive field over a small correction basis        #
# --------------------------------------------------------------------------- #
def _correction_basis(kind: str, T: int) -> np.ndarray:
    """Predeclared correction temporal basis Phi_corr [T, P]."""
    if T < 1:
        raise ValueError("T must be positive")
    const = np.ones((T, 1), dtype=np.float64)
    if kind == "constant":
        return const
    if kind == "constant_linear":
        if T == 1:
            return const
        lin = (np.arange(T, dtype=np.float64) / (T - 1) - 0.5).reshape(T, 1)
        return np.concatenate([const, lin], axis=1)
    raise ValueError("correction_basis must be 'constant' or 'constant_linear'")


def _delta_field(phi_corr: np.ndarray, A: np.ndarray) -> np.ndarray:
    """delta[t] = Phi_corr[t] @ A, shape [T, 2]."""
    return phi_corr @ A


class _CorrectedWindSampler:
    """Wraps a base WindSampler and adds a t-interpolated additive transport
    correction ``delta`` (position-independent). ``delta`` is defined at integer
    times ``0..T-1`` and linearly interpolated for fractional ``t``.
    """

    def __init__(self, base: Any, delta: np.ndarray) -> None:
        self._base = base
        self._delta = np.asarray(delta, dtype=np.float64)
        self._T = self._delta.shape[0]

    def sample(self, t_index: float, position_xy: Any) -> np.ndarray:
        base_val = np.asarray(self._base.sample(t_index, position_xy), dtype=np.float64).reshape(2)
        t = float(np.clip(t_index, 0.0, max(0, self._T - 1)))
        lo = int(np.floor(t))
        hi = min(lo + 1, self._T - 1)
        frac = t - lo
        d = (1.0 - frac) * self._delta[lo] + frac * self._delta[hi]
        return (base_val + d).astype(np.float32)


def _project_eps_w(A: np.ndarray, phi_corr: np.ndarray, eps_w: float) -> np.ndarray:
    """Scale A so max_t max_component |delta[t]| <= eps_w (delta linear in A)."""
    if A.size == 0:
        return A
    delta = _delta_field(phi_corr, A)
    peak = float(np.max(np.abs(delta))) if delta.size else 0.0
    if peak <= eps_w or peak == 0.0:
        return A
    return A * (eps_w / peak)


def _clamp_psi(psi: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(psi, lower), upper)


# --------------------------------------------------------------------------- #
# Candidate evaluation: rebuild -> re-project -> fit -> penalties             #
# --------------------------------------------------------------------------- #
def _sigma_J_eff(diag: Any) -> float:
    """sigma_J with the rank-deficiency rule: 0 when fewer than J nonzero SVs."""
    J = len(diag.reduced_to_original)
    if J == 0:
        return 0.0
    return float(diag.sigma_J) if diag.numerical_rank >= J else 0.0


def _max_eligible_coherence(diag: Any) -> float | None:
    """max over eligible pairs (i != j, i,j not in W) of coherence rho_ij."""
    coh = to_numpy(diag.coherence)
    if coh.size == 0 or coh.shape[0] < 2:
        return None
    iu = np.triu_indices(coh.shape[0], k=1)
    vals = coh[iu]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return float(vals.max())


class _Evaluator:
    """Builds the response for a candidate (A, psi), re-projects against the same
    background basis, fits sources, and returns the full L_refine breakdown.
    """

    def __init__(
        self,
        *,
        source_maps: np.ndarray,
        source_names: Sequence[str],
        basis: Any,
        observer: Any,
        base_sampler: Any,
        background_basis: Any,
        response_config: Any,
        base_dispersion: Any,
        phi_corr: np.ndarray,
        w0_field: np.ndarray,
        cfg: RefineConfig,
    ) -> None:
        self.source_maps = source_maps
        self.source_names = list(source_names)
        self.basis = basis
        self.observer = observer
        self.base_sampler = base_sampler
        self.background_basis = background_basis
        self.response_config = response_config
        self.base_dispersion = base_dispersion
        self.phi_corr = phi_corr
        self.w0_field = w0_field  # base transport field at integer times [T, 2]
        self.cfg = cfg
        self.fit_config = cfg.fit_config or FitConfig(lambda_reg=cfg.lambda_theta)

    def build_projection(self, A: np.ndarray, psi: np.ndarray) -> Any:
        from model.iasa.response import DispersionConfig, build_lagged_response_matrix
        from model.iasa.projection import project_response_and_observations

        delta = _delta_field(self.phi_corr, A)
        sampler = _CorrectedWindSampler(self.base_sampler, delta)
        dispersion = DispersionConfig(
            sigma_parallel=float(psi[0]), sigma_perp=float(psi[1]), min_dispersion_time=float(psi[2])
        )
        response = build_lagged_response_matrix(
            self.source_maps, self.source_names, self.basis, self.observer, sampler,
            response_config=self.response_config, dispersion_config=dispersion,
        )
        projection = project_response_and_observations(
            response.H_lag, self._Y, self.background_basis, response.row_index, response.column_index
        )
        return projection

    def set_observations(self, Y: np.ndarray) -> None:
        self._Y = np.asarray(Y, dtype=np.float64)

    def penalties(self, A: np.ndarray, psi: np.ndarray, psi0: np.ndarray) -> dict[str, float]:
        delta = _delta_field(self.phi_corr, A)
        wind_anchor = float(self.cfg.lambda_w) * float((delta * delta).sum())
        psi_anchor = float(self.cfg.lambda_psi) * float(((psi - psi0) ** 2).sum())
        w_phi = self.w0_field + delta
        if w_phi.shape[0] > 1:
            diff = w_phi[1:] - w_phi[:-1]
            smooth = float(self.cfg.lambda_sm) * float((diff * diff).sum())
        else:
            smooth = 0.0
        return {"wind_anchor": wind_anchor, "psi_anchor": psi_anchor, "smoothness": smooth}

    def evaluate(self, A: np.ndarray, psi: np.ndarray, psi0: np.ndarray) -> dict[str, Any]:
        projection = self.build_projection(A, psi)
        fit = fit_projection(projection, config=self.fit_config)
        obj = projected_data_objective(
            projection.H_tilde, projection.Y_tilde, fit.c_hat, lam=float(self.cfg.lambda_theta)
        )
        pen = self.penalties(A, psi, psi0)
        total = obj["total"] + pen["wind_anchor"] + pen["psi_anchor"] + pen["smoothness"]
        breakdown = {
            "data": obj["data"],
            "regularizer": obj["regularizer"],
            "wind_anchor": pen["wind_anchor"],
            "psi_anchor": pen["psi_anchor"],
            "smoothness": pen["smoothness"],
            "total": total,
        }
        return {"projection": projection, "fit": fit, "objective": breakdown}


# --------------------------------------------------------------------------- #
# Deterministic coordinate pattern search                                     #
# --------------------------------------------------------------------------- #
def _pattern_search(
    evaluator: _Evaluator,
    A0: np.ndarray,
    psi0: np.ndarray,
    cfg: RefineConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    P2 = A0.size
    n_psi = psi0.size

    def feasible(A: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Af = _project_eps_w(A, evaluator.phi_corr, cfg.eps_w) if cfg.refine_wind else np.zeros_like(A)
        psif = _clamp_psi(psi, lower, upper) if cfg.refine_dispersion else psi.copy()
        return Af, psif

    A_cur, psi_cur = feasible(A0.copy(), psi0.copy())
    best = evaluator.evaluate(A_cur, psi_cur, psi0)
    best_obj = best["objective"]["total"]
    n_eval = 1
    accepted_moves = 0

    # Coordinate layout: first P2 wind coeffs (if refine_wind), then psi (if refine_dispersion).
    coords: list[tuple[str, int]] = []
    if cfg.refine_wind:
        coords += [("A", i) for i in range(P2)]
    if cfg.refine_dispersion:
        coords += [("psi", i) for i in range(n_psi)]

    wind_step = float(cfg.init_wind_step)
    psi_step = float(cfg.init_psi_step)

    for _ in range(int(cfg.max_outer_iters)):
        improved = False
        for kind, idx in coords:
            step = wind_step if kind == "A" else psi_step
            for sign in (1.0, -1.0):
                A_try = A_cur.copy()
                psi_try = psi_cur.copy()
                if kind == "A":
                    A_try_flat = A_try.reshape(-1)
                    A_try_flat[idx] += sign * step
                    A_try = A_try_flat.reshape(A_cur.shape)
                else:
                    psi_try[idx] += sign * step
                A_try, psi_try = feasible(A_try, psi_try)
                cand = evaluator.evaluate(A_try, psi_try, psi0)
                n_eval += 1
                if cand["objective"]["total"] < best_obj - cfg.tol_obj:
                    best, best_obj = cand, cand["objective"]["total"]
                    A_cur, psi_cur = A_try, psi_try
                    improved = True
                    accepted_moves += 1
        if not improved:
            wind_step *= cfg.shrink
            psi_step *= cfg.shrink
            if max(wind_step, psi_step) < cfg.tol_step:
                break

    return {
        "A": A_cur,
        "psi": psi_cur,
        "best": best,
        "trace": {
            "n_evaluations": n_eval,
            "accepted_moves": accepted_moves,
            "final_wind_step": wind_step,
            "final_psi_step": psi_step,
        },
    }


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
def refine_end_to_end(
    source_maps: np.ndarray,
    source_names: Sequence[str],
    basis: Any,
    observer: Any,
    wind: Any,
    Y: np.ndarray,
    background_basis: Any,
    *,
    fixed_response_fit: FitResult,
    baseline_projection: Any,
    response_config: Any = None,
    dispersion_config: Any = None,
    config: RefineConfig | None = None,
) -> RefineResult:
    """Constrained local refinement of (phi, psi, c, gamma) around the
    fixed-response IASA solution.

    ``wind`` is the base ``WindSequence`` (or ``WindSampler``); ``Y`` the raw
    observations (pre-projection); ``background_basis`` the same
    ``BackgroundBasisResult`` used for the fixed-response projection; and
    ``fixed_response_fit`` / ``baseline_projection`` the default estimate to keep
    and improve upon.
    """
    from model.iasa.response import DispersionConfig, ResponseConfig
    from model.iasa.wind import WindSequence
    from model.iasa.response import CityWindSampler

    cfg = config or RefineConfig()
    diag_cfg = cfg.diagnostics_config or DiagnosticsConfig()
    tau_rho_ref = cfg.tau_rho_ref if cfg.tau_rho_ref is not None else diag_cfg.tau_rho
    warnings: list[str] = []

    maps = np.asarray(source_maps, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError("source_maps must have shape [K, Nx, Ny]")

    response_config = response_config or ResponseConfig()
    base_dispersion = dispersion_config or DispersionConfig()
    psi0 = np.asarray(
        [base_dispersion.sigma_parallel, base_dispersion.sigma_perp, base_dispersion.min_dispersion_time],
        dtype=np.float64,
    )
    lower = np.asarray(cfg.psi_lower, dtype=np.float64)
    upper = np.asarray(cfg.psi_upper, dtype=np.float64)
    if not (np.all(lower > 0) and np.all(upper >= lower)):
        raise ValueError("Psi_phys box must be positive with upper >= lower")
    if not (lower <= psi0).all() or not (psi0 <= upper).all():
        warnings.append("baseline dispersion psi0 lies outside Psi_phys; it will be clamped")

    # Base transport field at integer times (position-independent shift reference).
    base_sampler = (
        CityWindSampler.from_wind_sequence(wind, interpolation=response_config.wind_interpolation)
        if isinstance(wind, WindSequence)
        else wind
    )
    basis_values = getattr(basis, "values", None)
    if basis_values is None:
        basis_values = np.asarray(basis)
    T = int(np.asarray(basis_values).shape[0])
    center_xy = np.asarray(
        [float(np.mean(observer.sensor_xy[:, 0])), float(np.mean(observer.sensor_xy[:, 1]))],
        dtype=np.float64,
    )
    w0_field = np.stack(
        [np.asarray(base_sampler.sample(float(t), center_xy), dtype=np.float64).reshape(2) for t in range(T)],
        axis=0,
    )

    phi_corr = _correction_basis(cfg.correction_basis, T)
    P = phi_corr.shape[1]
    A0 = np.zeros((P, 2), dtype=np.float64)

    evaluator = _Evaluator(
        source_maps=maps, source_names=source_names, basis=basis, observer=observer,
        base_sampler=base_sampler, background_basis=background_basis,
        response_config=response_config, base_dispersion=base_dispersion,
        phi_corr=phi_corr, w0_field=w0_field, cfg=cfg,
    )
    evaluator.set_observations(Y)

    # Objective at the starting (fixed-response) point.
    start = evaluator.evaluate(A0, psi0, psi0)
    objective_start = start["objective"]

    search = _pattern_search(evaluator, A0, psi0, cfg, lower, upper)
    A_ref = search["A"]
    psi_ref = search["psi"]
    refined = search["best"]
    objective_end = refined["objective"]
    refined_projection = refined["projection"]
    refined_fit = refined["fit"]

    # Diagnostics: baseline vs refined.
    baseline_diag = diagnose_projection(baseline_projection, diag_cfg)
    refined_diag = diagnose_projection(refined_projection, diag_cfg)
    sigma_J0 = _sigma_J_eff(baseline_diag)
    sigma_Jr = _sigma_J_eff(refined_diag)
    rho0 = _max_eligible_coherence(baseline_diag)
    rhor = _max_eligible_coherence(refined_diag)
    threshold = (1.0 - float(cfg.eta_id)) * sigma_J0

    # Constraint satisfaction.
    delta_ref = _delta_field(phi_corr, A_ref)
    delta_inf = float(np.max(np.abs(delta_ref))) if delta_ref.size else 0.0
    eps_w_ok = delta_inf <= cfg.eps_w + 1e-9
    psi_in_box = bool((psi_ref >= lower - 1e-9).all() and (psi_ref <= upper + 1e-9).all())
    constraint_satisfaction = {
        "eps_w": float(cfg.eps_w),
        "wind_correction_inf_norm": delta_inf,
        "eps_w_satisfied": bool(eps_w_ok),
        "psi_lower": [float(v) for v in lower.tolist()],
        "psi_upper": [float(v) for v in upper.tolist()],
        "psi_in_box": psi_in_box,
        "c_nonnegative": bool(float(to_numpy(refined_fit.c_hat).min()) >= -1e-12),
    }

    # Acceptance (eqs. refinement_smin_check / refinement_coherence_check).
    sigma_ok = sigma_Jr >= threshold - 1e-12
    rho_ok = (rhor is None) or (rhor <= tau_rho_ref + 1e-12)
    fit_ok = (not cfg.require_fit_improvement) or (
        objective_end["data"] <= objective_start["data"] + 1e-9
    )
    accepted = bool(sigma_ok and rho_ok and fit_ok and eps_w_ok and psi_in_box)

    reasons: list[str] = []
    if not sigma_ok:
        reasons.append(f"sigma_J degraded: {sigma_Jr:.6g} < threshold {threshold:.6g}")
    if not rho_ok:
        reasons.append(f"eligible coherence {rhor:.6g} exceeds tau_rho_ref {tau_rho_ref:.6g}")
    if not fit_ok:
        reasons.append("data fit did not improve")
    if not eps_w_ok:
        reasons.append("wind correction exceeds eps_w")
    if not psi_in_box:
        reasons.append("psi left Psi_phys")
    reason = "accepted: constrained local correction preserves separability" if accepted else (
        "rejected: " + "; ".join(reasons)
    )

    device = baseline_projection.H_tilde.device
    dtype = baseline_projection.H_tilde.dtype
    metadata = {
        **runtime_provenance(device, dtype),
        "T": int(T),
        "correction_basis_dim": int(P),
        "n_sources": int(maps.shape[0]),
        "default_report": "fixed_response_fit",
        "note": "refinement is optional; the fixed-response estimate remains the primary report",
    }
    config_record = {
        "lambda_theta": cfg.lambda_theta,
        "lambda_w": cfg.lambda_w,
        "lambda_psi": cfg.lambda_psi,
        "lambda_sm": cfg.lambda_sm,
        "eps_w": cfg.eps_w,
        "eta_id": cfg.eta_id,
        "tau_rho_ref": tau_rho_ref,
        "refine_wind": cfg.refine_wind,
        "refine_dispersion": cfg.refine_dispersion,
        "correction_basis": cfg.correction_basis,
        "require_fit_improvement": cfg.require_fit_improvement,
        "max_outer_iters": cfg.max_outer_iters,
    }

    return RefineResult(
        accepted=accepted,
        reason=reason,
        wind_correction=(None if not cfg.refine_wind else [[float(v) for v in row] for row in A_ref.tolist()]),
        wind_correction_field=(None if not cfg.refine_wind else [[float(v) for v in row] for row in delta_ref.tolist()]),
        refined_psi=[float(v) for v in psi_ref.tolist()],
        baseline_psi=[float(v) for v in psi0.tolist()],
        refined_fit=refined_fit,
        fixed_response_fit=fixed_response_fit,
        sigma_J_baseline_eff=sigma_J0,
        sigma_J_refined_eff=sigma_Jr,
        sigma_J_threshold=threshold,
        rho_baseline=rho0,
        rho_refined=rhor,
        tau_rho_ref=float(tau_rho_ref),
        rank_baseline=int(baseline_diag.numerical_rank),
        rank_refined=int(refined_diag.numerical_rank),
        objective_start=objective_start,
        objective_end=objective_end,
        constraint_satisfaction=constraint_satisfaction,
        optimizer_trace=search["trace"],
        warnings=warnings,
        config=config_record,
        metadata=metadata,
    )


__all__ = [
    "RefineConfig",
    "RefineResult",
    "refine_end_to_end",
]
