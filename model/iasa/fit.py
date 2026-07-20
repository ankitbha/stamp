"""IASA nonnegative fitting via projected FISTA (Task 8).

Projected FISTA is the sole solver path for the projected inverse problem

    c_hat = argmin_{c >= 0} ||Y_tilde - H_tilde c||^2 + lambda ||c - c0||^2.

Everything runs in PyTorch on the input device with the inverse dtype
(float64); NumPy appears only at the JSON/serialization boundary. There is no
SciPy dependency and no CPU transfer for fitting. The fit uses the predeclared
fixed-zero mask (shared with Task 7 diagnostics), restores exact zeros in the
full coefficient vector, and never removes a column after inspecting fitted
coefficients.

Symbols follow paper/4.method.tex and paper/6.algorithm.tex: the NNLS estimator
(eq. nnls_estimator) with ridge/prior regularizer, the Lipschitz FISTA step from
sigma_1 with monotone restart, the active-set covariance (eq. active_covariance),
and the refitted parametric-bootstrap residual model-adequacy statistic
(eqs. adequacy_coordinates / residual_adequacy_statistic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from model.iasa.backend import (
    dtype_name,
    resolve_device,
    resolve_dtype,
    runtime_provenance,
    to_numpy,
    validate_ensemble_kind,
)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FitConfig:
    lambda_reg: float = 0.0
    prior_mean: Any = None  # c0; None -> zeros (ridge). Array-like over full columns.
    fixed_zero_indices: tuple[int, ...] = ()
    max_iters: int = 10000
    tol_kkt: float = 1e-6
    tol_rel_obj: float = 1e-12
    active_tolerance: float = 1e-8
    tau_sigma: float | None = None
    device: str | None = None
    dtype: str | None = "float64"
    ensemble_kind: str = "transport"
    lipschitz_sigma1: float | None = None
    use_power_iteration: bool = False
    power_iters: int = 100
    condition_warning_ratio: float = 1e8
    seed: int = 0


@dataclass(frozen=True)
class NoiseModel:
    """Externally declared observation-noise model for the adequacy check.

    ``covariance`` is either a scalar variance ``sigma_e^2`` or a full
    ``[N, N]`` covariance in the observed-row space. The check treats the model
    as calibrated only when it was calibrated independently of this fitted
    residual (``calibrated`` and ``source`` set, ``estimated_from_fit_residual``
    false) and the reduced covariance is positive definite.
    """

    covariance: Any = None
    calibrated: bool = False
    source: str | None = None
    estimated_from_fit_residual: bool = False


@dataclass(frozen=True)
class AdequacyConfig:
    alpha: float = 0.05
    n_replicates: int = 1000
    seed: int = 0


# --------------------------------------------------------------------------- #
# Results                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    c_hat: torch.Tensor
    c_reduced: torch.Tensor
    reduced_to_original: list[int]
    original_to_reduced: dict[int, int]
    theta: torch.Tensor | None
    source_names: list[str]
    source_contribution_summaries: dict[str, Any]
    fitted_sensor_vector: torch.Tensor
    residual_vector: torch.Tensor
    residual_norm: float
    zero_model_residual_norm: float
    source_basis_metadata: list[dict[str, Any]]
    active_indices: list[int]
    active_covariance: torch.Tensor | None
    coefficient_std: dict[int, float] | None
    objective_summary: dict[str, float]
    convergence_status: str
    iteration_count: int
    restart_count: int
    kkt_residual: float
    solver: dict[str, Any]
    ensemble_kind: str
    warnings: list[str]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "c_hat": _to_list(self.c_hat),
            "c_reduced": _to_list(self.c_reduced),
            "reduced_to_original": list(self.reduced_to_original),
            "original_to_reduced": {str(k): v for k, v in self.original_to_reduced.items()},
            "theta": (None if self.theta is None else [list(map(float, row)) for row in to_numpy(self.theta).tolist()]),
            "source_names": list(self.source_names),
            "source_contribution_summaries": self.source_contribution_summaries,
            "fitted_sensor_vector": _to_list(self.fitted_sensor_vector),
            "residual_vector": _to_list(self.residual_vector),
            "residual_norm": self.residual_norm,
            "zero_model_residual_norm": self.zero_model_residual_norm,
            "source_basis_metadata": self.source_basis_metadata,
            "active_indices": list(self.active_indices),
            "active_covariance": (None if self.active_covariance is None else [list(map(float, row)) for row in to_numpy(self.active_covariance).tolist()]),
            "coefficient_std": (None if self.coefficient_std is None else {str(k): v for k, v in self.coefficient_std.items()}),
            "objective_summary": self.objective_summary,
            "convergence_status": self.convergence_status,
            "iteration_count": self.iteration_count,
            "restart_count": self.restart_count,
            "kkt_residual": self.kkt_residual,
            "solver": self.solver,
            "ensemble_kind": self.ensemble_kind,
            "warnings": list(self.warnings),
            "config": self.config,
            "metadata": self.metadata,
        }


@dataclass
class AdequacyResult:
    calibration_status: str
    T_res: float | None
    bootstrap_quantile: float | None
    p_value: float | None
    alpha: float
    n_replicates: int | None
    inadequate: bool | None
    raw_residual_norm: float
    projected_residual_norm: float
    sensorwise_summary: dict[str, Any]
    timewise_summary: dict[str, Any]
    autocorrelation_summary: dict[str, Any]
    noise_model_provenance: dict[str, Any]
    warnings: list[str]
    metadata: dict[str, Any]

    def to_json_summary(self) -> dict[str, Any]:
        return {
            "calibration_status": self.calibration_status,
            "T_res": self.T_res,
            "bootstrap_quantile": self.bootstrap_quantile,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "n_replicates": self.n_replicates,
            "inadequate": self.inadequate,
            "raw_residual_norm": self.raw_residual_norm,
            "projected_residual_norm": self.projected_residual_norm,
            "sensorwise_summary": self.sensorwise_summary,
            "timewise_summary": self.timewise_summary,
            "autocorrelation_summary": self.autocorrelation_summary,
            "noise_model_provenance": self.noise_model_provenance,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def _to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in to_numpy(tensor).reshape(-1).tolist()]


# --------------------------------------------------------------------------- #
# Mask handling (shared contract with Task 7)                                 #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Projected FISTA (the sole solver path); batched over RHS columns            #
# --------------------------------------------------------------------------- #
def _largest_singular_value(H: torch.Tensor, *, use_power_iteration: bool, power_iters: int, seed: int) -> tuple[float, str]:
    if min(H.shape) == 0:
        return 0.0, "empty"
    if use_power_iteration:
        gen = torch.Generator(device=H.device)
        gen.manual_seed(int(seed))
        v = torch.randn(H.shape[1], dtype=H.dtype, device=H.device, generator=gen)
        v = v / max(float(torch.linalg.vector_norm(v)), 1e-30)
        sigma = 0.0
        for _ in range(max(1, power_iters)):
            w = H @ v
            v = H.transpose(0, 1) @ w
            nv = float(torch.linalg.vector_norm(v))
            if nv <= 0:
                break
            v = v / nv
            sigma = math.sqrt(nv)
        return sigma, "power_iteration"
    return float(torch.linalg.svdvals(H)[0]), "svd"


def _projected_fista(
    H: torch.Tensor,
    Y: torch.Tensor,
    prior: torch.Tensor,
    lam: float,
    *,
    sigma1: float,
    max_iters: int,
    tol_kkt: float,
    tol_rel_obj: float,
    c_init: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Minimize ||H c - Y||^2 + lam ||c - prior||^2 over c >= 0.

    ``Y`` is ``[N]`` or ``[N, B]``; solves all ``B`` right-hand sides jointly
    with per-column monotone restart. Returns the solution and solver metadata.
    """

    batched = Y.ndim == 2
    Ym = Y if batched else Y.reshape(-1, 1)
    device, dtype = H.device, H.dtype
    N, J = H.shape
    B = Ym.shape[1]
    prior_m = prior.reshape(J, 1).to(device=device, dtype=dtype)

    L = 2.0 * (sigma1 * sigma1 + lam)
    step = (1.0 / L) if L > 0 else 0.0

    def objective(cv: torch.Tensor) -> torch.Tensor:
        r = H @ cv - Ym
        val = (r * r).sum(dim=0)
        if lam > 0:
            d = cv - prior_m
            val = val + lam * (d * d).sum(dim=0)
        return val

    def gradient(cv: torch.Tensor) -> torch.Tensor:
        g = 2.0 * (H.transpose(0, 1) @ (H @ cv - Ym))
        if lam > 0:
            g = g + 2.0 * lam * (cv - prior_m)
        return g

    c = torch.zeros((J, B), dtype=dtype, device=device) if c_init is None else c_init.reshape(J, B).clone()
    c = torch.clamp(c, min=0.0)
    y = c.clone()
    t = torch.ones(B, dtype=dtype, device=device)
    f_c = objective(c)
    restart_count = 0
    iteration = 0
    kkt_norm = torch.full((B,), math.inf, dtype=dtype, device=device)

    if J == 0 or step == 0.0:
        # Degenerate problem: no free coefficients or zero operator.
        g = gradient(c)
        kkt_vec = torch.where(c > 0, g, torch.clamp(g, max=0.0))
        kkt_norm = torch.linalg.vector_norm(kkt_vec, dim=0)
        return {
            "c": c if batched else c.reshape(-1),
            "iterations": 0,
            "restart_count": 0,
            "kkt_residual": float(kkt_norm.max()) if B else 0.0,
            "converged": True,
            "objective_initial": [float(v) for v in objective(torch.zeros_like(c)).tolist()],
            "objective_final": [float(v) for v in f_c.tolist()],
            "lipschitz_L": L,
            "step": step,
        }

    f_initial = objective(torch.zeros_like(c))
    for iteration in range(1, max_iters + 1):
        g = gradient(y)
        c_new = torch.clamp(y - step * g, min=0.0)
        f_new = objective(c_new)

        bad = f_new > f_c
        if bool(bad.any()):
            g_c = gradient(c)
            c_plain = torch.clamp(c - step * g_c, min=0.0)
            f_plain = objective(c_plain)
            c_new = torch.where(bad[None, :], c_plain, c_new)
            f_new = torch.where(bad, f_plain, f_new)
            t = torch.where(bad, torch.ones_like(t), t)
            restart_count += int(bad.sum())

        t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_new
        y = c_new + momentum[None, :] * (c_new - c)

        rel_obj = torch.abs(f_new - f_c) / torch.clamp(torch.abs(f_c), min=1e-30)
        c = c_new
        t = t_new

        g_c = gradient(c)
        kkt_vec = torch.where(c > 0, g_c, torch.clamp(g_c, max=0.0))
        kkt_norm = torch.linalg.vector_norm(kkt_vec, dim=0)
        f_c = f_new

        if bool((kkt_norm <= tol_kkt).all()) and bool((rel_obj <= tol_rel_obj).all()):
            break

    converged = bool((kkt_norm <= tol_kkt).all())
    return {
        "c": c if batched else c.reshape(-1),
        "iterations": int(iteration),
        "restart_count": int(restart_count),
        "kkt_residual": float(kkt_norm.max()) if B else 0.0,
        "converged": converged,
        "objective_initial": [float(v) for v in f_initial.tolist()],
        "objective_final": [float(v) for v in f_c.tolist()],
        "lipschitz_L": L,
        "step": step,
    }


# --------------------------------------------------------------------------- #
# Activity reconstruction and contribution summaries                          #
# --------------------------------------------------------------------------- #
def _source_order(column_index: Sequence[dict[str, Any]]) -> tuple[list[int], dict[int, str]]:
    order: list[int] = []
    names: dict[int, str] = {}
    for col in column_index:
        k = int(col["source_index"])
        if k not in names:
            names[k] = str(col.get("source_name"))
            order.append(k)
    return sorted(order), names


def _reconstruct_theta(
    c_hat: torch.Tensor,
    column_index: Sequence[dict[str, Any]],
    temporal_basis: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, list[str]]:
    order, names = _source_order(column_index)
    source_names = [names[k] for k in order]
    if temporal_basis is None:
        return None, source_names
    phi = temporal_basis.to(device=device, dtype=dtype)
    if phi.ndim != 2:
        raise ValueError("temporal_basis must have shape [T, B]")
    B = phi.shape[1]
    K = len(order)
    index_of = {k: i for i, k in enumerate(order)}
    C = torch.zeros((K, B), dtype=dtype, device=device)
    for col, value in zip(column_index, c_hat.reshape(-1)):
        b = int(col["basis_index"])
        if b < 0 or b >= B:
            raise ValueError("temporal_basis columns do not cover basis_index range")
        C[index_of[int(col["source_index"])], b] = value
    theta = phi @ C.transpose(0, 1)  # [T, K]
    return theta, source_names


def _contribution_summaries(
    c_hat: torch.Tensor,
    column_index: Sequence[dict[str, Any]],
    theta: torch.Tensor | None,
    timestamps: Any,
    source_names: list[str],
) -> dict[str, Any]:
    order, _ = _source_order(column_index)
    summaries: dict[str, Any] = {}

    coeff_by_source: dict[str, dict[str, float]] = {name: {} for name in source_names}
    for col, value in zip(column_index, c_hat.reshape(-1)):
        coeff_by_source[str(col.get("source_name"))][str(col.get("basis_name"))] = float(value)
    summaries["coefficient_by_source_basis"] = coeff_by_source

    if theta is not None:
        theta_np = to_numpy(theta)
        summaries["total_contribution"] = {
            name: float(theta_np[:, i].sum()) for i, name in enumerate(source_names)
        }
        peak = theta_np.max(axis=0)
        active_threshold = 0.5 * peak
        summaries["active_period_fraction"] = {
            name: float((theta_np[:, i] > active_threshold[i]).mean()) if peak[i] > 0 else 0.0
            for i, name in enumerate(source_names)
        }
        if timestamps is not None and len(timestamps) == theta_np.shape[0]:
            hours = _hours(timestamps)
            days = _day_index(timestamps)
            if hours is not None:
                diurnal: dict[str, list[float]] = {}
                for i, name in enumerate(source_names):
                    profile = [0.0] * 24
                    counts = [0] * 24
                    for t_idx, h in enumerate(hours):
                        profile[h] += float(theta_np[t_idx, i])
                        counts[h] += 1
                    diurnal[name] = [profile[h] / counts[h] if counts[h] else 0.0 for h in range(24)]
                summaries["diurnal_hourly_mean"] = diurnal
            if days is not None:
                unique_days = sorted(set(days))
                daily: dict[str, dict[str, float]] = {}
                for i, name in enumerate(source_names):
                    per_day = {str(d): 0.0 for d in unique_days}
                    for t_idx, d in enumerate(days):
                        per_day[str(d)] += float(theta_np[t_idx, i])
                    daily[name] = per_day
                summaries["daily_totals"] = daily
    else:
        summaries["total_contribution"] = {
            name: float(sum(coeff_by_source[name].values())) for name in source_names
        }
    return summaries


def _hours(timestamps: Any) -> list[int] | None:
    try:
        return [int(ts.hour) for ts in timestamps]
    except (AttributeError, TypeError):
        return None


def _day_index(timestamps: Any) -> list[Any] | None:
    out: list[Any] = []
    for ts in timestamps:
        if hasattr(ts, "date"):
            out.append(ts.date().isoformat() if callable(ts.date) else str(ts.date))
        else:
            return None
    return out


def projected_data_objective(
    H_tilde: torch.Tensor,
    Y_tilde: torch.Tensor,
    c_hat: torch.Tensor,
    *,
    lam: float = 0.0,
    prior: torch.Tensor | None = None,
) -> dict[str, float]:
    """Projected objective ``||Y_tilde - H_tilde c||^2 + lam ||c - prior||^2``.

    The shared scoring function used by the end-to-end refinement (Task 9C) so a
    candidate response is scored with exactly the projected FISTA objective. The
    data term equals ``min_gamma ||Y - H_lag c - Q gamma||^2`` by the background
    projection identity. Returns the data term, the regularizer, and their sum.
    """

    device, dtype = H_tilde.device, H_tilde.dtype
    c = c_hat.to(device=device, dtype=dtype).reshape(-1)
    r = Y_tilde.to(device=device, dtype=dtype).reshape(-1) - H_tilde @ c
    data = float((r * r).sum())
    reg = 0.0
    if lam > 0:
        p = torch.zeros_like(c) if prior is None else prior.to(device=device, dtype=dtype).reshape(-1)
        d = c - p
        reg = float(lam) * float((d * d).sum())
    return {"data": data, "regularizer": reg, "total": data + reg}


def summarize_report_groups(result: FitResult, groups: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Sum fitted activity trajectories and contributions over merge groups.

    ``groups`` is a sequence of source-index groups (e.g. deterministic
    connected components from Task 9). The fine-resolution fit is unchanged; the
    group products are pure sums of member sources, never a refit.
    """

    order, names = _source_order(result.source_basis_metadata)
    index_of = {k: i for i, k in enumerate(order)}
    out: dict[str, Any] = {"groups": [], "is_sum_of_members": True}
    theta_np = None if result.theta is None else to_numpy(result.theta)
    totals = result.source_contribution_summaries.get("total_contribution", {})
    for members in groups:
        member_list = sorted(int(m) for m in members)
        for m in member_list:
            if m not in index_of:
                raise ValueError(f"group member source_index {m} is not a fitted source")
        member_names = [names[m] for m in member_list]
        group_total = float(sum(totals.get(names[m], 0.0) for m in member_list))
        entry: dict[str, Any] = {
            "members": member_list,
            "member_names": member_names,
            "total_contribution": group_total,
        }
        if theta_np is not None:
            trajectory = theta_np[:, [index_of[m] for m in member_list]].sum(axis=1)
            entry["activity_trajectory"] = [float(v) for v in trajectory.tolist()]
        out["groups"].append(entry)
    return out


# --------------------------------------------------------------------------- #
# Active-set covariance                                                        #
# --------------------------------------------------------------------------- #
def _active_set_covariance(
    H: torch.Tensor,
    residual: torch.Tensor,
    c_reduced: torch.Tensor,
    lam: float,
    reduced_to_original: list[int],
    *,
    active_tolerance: float,
    tau_sigma: float | None,
    singular_values: torch.Tensor,
) -> tuple[list[int], torch.Tensor | None, dict[int, float] | None]:
    device, dtype = H.device, H.dtype
    N, J = H.shape
    eps = float(torch.finfo(dtype).eps)
    sigma_1 = float(singular_values[0]) if singular_values.numel() else 0.0
    tau_num = max(N, J) * eps * sigma_1
    tau = tau_num if tau_sigma is None else float(tau_sigma)
    r_eff = int(torch.count_nonzero(singular_values > tau)) if singular_values.numel() else 0
    dof = max(N - r_eff, 1)
    sigma_hat_sq = float((residual * residual).sum()) / dof

    active_mask = c_reduced.reshape(-1) > active_tolerance
    active_reduced = [j for j in range(J) if bool(active_mask[j])]
    active_indices = [reduced_to_original[j] for j in active_reduced]
    if not active_reduced:
        return active_indices, None, {}

    idx = torch.tensor(active_reduced, dtype=torch.long, device=device)
    H_a = H.index_select(1, idx)
    gram = H_a.transpose(0, 1) @ H_a
    if lam > 0:
        gram = gram + lam * torch.eye(gram.shape[0], dtype=dtype, device=device)
    covariance = sigma_hat_sq * torch.linalg.pinv(gram)
    std_diag = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))
    coefficient_std = {active_indices[i]: float(std_diag[i]) for i in range(len(active_indices))}
    return active_indices, covariance, coefficient_std


# --------------------------------------------------------------------------- #
# Main fit entry points                                                        #
# --------------------------------------------------------------------------- #
def fit_sources(
    H_tilde: torch.Tensor,
    Y_tilde: torch.Tensor,
    column_index: Sequence[dict[str, Any]],
    *,
    config: FitConfig | None = None,
    temporal_basis: torch.Tensor | None = None,
    timestamps: Any = None,
    H_lag: torch.Tensor | None = None,
    U_r: torch.Tensor | None = None,
    Y: torch.Tensor | None = None,
    diagnostics: Any = None,
) -> FitResult:
    cfg = config or FitConfig()
    validate_ensemble_kind(cfg.ensemble_kind)
    if not isinstance(H_tilde, torch.Tensor) or H_tilde.ndim != 2:
        raise ValueError("H_tilde must be a 2-D torch.Tensor [N, J]")
    device = resolve_device(cfg.device) if cfg.device is not None else H_tilde.device
    dtype = resolve_dtype(cfg.dtype, default=torch.float64) if cfg.dtype is not None else H_tilde.dtype
    H_full = H_tilde.to(device=device, dtype=dtype)
    y = Y_tilde.to(device=device, dtype=dtype).reshape(-1)
    if not torch.isfinite(H_full).all() or not torch.isfinite(y).all():
        raise ValueError("H_tilde and Y_tilde must contain only finite values")
    N, J_full = H_full.shape
    if y.shape[0] != N:
        raise ValueError("Y_tilde length must match H_tilde rows")
    if len(column_index) != J_full:
        raise ValueError("column_index length must match H_tilde columns")
    if cfg.lambda_reg < 0:
        raise ValueError("lambda_reg must be nonnegative")

    kept = _validate_fixed_zero(cfg.fixed_zero_indices, J_full)
    reduced_to_original = list(kept)
    original_to_reduced = {orig: reduced for reduced, orig in enumerate(kept)}
    for orig in range(J_full):
        original_to_reduced.setdefault(orig, -1)

    if diagnostics is not None:
        diag_mask = getattr(diagnostics, "reduced_to_original", None)
        if diag_mask is None or list(diag_mask) != reduced_to_original:
            raise ValueError("fit/diagnostic fixed-zero mask mismatch")

    index = torch.tensor(kept, dtype=torch.long, device=device)
    H = H_full.index_select(1, index) if kept else H_full[:, :0]
    J = H.shape[1]

    # Prior c0 (default zeros -> ridge).
    if cfg.prior_mean is None:
        prior_full = torch.zeros(J_full, dtype=dtype, device=device)
    else:
        prior_full = torch.as_tensor(cfg.prior_mean, dtype=dtype, device=device).reshape(-1)
        if prior_full.shape[0] != J_full:
            raise ValueError("prior_mean length must match H_tilde columns")
    prior = prior_full.index_select(0, index) if kept else prior_full[:0]

    sigma1, sigma1_method = (
        (float(cfg.lipschitz_sigma1), "override")
        if cfg.lipschitz_sigma1 is not None
        else _largest_singular_value(
            H, use_power_iteration=cfg.use_power_iteration, power_iters=cfg.power_iters, seed=cfg.seed
        )
    )

    solve = _projected_fista(
        H, y, prior, float(cfg.lambda_reg),
        sigma1=sigma1, max_iters=cfg.max_iters, tol_kkt=cfg.tol_kkt, tol_rel_obj=cfg.tol_rel_obj,
    )
    c_reduced = solve["c"].reshape(-1)

    c_hat = torch.zeros(J_full, dtype=dtype, device=device)
    if kept:
        c_hat.index_copy_(0, index, c_reduced)

    fitted_sensor_vector = H @ c_reduced if J else torch.zeros(N, dtype=dtype, device=device)
    residual_vector = y - fitted_sensor_vector
    residual_norm = float(torch.linalg.vector_norm(residual_vector))
    zero_model_residual_norm = float(torch.linalg.vector_norm(y))

    singular_values = torch.linalg.svdvals(H) if J else torch.zeros(0, dtype=dtype, device=device)
    active_indices, active_covariance, coefficient_std = _active_set_covariance(
        H, residual_vector, c_reduced, float(cfg.lambda_reg), reduced_to_original,
        active_tolerance=cfg.active_tolerance, tau_sigma=cfg.tau_sigma, singular_values=singular_values,
    )

    theta, source_names = _reconstruct_theta(c_hat, column_index, temporal_basis, device, dtype)
    summaries = _contribution_summaries(c_hat, column_index, theta, timestamps, source_names)

    warnings: list[str] = []
    numerical_rank = int(torch.count_nonzero(singular_values > (max(N, J) * float(torch.finfo(dtype).eps) * (float(singular_values[0]) if singular_values.numel() else 0.0)))) if J else 0
    kappa = None
    if J and numerical_rank == J and float(singular_values[-1]) > 0:
        kappa = float(singular_values[0]) / float(singular_values[-1])
    if J and numerical_rank < J:
        warnings.append("ill_conditioned: H_tilde is numerically rank deficient; activity split is non-unique")
    elif kappa is not None and kappa > cfg.condition_warning_ratio:
        warnings.append(f"ill_conditioned: condition number {kappa:.3e} exceeds {cfg.condition_warning_ratio:.1e}")
    if not solve["converged"]:
        warnings.append(f"not_converged: KKT residual {solve['kkt_residual']:.3e} above tol {cfg.tol_kkt:.1e} after {solve['iterations']} iters")

    objective_summary = {
        "initial": float(solve["objective_initial"][0]),
        "final": float(solve["objective_final"][0]),
        "zero_model": float(zero_model_residual_norm ** 2),
    }
    solver = {
        "method": "projected_fista",
        "sigma_1": sigma1,
        "sigma_1_method": sigma1_method,
        "lipschitz_L": solve["lipschitz_L"],
        "step": solve["step"],
        "numerical_rank": numerical_rank,
        "condition_number": kappa,
    }
    config_record = {
        "lambda_reg": cfg.lambda_reg,
        "fixed_zero_indices": list(cfg.fixed_zero_indices),
        "max_iters": cfg.max_iters,
        "tol_kkt": cfg.tol_kkt,
        "tol_rel_obj": cfg.tol_rel_obj,
        "active_tolerance": cfg.active_tolerance,
        "tau_sigma": cfg.tau_sigma,
        "ensemble_kind": cfg.ensemble_kind,
        "prior_mean_provided": cfg.prior_mean is not None,
    }
    metadata = {
        **runtime_provenance(device, dtype),
        "N": int(N),
        "J": int(J),
        "J_full": int(J_full),
        "response_dtype": dtype_name(dtype),
    }
    return FitResult(
        c_hat=c_hat,
        c_reduced=c_reduced,
        reduced_to_original=reduced_to_original,
        original_to_reduced=original_to_reduced,
        theta=theta,
        source_names=source_names,
        source_contribution_summaries=summaries,
        fitted_sensor_vector=fitted_sensor_vector,
        residual_vector=residual_vector,
        residual_norm=residual_norm,
        zero_model_residual_norm=zero_model_residual_norm,
        source_basis_metadata=[dict(col) for col in column_index],
        active_indices=active_indices,
        active_covariance=active_covariance,
        coefficient_std=coefficient_std,
        objective_summary=objective_summary,
        convergence_status="converged" if solve["converged"] else "max_iters",
        iteration_count=int(solve["iterations"]),
        restart_count=int(solve["restart_count"]),
        kkt_residual=float(solve["kkt_residual"]),
        solver=solver,
        ensemble_kind=cfg.ensemble_kind,
        warnings=warnings,
        config=config_record,
        metadata=metadata,
    )


def fit_projection(
    projection_result: Any,
    *,
    config: FitConfig | None = None,
    temporal_basis: torch.Tensor | None = None,
    timestamps: Any = None,
    diagnostics: Any = None,
) -> FitResult:
    H_tilde = projection_result.H_tilde
    Y_tilde = projection_result.Y_tilde
    H_removed = getattr(projection_result, "H_removed", None)
    Y_removed = getattr(projection_result, "Y_removed", None)
    H_lag = None if H_removed is None else H_tilde + H_removed
    Y = None if Y_removed is None else Y_tilde + Y_removed
    U_r = getattr(projection_result, "U_r", None)
    return fit_sources(
        H_tilde, Y_tilde, projection_result.column_index,
        config=config, temporal_basis=temporal_basis, timestamps=timestamps,
        H_lag=H_lag, U_r=U_r, Y=Y, diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Residual model-adequacy check (refitted parametric bootstrap)               #
# --------------------------------------------------------------------------- #
def _complement_basis(U_r: torch.Tensor, N: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if U_r is None or U_r.shape[1] == 0:
        return torch.eye(N, dtype=dtype, device=device)
    P_perp = torch.eye(N, dtype=dtype, device=device) - U_r @ U_r.transpose(0, 1)
    U, S, _ = torch.linalg.svd(P_perp)
    keep = int(round(float(S.sum())))  # rank of an idempotent projector = trace
    keep = max(0, min(keep, N))
    return U[:, :keep]


def _residual_summaries(
    raw_residual: torch.Tensor, row_index: Sequence[dict[str, Any]] | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r = to_numpy(raw_residual).reshape(-1)
    sensorwise: dict[str, Any] = {}
    timewise: dict[str, Any] = {}
    if row_index is not None and len(row_index) == r.shape[0]:
        by_sensor: dict[str, list[float]] = {}
        by_time: dict[str, list[float]] = {}
        for value, row in zip(r.tolist(), row_index):
            by_sensor.setdefault(str(row.get("sensor_id", row.get("sensor_index"))), []).append(float(value))
            by_time.setdefault(str(row.get("time_index")), []).append(float(value))
        sensorwise = {k: {"rms": float((sum(v * v for v in vals) / len(vals)) ** 0.5), "count": len(vals)} for k, vals in by_sensor.items()}
        timewise = {k: {"rms": float((sum(v * v for v in vals) / len(vals)) ** 0.5), "count": len(vals)} for k, vals in by_time.items()}
    lag1 = 0.0
    if r.shape[0] > 1:
        num = float((r[:-1] * r[1:]).sum())
        den = float((r * r).sum())
        lag1 = num / den if den > 0 else 0.0
    autocorrelation = {"lag1": lag1, "n": int(r.shape[0])}
    return sensorwise, timewise, autocorrelation


def residual_adequacy_check(
    fit_result: FitResult,
    projection_result: Any,
    noise_model: NoiseModel | None = None,
    *,
    config: AdequacyConfig | None = None,
) -> AdequacyResult:
    cfg = config or AdequacyConfig()
    H_tilde = projection_result.H_tilde
    Y_tilde = projection_result.Y_tilde
    device, dtype = H_tilde.device, H_tilde.dtype
    N = H_tilde.shape[0]
    U_r = getattr(projection_result, "U_r", None)
    H_removed = getattr(projection_result, "H_removed", None)
    Y_removed = getattr(projection_result, "Y_removed", None)
    row_index = getattr(projection_result, "row_index", None)

    H_lag = H_tilde if H_removed is None else H_tilde + H_removed
    Y = Y_tilde if Y_removed is None else Y_tilde + Y_removed
    c_full = fit_result.c_hat.to(device=device, dtype=dtype)

    raw_residual = Y - H_lag @ c_full
    projected_residual = fit_result.residual_vector.to(device=device, dtype=dtype)
    raw_norm = float(torch.linalg.vector_norm(raw_residual))
    projected_norm = float(torch.linalg.vector_norm(projected_residual))
    sensorwise, timewise, autocorrelation = _residual_summaries(raw_residual, row_index)

    provenance = {
        "provided": noise_model is not None and noise_model.covariance is not None,
        "calibrated_flag": bool(noise_model.calibrated) if noise_model is not None else False,
        "source": None if noise_model is None else noise_model.source,
        "estimated_from_fit_residual": bool(noise_model.estimated_from_fit_residual) if noise_model is not None else False,
    }
    warnings: list[str] = []

    def uncalibrated(reason: str) -> AdequacyResult:
        warnings.append(f"uncalibrated: {reason}")
        return AdequacyResult(
            calibration_status="uncalibrated",
            T_res=None, bootstrap_quantile=None, p_value=None,
            alpha=cfg.alpha, n_replicates=None, inadequate=None,
            raw_residual_norm=raw_norm, projected_residual_norm=projected_norm,
            sensorwise_summary=sensorwise, timewise_summary=timewise,
            autocorrelation_summary=autocorrelation, noise_model_provenance=provenance,
            warnings=warnings, metadata={**runtime_provenance(device, dtype), "N": int(N)},
        )

    if noise_model is None or noise_model.covariance is None:
        return uncalibrated("no external observation-noise model supplied")
    if not noise_model.calibrated or noise_model.source is None:
        return uncalibrated("noise model is not externally calibrated")
    if noise_model.estimated_from_fit_residual:
        return uncalibrated("noise model was estimated from the same fitted residual")

    # Build Sigma_e in observed-row space.
    cov = noise_model.covariance
    if isinstance(cov, (int, float)):
        if float(cov) <= 0:
            return uncalibrated("scalar noise variance must be positive")
        Sigma_e = float(cov) * torch.eye(N, dtype=dtype, device=device)
    else:
        Sigma_e = torch.as_tensor(cov, dtype=dtype, device=device)
        if Sigma_e.shape != (N, N):
            return uncalibrated("covariance shape does not match observed-row count")

    Z = _complement_basis(U_r, N, dtype, device)  # [N, N-r]
    if Z.shape[1] == 0:
        return uncalibrated("background occupies the full observed-row space; no residual degrees of freedom")
    bar_Sigma = Z.transpose(0, 1) @ Sigma_e @ Z
    try:
        chol = torch.linalg.cholesky(bar_Sigma)
    except RuntimeError:
        return uncalibrated("reduced noise covariance is not positive definite")

    def quad(bar_r: torch.Tensor) -> torch.Tensor:
        # bar_r: [m] or [m, B]; returns scalar or [B] of bar_r^T bar_Sigma^{-1} bar_r.
        rhs = bar_r.reshape(bar_r.shape[0], -1)
        sol = torch.cholesky_solve(rhs, chol)
        return (rhs * sol).sum(dim=0)

    bar_r_obs = Z.transpose(0, 1) @ raw_residual
    T_res = float(quad(bar_r_obs)[0])

    # Fitted complete model mean in full observed space (source + best-fit
    # background). With no background the background component is zero, not the
    # whole residual -- otherwise the mean collapses to Y and the bootstrap null
    # would absorb any omitted signal.
    if U_r is None or U_r.shape[1] == 0:
        background_component = torch.zeros_like(raw_residual)
    else:
        background_component = U_r @ (U_r.transpose(0, 1) @ raw_residual)
    mean_full = H_lag @ c_full + background_component

    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))
    B = int(cfg.n_replicates)
    noise = torch.randn(N, B, dtype=dtype, device=device, generator=gen)
    if isinstance(cov, (int, float)):
        noise = math.sqrt(float(cov)) * noise
    else:
        L_full = torch.linalg.cholesky(Sigma_e + 1e-12 * torch.eye(N, dtype=dtype, device=device))
        noise = L_full @ noise
    Y_b = mean_full[:, None] + noise  # [N, B]

    # Refit each replicate with the same projected FISTA (batched on device).
    if U_r is None or U_r.shape[1] == 0:
        Y_tilde_b = Y_b
    else:
        Y_tilde_b = Y_b - U_r @ (U_r.transpose(0, 1) @ Y_b)

    kept = fit_result.reduced_to_original
    index = torch.tensor(kept, dtype=torch.long, device=device)
    H_red = H_tilde.index_select(1, index) if kept else H_tilde[:, :0]
    lam = float(fit_result.config.get("lambda_reg", 0.0))
    prior = torch.zeros(H_red.shape[1], dtype=dtype, device=device)
    sigma1 = float(fit_result.solver.get("sigma_1", 0.0)) or (float(torch.linalg.svdvals(H_red)[0]) if min(H_red.shape) else 0.0)
    solve_b = _projected_fista(
        H_red, Y_tilde_b, prior, lam,
        sigma1=sigma1, max_iters=int(fit_result.config.get("max_iters", 10000)),
        tol_kkt=float(fit_result.config.get("tol_kkt", 1e-6)),
        tol_rel_obj=float(fit_result.config.get("tol_rel_obj", 1e-12)),
    )
    c_b = solve_b["c"].reshape(H_red.shape[1], B)
    c_b_full = torch.zeros(H_lag.shape[1], B, dtype=dtype, device=device)
    if kept:
        c_b_full.index_copy_(0, index, c_b)
    bar_r_b = Z.transpose(0, 1) @ (Y_b - H_lag @ c_b_full)  # [m, B]
    T_b = quad(bar_r_b)  # [B]

    quantile = float(torch.quantile(T_b, 1.0 - cfg.alpha))
    exceed = int(torch.count_nonzero(T_b >= T_res))
    p_value = (1.0 + exceed) / (B + 1.0)
    inadequate = bool(T_res > quantile)

    return AdequacyResult(
        calibration_status="calibrated",
        T_res=T_res, bootstrap_quantile=quantile, p_value=p_value,
        alpha=cfg.alpha, n_replicates=B, inadequate=inadequate,
        raw_residual_norm=raw_norm, projected_residual_norm=projected_norm,
        sensorwise_summary=sensorwise, timewise_summary=timewise,
        autocorrelation_summary=autocorrelation, noise_model_provenance=provenance,
        warnings=warnings,
        metadata={**runtime_provenance(device, dtype), "N": int(N), "residual_dof": int(Z.shape[1])},
    )


# --------------------------------------------------------------------------- #
# Ensemble aggregation (provenance-gated; kinds never pooled)                 #
# --------------------------------------------------------------------------- #
def _require_single_kind(results: Sequence[FitResult], kind: str) -> None:
    if not results:
        raise ValueError("ensemble aggregation requires at least one FitResult")
    kinds = {r.ensemble_kind for r in results}
    if kinds != {kind}:
        raise ValueError(f"ensemble aggregation for {kind!r} requires all members tagged {kind!r}; got {sorted(kinds)}")
    reference = results[0].reduced_to_original
    for r in results:
        if r.reduced_to_original != reference:
            raise ValueError("ensemble members must share the same reduced column mapping")


def aggregate_transport_ensemble(
    results: Sequence[FitResult], *, quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
) -> dict[str, Any]:
    _require_single_kind(results, "transport")
    device = results[0].c_hat.device
    stacked = torch.stack([r.c_hat.to(device=device, dtype=torch.float64) for r in results], dim=1)  # [J_full, B]
    q = torch.tensor(list(quantiles), dtype=torch.float64, device=device)
    cq = torch.quantile(stacked, q, dim=1)  # [len(q), J_full]
    coefficient_intervals = {
        f"q{int(round(quant * 100)):02d}": [float(v) for v in cq[i].tolist()]
        for i, quant in enumerate(quantiles)
    }
    totals = {}
    for r in results:
        for name, value in r.source_contribution_summaries.get("total_contribution", {}).items():
            totals.setdefault(name, []).append(float(value))
    source_total_intervals = {
        name: {
            f"q{int(round(quant * 100)):02d}": float(v)
            for quant, v in zip(quantiles, torch.quantile(torch.tensor(vals, dtype=torch.float64), q.cpu()).tolist())
        }
        for name, vals in totals.items()
    }
    return {
        "ensemble_kind": "transport",
        "product": "uncertainty_intervals",
        "n_members": len(results),
        "quantiles": list(quantiles),
        "coefficient_intervals": coefficient_intervals,
        "source_total_intervals": source_total_intervals,
    }


def aggregate_inventory_scenarios(results: Sequence[FitResult], *, scenario_names: Sequence[str] | None = None) -> dict[str, Any]:
    _require_single_kind(results, "inventory")
    names = list(scenario_names) if scenario_names is not None else [f"scenario_{i}" for i in range(len(results))]
    if len(names) != len(results):
        raise ValueError("scenario_names length must match the number of results")
    scenarios = []
    for name, r in zip(names, results):
        scenarios.append({
            "scenario": name,
            "total_contribution": r.source_contribution_summaries.get("total_contribution", {}),
            "c_hat": _to_list(r.c_hat),
        })
    return {
        "ensemble_kind": "inventory",
        "product": "robustness_scenarios",
        "n_scenarios": len(results),
        "scenarios": scenarios,
    }


__all__ = [
    "AdequacyConfig",
    "AdequacyResult",
    "FitConfig",
    "FitResult",
    "NoiseModel",
    "aggregate_inventory_scenarios",
    "aggregate_transport_ensemble",
    "fit_projection",
    "fit_sources",
    "projected_data_objective",
    "residual_adequacy_check",
    "summarize_report_groups",
]
