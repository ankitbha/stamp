#!/usr/bin/env python3
"""Synthetic study of identifiable groupings (paper Section 5.1-5.2).

Runs five parts of E1 and the baseline comparison E2 on planted operators and
writes one JSON file of summary statistics.

Usage:
    python experiments/iasa_synthetic/run_study.py --out evaluation/iasa_synthetic/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.iasa_synthetic import baselines as bl  # noqa: E402
from experiments.iasa_synthetic.generator import observe, planted_problem, project_out  # noqa: E402
from model.iasa.grouping import (  # noqa: E402
    GroupingConfig,
    largest_discarded,
    latent_grouping,
    _minimal_partitions_exhaustive,
    exact_components,
    group_report,
    numerical_tolerance,
    separation,
)

TAU_THETA = math.sqrt(1.0 - 0.99**2)


def _canon(P):
    return sorted((sorted(int(j) for j in g) for g in P), key=lambda g: g[0])


def _random_sizes(rng, J, kmax):
    sizes, left = [], J
    while left:
        k = int(rng.integers(1, min(kmax, left) + 1))
        sizes.append(k)
        left -= k
    return sizes


def _group_errors(A, chat, c, P):
    return [float(np.linalg.norm(A[:, g] @ (chat[g] - c[g]))) for g in P]


# --------------------------------------------------------------------------- #
# E1a: exact recovery of planted matroid components, and run time
# --------------------------------------------------------------------------- #
def e1a_exact_recovery(rng, Js=(10, 25, 50, 100), n_inst=30, N=300):
    rows = []
    for J in Js:
        hit = {"components": 0, "pairwise": 0, "bkw": 0}
        split_multi, n_multi, times = 0, 0, []
        for _ in range(n_inst):
            sizes = _random_sizes(rng, J, 5)
            prob = planted_problem(rng, N, sizes, mode="exact", slack=5, nuisance_dim=5)
            truth = _canon(prob.blocks)
            t0 = time.perf_counter()
            comps = exact_components(prob.A)
            times.append(time.perf_counter() - t0)
            pw = bl.pairwise_coherence_grouping(prob.A, 0.99)
            bk = bl.bkw_grouping(prob.A)
            hit["components"] += int(_canon(comps) == truth)
            hit["pairwise"] += int(_canon(pw) == truth)
            hit["bkw"] += int(_canon(bk) == truth)
            for g in prob.blocks:
                if len(g) >= 3:
                    n_multi += 1
                    split_multi += int(not any(set(g) <= set(h) for h in pw))
        rows.append({
            "J": J, "instances": n_inst,
            "recovery_rate": {k: v / n_inst for k, v in hit.items()},
            "pairwise_splits_blocks_of_size_ge3": split_multi / max(n_multi, 1),
            "n_blocks_size_ge3": n_multi,
            "components_seconds_mean": float(np.mean(times)),
        })
    return rows


# --------------------------------------------------------------------------- #
# E1b: coefficient error against group-signal error as a block becomes ill-conditioned
# --------------------------------------------------------------------------- #
def e1b_within_block_conditioning(rng, omegas=None, reps=2000, N=200, sigma=0.05):
    omegas = np.logspace(0, -4, 9) if omegas is None else omegas
    sizes = [3, 2, 1]
    rows = []
    base_seed = int(rng.integers(1 << 30))
    for omega in omegas:
        prob = planted_problem(np.random.default_rng(base_seed), N, sizes, mode="near",
                               omega=float(omega), slack=2, nuisance_dim=5, shuffle=False)
        A, c, P = prob.A, prob.c, prob.blocks
        sv = np.linalg.svd(A, compute_uv=False)
        seps = [separation(A, g) for g in P]
        ranks = [int(np.linalg.matrix_rank(A[:, g])) for g in P]
        coef_ls, coef_nn, grp_ls, grp_nn = [], [], [], []
        noise_rng = np.random.default_rng(base_seed + 1)
        for _ in range(reps):
            y = observe(noise_rng, prob, sigma)
            c_ls = np.linalg.lstsq(A, y, rcond=None)[0]
            c_nn = bl.fit_nnls(A, y)
            coef_ls.append(np.linalg.norm(c_ls - c))
            coef_nn.append(np.linalg.norm(c_nn - c))
            grp_ls.append(_group_errors(A, c_ls, c, P))
            grp_nn.append(_group_errors(A, c_nn, c, P))
        grp_ls, grp_nn = np.array(grp_ls), np.array(grp_nn)
        rows.append({
            "omega": float(omega), "sigma_J": float(sv[-1]), "sigma_1": float(sv[0]),
            "separations": seps, "group_ranks": ranks, "group_sizes": [len(g) for g in P],
            "coef_rmse_ls": float(np.sqrt(np.mean(np.square(coef_ls)))),
            "coef_rmse_nnls": float(np.sqrt(np.mean(np.square(coef_nn)))),
            "group_rmse_ls": np.sqrt(np.mean(grp_ls**2, axis=0)).tolist(),
            "group_rmse_nnls": np.sqrt(np.mean(grp_nn**2, axis=0)).tolist(),
            "group_rms_bound": [sigma * math.sqrt(r) / s for r, s in zip(ranks, seps)],
            "coef_rms_bound_ls": float(sigma * math.sqrt(len(c)) / sv[-1]),
        })
    return {"sigma": sigma, "N": N, "sizes": sizes, "reps": reps, "rows": rows}


# --------------------------------------------------------------------------- #
# E1c: phase diagram of group-signal recovery against the T2 boundary
# --------------------------------------------------------------------------- #
def _two_block_operator(N, theta, rng):
    """Two 2-column blocks whose spans meet at smallest principal angle theta."""
    E = np.linalg.qr(rng.standard_normal((N, 4)))[0]
    v1 = E[:, [0, 1]]
    v2 = np.stack([math.cos(theta) * E[:, 0] + math.sin(theta) * E[:, 2], E[:, 3]], axis=1)
    W1 = np.array([[1.0, 0.6], [0.0, 0.8]])
    W2 = np.array([[1.0, 0.6], [0.0, 0.8]])
    return np.concatenate([v1 @ W1, v2 @ W2], axis=1)


def e1c_phase_diagram(rng, n_theta=13, n_sigma=13, reps=200, N=100, eta=1.0, delta=0.05):
    thetas = np.logspace(-3, math.log10(math.pi / 2), n_theta)
    sigmas = np.logspace(-3, 0, n_sigma)
    rate = np.zeros((n_theta, n_sigma))
    for i, th in enumerate(thetas):
        A = _two_block_operator(N, float(th), rng)
        c = np.array([1.0, 0.7, 0.9, 1.2])
        U = np.linalg.qr(A)[0]
        for k, sg in enumerate(sigmas):
            ok = 0
            for _ in range(reps):
                y = A @ c + sg * rng.standard_normal(N)
                chat = np.linalg.lstsq(A, y, rcond=None)[0]
                ok += int(np.linalg.norm(A[:, :2] @ (chat[:2] - c[:2])) <= eta)
            rate[i, k] = ok / reps
    s = np.sin(thetas)
    # T2: ||err|| <= sigma (sqrt(2) + sqrt(2 log(1/delta))) / s with probability >= 1 - delta
    boundary_sigma = eta * s / (math.sqrt(2) + math.sqrt(2 * math.log(1 / delta)))
    empirical_sigma = []
    for i in range(n_theta):
        good = [sigmas[k] for k in range(n_sigma) if rate[i, k] >= 1 - delta]
        empirical_sigma.append(float(max(good)) if good else None)
    return {
        "thetas": thetas.tolist(), "separations": s.tolist(), "sigmas": sigmas.tolist(),
        "success_rate": rate.tolist(), "eta": eta, "delta": delta,
        "t2_boundary_sigma": boundary_sigma.tolist(),
        "largest_sigma_with_rate_ge_1_minus_delta": empirical_sigma,
        "boundary_violations": int(sum(
            1 for i in range(n_theta) for k in range(n_sigma)
            if sigmas[k] <= boundary_sigma[i] and rate[i, k] < 1 - delta - 3 * math.sqrt(delta * (1 - delta) / reps))),
    }


# --------------------------------------------------------------------------- #
# E1d: greedy against exhaustive at a noise-level threshold; non-uniqueness
# --------------------------------------------------------------------------- #
def _near_dependent_operator(rng, J, N=50, n_dep=2, spread=0.1):
    A = rng.standard_normal((N, J))
    for _ in range(n_dep):
        i, j, k = rng.choice(J, 3, replace=False)
        w = rng.standard_normal(2)
        A[:, k] = w[0] * A[:, i] + w[1] * A[:, j] + spread * rng.standard_normal(N)
    return A


def e1d_greedy_vs_exhaustive(rng, Js=(6, 8, 10), n_inst=60, tau=0.2):
    rows = []
    import torch
    for J in Js:
        in_minimal, size_ratio, nonunique, tie, times_ex, times_gr = 0, [], 0, 0, [], []
        for _ in range(n_inst):
            A = _near_dependent_operator(rng, J, n_dep=int(rng.integers(1, 4)))
            t = numerical_tolerance(A)
            t0 = time.perf_counter()
            minimal, _ = _minimal_partitions_exhaustive(
                torch.as_tensor(A, dtype=torch.float64), [[j] for j in range(J)], tau, t)
            times_ex.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            greedy = group_report(A, GroupingConfig(tau_theta=tau, max_exhaustive_atoms=0)).reported
            times_gr.append(time.perf_counter() - t0)
            minimal_c = [_canon(P) for P in minimal]
            in_minimal += int(_canon(greedy) in minimal_c)
            best = max(len(P) for P in minimal_c)
            size_ratio.append(len(greedy) / best)
            nonunique += int(len(minimal_c) > 1)
            tie += int(sum(1 for P in minimal_c if len(P) == best) > 1)
        rows.append({
            "J": J, "instances": n_inst, "tau_theta": tau,
            "greedy_is_minimal_rate": in_minimal / n_inst,
            "greedy_blocks_over_best_mean": float(np.mean(size_ratio)),
            "greedy_blocks_over_best_min": float(np.min(size_ratio)),
            "more_than_one_minimal_rate": nonunique / n_inst,
            "tie_in_block_count_rate": tie / n_inst,
            "exhaustive_seconds_mean": float(np.mean(times_ex)),
            "greedy_seconds_mean": float(np.mean(times_gr)),
        })
    return rows


# --------------------------------------------------------------------------- #
# E1e: separations under operator perturbation (T3)
# --------------------------------------------------------------------------- #
def _theta_bound(A, D, g):
    """T3 bound on the change of the smallest principal angle of block g, or None if it does not apply."""
    rest = [j for j in range(A.shape[1]) if j not in g]
    ng, nr = np.linalg.norm(D[:, g], 2), np.linalg.norm(D[:, rest], 2)
    mg = np.linalg.svd(A[:, g], compute_uv=False).min()
    mr = np.linalg.svd(A[:, rest], compute_uv=False).min()
    if ng >= mg or nr >= mr:
        return None
    return math.asin(min(1, ng / (mg - ng))) + math.asin(min(1, nr / (mr - nr)))


def e1e_operator_perturbation(rng, n_inst=40, n_pert=25, N=200, tau=TAU_THETA):
    """T3 on every candidate block; the reported partition must not change when every margin exceeds its bound."""
    held, total, changed, changed_with_margin, all_margins = 0, 0, 0, 0, 0
    for _ in range(n_inst):
        prob = planted_problem(rng, N, [3, 2, 1, 1], mode="near", omega=0.3, slack=1, nuisance_dim=5)
        A = prob.A
        J = A.shape[1]
        subsets = [[j for j in range(J) if m >> j & 1] for m in range(1, (1 << J) - 1)]
        theta0 = [math.asin(min(1, separation(A, g))) for g in subsets]
        base = group_report(A, GroupingConfig(tau_theta=tau))
        for _ in range(n_pert):
            D = rng.standard_normal(A.shape)
            D *= 10 ** rng.uniform(-4, -1) * np.linalg.norm(A, 2) / np.linalg.norm(D, 2)
            pert = group_report(A + D, GroupingConfig(tau_theta=tau))
            margins_ok = True
            for g, t0 in zip(subsets, theta0):
                b = _theta_bound(A, D, g)
                if b is None:
                    margins_ok = False
                    continue
                t1 = math.asin(min(1, separation(A + D, g)))
                total += 1
                held += int(abs(t1 - t0) <= b + 1e-12)
                if abs(t0 - math.asin(tau)) <= b:
                    margins_ok = False
            all_margins += int(margins_ok)
            if _canon(pert.reported) != _canon(base.reported):
                changed += 1
                changed_with_margin += int(margins_ok)
    return {"bound_held": held, "bound_checked": total,
            "perturbations": n_inst * n_pert,
            "perturbations_with_every_margin_above_bound": all_margins,
            "reported_partition_changed": changed,
            "changed_although_every_margin_exceeded_bound": changed_with_margin}


# --------------------------------------------------------------------------- #
# E2: baselines on one planted family
# --------------------------------------------------------------------------- #
def e2_baselines(rng, n_ops=40, n_draws=40, n_boot_draws=8, N=200, sigma=0.05, omega=1e-3):
    sizes = [4, 3, 1, 1]
    grouping_rules = ["iasa_separation", "exact_components", "pairwise_0.99", "bkw"]
    hit = {k: 0 for k in grouping_rules}
    worst_rel_err = {k: [] for k in grouping_rules}
    est = {k: {"near": [], "separated": []} for k in ("nnls", "ridge_gcv", "tsvd")}
    cover = {k: {"near": [], "separated": []} for k in ("posterior_correct", "posterior_shifted", "bootstrap")}
    width = {k: {"near": [], "separated": []} for k in cover}
    tsvd_interpretable, tsvd_rank, pw_within = [], [], []
    for _ in range(n_ops):
        prob = planted_problem(rng, N, sizes, mode="near_circuit", omega=omega, slack=4, nuisance_dim=5)
        A, c, truth = prob.A, prob.c, _canon(prob.blocks)
        prior_shifted = 1.0 + 0.5 * rng.choice([-1.0, 1.0], size=len(c))
        near_cols = {j for g in truth if len(g) > 1 for j in g}
        nrm = A / np.linalg.norm(A, axis=0)
        pw_within.append(max(abs(float(nrm[:, i] @ nrm[:, j])) for g in truth for i in g for j in g if i < j))
        parts = {
            "iasa_separation": group_report(A, GroupingConfig(tau_theta=TAU_THETA)).reported,
            "exact_components": exact_components(A),
            "pairwise_0.99": bl.pairwise_coherence_grouping(A, 0.99),
            "bkw": bl.bkw_grouping(A),
        }
        draws = [observe(rng, prob, sigma) for _ in range(n_draws)]
        fits = [bl.fit_nnls(A, y) for y in draws]
        signal = float(np.linalg.norm(A @ c))
        for k, P in parts.items():
            hit[k] += int(_canon(P) == truth)
            errs = np.array([_group_errors(A, ch, c, P) for ch in fits])
            worst_rel_err[k].append(float(np.sqrt((errs**2).mean(axis=0)).max() / signal))
        # group-signal accuracy of the estimators at the planted grouping
        for y, c_nn in zip(draws, fits):
            c_r, _ = bl.ridge_gcv(A, y)
            c_t, _, k_t = bl.tsvd_gcv(A, y)
            tsvd_rank.append(k_t)
            for name, ch in (("nnls", c_nn), ("ridge_gcv", c_r), ("tsvd", c_t)):
                for g in truth:
                    rel = np.linalg.norm(A[:, g] @ (ch[g] - c[g])) / np.linalg.norm(A[:, g] @ c[g])
                    est[name]["near" if len(g) > 1 else "separated"].append(float(rel))
        _, Vt, _ = bl.tsvd_gcv(A, draws[0])
        for v in Vt:
            tsvd_interpretable.append(float(max(np.sum(v[g] ** 2) for g in truth) >= 0.9))
        # coverage of 95% intervals for individual coefficients
        z = 1.959963984540054
        for y in draws[:n_boot_draws]:
            for name, mu0 in (("posterior_correct", np.ones_like(c)), ("posterior_shifted", prior_shifted)):
                m, S = bl.gaussian_posterior(A, y, sigma, mu0, 0.29)
                sd = np.sqrt(np.diag(S))
                for j in range(len(c)):
                    key = "near" if j in near_cols else "separated"
                    cover[name][key].append(float(abs(m[j] - c[j]) <= z * sd[j]))
                    width[name][key].append(float(2 * z * sd[j]))
            boot = bl.residual_bootstrap_nnls(rng, A, y, n_boot=200)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            for j in range(len(c)):
                key = "near" if j in near_cols else "separated"
                cover["bootstrap"][key].append(float(lo[j] <= c[j] <= hi[j]))
                width["bootstrap"][key].append(float(hi[j] - lo[j]))
    return_est = {k: {kk: float(np.median(vv)) for kk, vv in v.items()} for k, v in est.items()}
    return {
        "sizes": sizes, "omega": omega, "sigma": sigma, "N": N, "operators": n_ops, "draws": n_draws,
        "grouping_recovery_rate": {k: v / n_ops for k, v in hit.items()},
        "grouping_worst_block_rel_rmse_median": {k: float(np.median(v)) for k, v in worst_rel_err.items()},
        "group_signal_rel_error_median_at_planted_grouping": return_est,
        "interval_coverage": {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in cover.items()},
        "interval_width_median": {k: {kk: float(np.median(vv)) for kk, vv in v.items()} for k, v in width.items()},
        "tsvd_directions_within_one_block_rate": float(np.mean(tsvd_interpretable)),
        "tsvd_gcv_rank_mean": float(np.mean(tsvd_rank)),
        "pairwise_coherence_max_within_planted_blocks_median": float(np.median(pw_within)),
    }


# --------------------------------------------------------------------------- #
# E1f: recovering the latent grouping from an operator observed with error
# --------------------------------------------------------------------------- #
def _latent_three_near(rng, theta, N=10):
    """mu0 and mu2 exactly parallel, mu1 at angle theta from both, mu3 separate."""
    e = np.linalg.qr(rng.standard_normal((N, 4)))[0]
    mu0 = e[:, 0]
    mu1 = math.cos(theta) * e[:, 0] + math.sin(theta) * e[:, 1]
    mu2 = 1.3 * e[:, 0]
    mu3 = 0.5 * e[:, 0] + e[:, 2]
    return np.stack([mu0, mu1, mu2, mu3], axis=1)


def _block_rank(M, tau):
    return int((np.linalg.svd(M, compute_uv=False) > tau).sum())


def e1f_latent_grouping(rng, n_inst=6000, N=10):
    latent = [[0, 2], [1], [3]]
    amb = cond = cond_ok = rule_ok = ineq_ok = unique = 0
    for _ in range(n_inst):
        Hs = _latent_three_near(rng, rng.uniform(0.02, 0.4), N)
        sf = 10 ** rng.uniform(-3.5, -1.3)
        H = Hs + sf * rng.standard_normal(Hs.shape)
        tau = sf * (math.sqrt(N) + math.sqrt(4) + rng.uniform(1.0, 6.0))
        res = latent_grouping(H, tau, sigma_f=sf)
        if latent not in res.candidates:
            continue
        if any(_block_rank(H[:, g], tau) != np.linalg.matrix_rank(Hs[:, g], 1e-9) for g in latent):
            continue
        if len(res.candidates) == 1:
            unique += 1
            continue
        amb += 1
        err = float(np.linalg.norm(H - Hs, 2))
        ok = largest_discarded(H, latent, tau) <= err + 1e-12
        dmin = math.inf
        for P in res.candidates:
            if P == latent:
                continue
            dP = 0.0
            for g in P:
                k = _block_rank(H[:, g], tau)
                if k < len(g):
                    dP = max(dP, float(np.linalg.svd(Hs[:, g], compute_uv=False)[k]))
            ok = ok and dP > 0 and largest_discarded(H, P, tau) >= dP - err - 1e-12
            dmin = min(dmin, dP)
        ineq_ok += int(ok)
        rule_ok += int(res.estimate == latent)
        if err < dmin / 2:
            cond += 1
            cond_ok += int(res.estimate == latent)
    return {"instances": n_inst, "N": N, "J": 4,
            "latent_is_the_only_candidate": unique, "ambiguous_with_latent_candidate": amb,
            "inequalities_held": ineq_ok, "condition_met": cond, "rule_correct_when_condition_met": cond_ok,
            "rule_correct_all_ambiguous": rule_ok}


def _provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:  # noqa: BLE001
        sha, dirty = None, None
    import scipy
    import torch
    return {"git_sha": sha, "git_dirty": dirty, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "machine": platform.machine(), "system": platform.system()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "iasa_synthetic" / "results.json"))
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--parts", default="e1a,e1b,e1c,e1d,e1e,e2,e1f")
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    runners = {"e1a": e1a_exact_recovery, "e1b": e1b_within_block_conditioning, "e1c": e1c_phase_diagram,
               "e1d": e1d_greedy_vs_exhaustive, "e1e": e1e_operator_perturbation, "e2": e2_baselines,
               "e1f": e1f_latent_grouping}
    for i, (name, fn) in enumerate(runners.items()):
        if name not in parts:
            continue
        t0 = time.perf_counter()
        result[name] = fn(np.random.default_rng(args.seed + i))
        print(f"[{name}] {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
        result["provenance"] = {**_provenance(), "seed": args.seed}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
