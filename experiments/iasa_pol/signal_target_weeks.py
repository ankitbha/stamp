#!/usr/bin/env python3
"""Signal-target re-analysis of the four observed New Delhi weeks (paper Section 6).

Reads the committed projected operators and fits from
``evaluation/iasa_pol/runs/week{1..4}/observed_seed0`` (no pipeline re-run) and
reports, per week: the exact components, the separation of each source group,
the reported grouping, the bound on each group's signal error at the
instrument-noise level estimated from the two co-located Pusa monitors, and a
parametric bootstrap of the apportionment shares at that noise level.

Usage:
    python experiments/iasa_pol/signal_target_weeks.py --out evaluation/iasa_pol/signal_target/weeks.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.grouping import GroupingConfig, exact_components, group_report, separation  # noqa: E402

RUNS = REPO_ROOT / "evaluation" / "iasa_pol" / "runs"
N_BASIS = 7  # columns are ordered source-major: column = 7 * source + basis


def pusa_noise(data_csv: Path, window: tuple[str, str] | None = None) -> dict:
    """Per-monitor error sd and lag-1 autocorrelation from the two co-located Pusa monitors.

    If the two monitors' errors are independent with equal variance, the
    difference has variance 2 sigma_e^2.  The lag-1 autocorrelation of the
    difference series estimates the AR(1) coefficient of the errors.
    """
    df = pd.read_csv(data_csv, parse_dates=["timestamp_round"])
    wide = df.pivot_table(index="timestamp_round", columns="monitor_id", values="pm25")
    ids = [c for c in wide.columns if c.lower().startswith("pusa")]
    if len(ids) != 2:
        raise ValueError(f"expected two Pusa monitors, found {ids}")
    pair = wide[ids]
    if window is not None:
        pair = pair.loc[window[0]:window[1]]
    pair = pair.asfreq("h")
    d = (pair[ids[0]] - pair[ids[1]])
    both = d.notna()
    lagged = both & both.shift(1, fill_value=False)
    phi = float(np.corrcoef(d[lagged].to_numpy(), d.shift(1)[lagged].to_numpy())[0, 1])
    sd = float(d[both].std(ddof=1))
    return {"monitors": ids, "hours": int(both.sum()), "sd_difference": sd,
            "sigma_e": sd / math.sqrt(2.0), "lag1_autocorrelation": phi,
            "ar1_inflation": (1 + phi) / (1 - phi) if phi < 1 else math.inf}


def analyse_week(week: int, sigmas: dict[str, float], tau_theta: float, n_boot: int, delta: float, rng) -> dict:
    run = RUNS / f"week{week}" / "observed_seed0"
    res = json.loads((run / "result.json").read_text())
    arr = np.load(run / "arrays.npz")
    fixed = set(res["fixed_zero_indices"])
    active = [j for j in range(arr["H_tilde"].shape[1]) if j not in fixed]
    H = arr["H_tilde"][:, active].astype(np.float64)
    c_hat = arr["c_hat"][active].astype(np.float64)
    names = res["source_names"]
    declared = [[i for i, j in enumerate(active) if j // N_BASIS == k] for k in range(len(names))]
    N = H.shape[0]
    comps = exact_components(H)
    report = group_report(H, GroupingConfig(tau_theta=tau_theta), declared=declared)
    seps = {names[k]: separation(H, g) for k, g in enumerate(declared)}
    ranks = {names[k]: int(np.linalg.matrix_rank(H[:, g])) for k, g in enumerate(declared)}
    signals = {names[k]: H[:, g] @ c_hat[g] for k, g in enumerate(declared)}
    l1 = {k: float(np.abs(v).sum()) for k, v in signals.items()}
    total = sum(l1.values()) or 1.0
    shares = {k: v / total for k, v in l1.items()}
    U = np.linalg.qr(H)[0]
    fitted = H @ c_hat

    out_groups = {}
    for k, g in enumerate(declared):
        name = names[k]
        norm = float(np.linalg.norm(signals[name]))
        bounds = {}
        r_all = int(np.linalg.matrix_rank(H))
        for label, sg in sigmas.items():
            # least squares: Theorem (b) with the group rank; NNLS: Theorem (a) with
            # ||P_A eps|| <= sigma (sqrt(rank A) + sqrt(2 log(1/delta))) w.p. 1 - delta
            for est, r in (("ls", ranks[name]), ("nnls", r_all)):
                b = sg * (math.sqrt(r) + math.sqrt(2 * math.log(1 / delta))) / seps[name]
                bounds[f"{label}_{est}"] = {"error_norm_bound": b, "rms_per_reading_bound": b / math.sqrt(N),
                                            "bound_over_signal_norm": b / norm if norm > 0 else math.inf}
        out_groups[name] = {"columns": [active[i] for i in g], "rank": ranks[name], "separation": seps[name],
                            "signal_norm": norm, "signal_rms_per_reading": norm / math.sqrt(N),
                            "share": shares[name], "bounds": bounds}

    boot = {}
    for label, sg in sigmas.items():
        draws = np.empty((n_boot, len(names)))
        for b in range(n_boot):
            # only the component of the noise in col(H) changes the NNLS fit
            y = fitted + U @ (U.T @ (sg * rng.standard_normal(N)))
            c_b, _ = nnls(H, y, maxiter=5000)
            mags = np.array([np.abs(H[:, g] @ c_b[g]).sum() for g in declared])
            draws[b] = mags / (mags.sum() or 1.0)
        boot[label] = {names[k]: {"p2.5": float(np.percentile(draws[:, k], 2.5)),
                                  "p50": float(np.percentile(draws[:, k], 50)),
                                  "p97.5": float(np.percentile(draws[:, k], 97.5))}
                       for k in range(len(names))}

    sv = np.linalg.svd(H, compute_uv=False)
    return {"week": week, "N": N, "J": len(active), "singular_values": sv.tolist(),
            "exact_components": comps, "declared_partition": declared,
            "reported": report.reported, "reported_separations": report.separations,
            "minimal_partitions": report.minimal_partitions, "groups": out_groups,
            "share_bootstrap": boot, "residual_norm": res.get("residual_norm")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "iasa_pol" / "signal_target" / "weeks.json"))
    parser.add_argument("--data", default=str(REPO_ROOT / "sim" / "govdata_1H_current.csv"))
    parser.add_argument("--tau-theta", type=float, default=math.sqrt(1 - 0.99**2))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    noise_all = pusa_noise(Path(args.data))
    noise_may = pusa_noise(Path(args.data), ("2018-05-01", "2018-05-28 23:59"))
    sigma = noise_all["sigma_e"]
    sigmas = {"iid": sigma, "ar1_inflated": sigma * math.sqrt(noise_all["ar1_inflation"])}
    rng = np.random.default_rng(args.seed)
    weeks = [analyse_week(w, sigmas, args.tau_theta, args.n_boot, args.delta, rng) for w in (1, 2, 3, 4)]
    out = {"noise_full_record": noise_all, "noise_may_2018": noise_may, "sigmas_used": sigmas,
           "tau_theta": args.tau_theta, "delta": args.delta, "n_boot": args.n_boot, "seed": args.seed,
           "weeks": weeks}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
