#!/usr/bin/env python3
"""Tier-0 summarizer for the windowed observed New Delhi study.

Collects the per-week ``observed`` result.json files (weeks 1..N, each a
self-contained study over a consecutive one-week window of the real record) into
a single paper-facing table. NOTHING is aggregated across windows: each week is
reported on its own terms (its own report groups, its own diagnostics). This is
the "weeks 1 to N" artifact.

Usage:
    python3 experiments/iasa_pol/summarize_weeks.py \
        --runs evaluation/iasa_pol/runs/week1/observed_seed0 \
               evaluation/iasa_pol/runs/week2/observed_seed0 ... \
        --out evaluation/iasa_pol/summaries
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "result.json").read_text())


def summarize(run_dirs: list[Path]) -> dict:
    weeks = []
    for rd in run_dirs:
        r = _load(rd)
        diag = r.get("diagnostics", {}) or {}
        weeks.append({
            "run_dir": str(rd),
            "window_index": r.get("window_index"),
            "window_start": r.get("window_start"),
            "window_end": r.get("window_end"),
            "window_hours": r.get("window_hours"),
            "status": r.get("status", "ok"),
            "n_source_groups": r.get("n_source_groups"),
            "source_names": r.get("source_names"),
            "n_observed_rows": r.get("n_observed_rows"),
            "n_total_rows": r.get("n_total_rows"),
            "observed_mask_fraction": r.get("observed_mask_fraction"),
            "wind_provider": r.get("wind_provider"),
            "pm25_imputed": r.get("pm25_imputed"),
            "kriged_baseline_subtracted": r.get("kriged_baseline_subtracted"),
            "calibration_status": r.get("calibration_status"),
            "residual_norm": r.get("residual_norm"),
            "sensor_signal_contribution_shares": r.get("sensor_signal_contribution_shares"),
            "report_components": r.get("report_components"),
            "sigma_J": diag.get("sigma_J"),
            "numerical_rank": diag.get("numerical_rank"),
            "condition_status": diag.get("condition_status"),
            "max_eligible_coherence": diag.get("max_eligible_coherence"),
            "weak_set": diag.get("weak_set"),
        })
    weeks.sort(key=lambda w: (w["window_index"] is None, w["window_index"]))
    return {
        "study": "observed_new_delhi_weeks",
        "tier": 0,
        "aggregation": "none -- each week is a self-contained study; report groups and "
                       "diagnostics may differ across weeks and are NOT combined",
        "n_weeks": len(weeks),
        "weeks": weeks,
    }


def _write_csv(summary: dict, path: Path) -> None:
    # One row per (week, source group): share is only meaningful per group, and the
    # group set is identical within a week (report groups noted separately).
    fields = ["window_index", "window_start", "window_end", "status", "source_group",
              "sensor_signal_share", "n_observed_rows", "observed_mask_fraction",
              "sigma_J", "numerical_rank", "condition_status", "max_eligible_coherence",
              "residual_norm", "calibration_status"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for wk in summary["weeks"]:
            shares = wk.get("sensor_signal_contribution_shares") or {}
            base = {k: wk.get(k) for k in
                    ("window_index", "window_start", "window_end", "status",
                     "n_observed_rows", "observed_mask_fraction", "sigma_J",
                     "numerical_rank", "condition_status", "max_eligible_coherence",
                     "residual_norm", "calibration_status")}
            if shares:
                for group, share in shares.items():
                    w.writerow({**base, "source_group": group, "sensor_signal_share": share})
            else:
                w.writerow({**base, "source_group": None, "sensor_signal_share": None})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="Per-week run directories")
    ap.add_argument("--out", default="evaluation/iasa_pol/summaries")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize([Path(r) for r in args.runs])
    (out / "observed_weeks.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(summary, out / "observed_weeks.csv")
    print(json.dumps({"status": "ok", "n_weeks": summary["n_weeks"],
                      "out_json": str(out / "observed_weeks.json"),
                      "out_csv": str(out / "observed_weeks.csv")}, indent=2))


if __name__ == "__main__":
    main()
