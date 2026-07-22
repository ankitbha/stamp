#!/usr/bin/env python3
"""Task 11 evaluation driver: build paper-matching report tables from saved runs.

Loads an IASA runs directory (controlled ``<exp>_seed<N>/`` and windowed observed
``week<K>/observed_seed<N>/``), dispatches each to model/iasa/reporting.py, and
writes machine-readable (``report.json``) + human-readable (``report.md`` and one
``<label>.csv`` per table) reports.

Works on a single saved run or aggregates many runs of the same experiment
(rows concatenated with a ``run`` provenance column).

Usage:
    python3 evaluation/eval_pol_iasa.py --runs evaluation/iasa_pol/runs \
        --out evaluation/iasa_pol/reports [--wind-report checkpoints/xxx.report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa import reporting  # noqa: E402


def load_runs(runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (controlled_results, observed_results). Each result is tagged with its
    run directory name under ``_run`` for provenance."""
    controlled, observed = [], []
    for result_path in sorted(runs_dir.rglob("result.json")):
        result = json.loads(result_path.read_text())
        result["_run"] = result_path.parent.relative_to(runs_dir).as_posix()
        if result.get("experiment") == "observed_new_delhi":
            observed.append(result)
        else:
            controlled.append(result)
    return controlled, observed


def _merge_tables(tables: list[reporting.ReportTable]) -> list[reporting.ReportTable]:
    """Concatenate tables that share a label (multi-run aggregation), tagging each
    row with its ``run`` and keeping ``run`` as the leading column."""
    by_label: dict[str, reporting.ReportTable] = {}
    order: list[str] = []
    for t in tables:
        lbl = t["label"]
        if lbl not in by_label:
            by_label[lbl] = {"label": lbl, "title": t["title"],
                             "columns": list(t["columns"]), "rows": [], "notes": list(t["notes"])}
            order.append(lbl)
        agg = by_label[lbl]
        for r in t["rows"]:
            agg["rows"].append(r)
        for n in t["notes"]:
            if n not in agg["notes"]:
                agg["notes"].append(n)
    # The leading `run` column (for multi-run aggregation) is added upstream in
    # build_report before merging; here we just preserve first-seen label order.
    return [by_label[lbl] for lbl in order]


def build_report(controlled: list[dict[str, Any]], observed: list[dict[str, Any]],
                 wind_report: dict[str, Any] | None) -> list[reporting.ReportTable]:
    tagged: list[reporting.ReportTable] = []
    # Controlled: build per run, tag rows with run so aggregation stays traceable.
    for result in controlled:
        run = result.get("_run", "")
        for tbl in reporting.report_result(result):
            if len(_runs_for(controlled, result.get("experiment"))) > 1:
                tbl["columns"] = ["run"] + tbl["columns"]
                for row in tbl["rows"]:
                    row["run"] = run
            tagged.append(tbl)
    tables = _merge_tables(tagged)
    if observed:
        tables.extend(reporting.report_observed(observed))
        tables.append(reporting.report_wind_imputation(wind_report))
    return tables


def _runs_for(controlled: list[dict[str, Any]], experiment: str | None) -> list[str]:
    return [r.get("_run") for r in controlled if r.get("experiment") == experiment]


def _render_md(tables: list[reporting.ReportTable]) -> str:
    lines = ["# IASA Evaluation Report", ""]
    for t in tables:
        lines.append(f"## {t['title']}")
        lines.append("")
        cols = t["columns"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in t["rows"]:
            lines.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
        lines.append("")
        for n in t["notes"]:
            lines.append(f"- _{n}_")
        lines.append("")
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def _write_csv(table: reporting.ReportTable, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table["columns"], extrasaction="ignore")
        w.writeheader()
        for r in table["rows"]:
            w.writerow({c: _cell(r.get(c)) if isinstance(r.get(c), (list, dict)) else r.get(c)
                        for c in table["columns"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="evaluation/iasa_pol/runs")
    ap.add_argument("--out", default="evaluation/iasa_pol/reports")
    ap.add_argument("--wind-report", default=None,
                    help="Optional wind-imputation checkpoint .report.json")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    controlled, observed = load_runs(runs_dir)
    wind_report = None
    if args.wind_report and Path(args.wind_report).exists():
        wind_report = json.loads(Path(args.wind_report).read_text())

    tables = build_report(controlled, observed, wind_report)
    # allow_nan=False: refuse to emit bare NaN/Infinity (invalid JSON); reporting
    # already scrubs NaN->null, so this is a belt-and-suspenders guard.
    (out / "report.json").write_text(json.dumps(
        {"n_tables": len(tables), "n_controlled_runs": len(controlled),
         "n_observed_runs": len(observed), "tables": tables},
        indent=2, sort_keys=False, allow_nan=False))
    (out / "report.md").write_text(_render_md(tables))
    for t in tables:
        _write_csv(t, out / f"{t['label']}.csv")

    print(json.dumps({"status": "ok", "n_tables": len(tables),
                      "n_controlled_runs": len(controlled),
                      "n_observed_runs": len(observed),
                      "labels": [t["label"] for t in tables]}, indent=2))


if __name__ == "__main__":
    main()
