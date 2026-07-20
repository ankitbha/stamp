#!/usr/bin/env python3
"""Roll up ``runs/`` into small committed summary tables for Task 11.

Scans ``runs/<experiment>_seed<N>/result.json`` and emits:
  - ``summaries/<experiment>_summary.csv`` -- flat per-row accuracy + diagnostics
  - ``summaries/all_experiments.json`` -- combined index Task 11 reads

CSV columns are the union of scalar keys across an experiment's rows; nested
blocks (accuracy/diagnostics) are flattened with dotted keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, (list, tuple)):
        # Keep short scalar lists as JSON; skip long/array payloads in the CSV.
        if len(obj) <= 8 and all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
            out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def _rows_for(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the per-row list an experiment reports (rows/grid_rows/parametric),
    or a single flattened summary row when the experiment has no explicit rows."""
    for key in ("rows", "grid_rows"):
        if key in result and isinstance(result[key], list):
            base = {k: v for k, v in result.items()
                    if k not in ("rows", "grid_rows", "arrays") and not isinstance(v, (list, dict))}
            flat_rows = []
            for row in result[key]:
                merged = dict(base)
                _flatten("", row, merged)
                flat_rows.append(merged)
            return flat_rows
    single: dict[str, Any] = {}
    _flatten("", {k: v for k, v in result.items() if k != "arrays"}, single)
    return [single]


def summarize(runs_dir: Path, summaries_dir: Path) -> dict[str, Any]:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    for run_dir in sorted(runs_dir.glob("*_seed*")):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        experiment = result.get("experiment", run_dir.name.split("_seed")[0])
        rows = _rows_for(result)
        index.setdefault(experiment, {"runs": [], "n_rows": 0})
        index[experiment]["runs"].append(run_dir.name)
        index[experiment]["n_rows"] += len(rows)

        csv_path = summaries_dir / f"{experiment}_summary.csv"
        fieldnames: list[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    combined = summaries_dir / "all_experiments.json"
    combined.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--summaries", default=str(Path(__file__).resolve().parent / "summaries"))
    args = parser.parse_args()
    index = summarize(Path(args.runs), Path(args.summaries))
    print(json.dumps({"status": "ok", "experiments": index}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
