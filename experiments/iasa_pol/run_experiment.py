#!/usr/bin/env python3
"""Run a single configured Task 10 controlled experiment reproducibly.

sbatch-friendly: non-interactive, all progress on stderr, a clean JSON summary on
stdout, writes artifacts under ``runs/<experiment>_seed<N>/`` and exits.

Usage:
    python3 experiments/iasa_pol/run_experiment.py --config experiments/iasa_pol/configs/exp01.json --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.iasa_pol import runio  # noqa: E402
from experiments.iasa_pol.experiments import run_named_experiment  # noqa: E402
from experiments.iasa_pol.nd_platform import PlatformConfig, build_platform  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def build_platform_config(cfg: dict, device: str) -> PlatformConfig:
    p = dict(cfg.get("platform", {}))
    if "grid_shape" in p:
        p["grid_shape"] = tuple(p["grid_shape"])
    p["device"] = device
    return PlatformConfig(**p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/*.json experiment config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    experiment = cfg["experiment"]
    _log(f"[run] experiment={experiment} seed={args.seed} device={args.device}")

    platform_cfg = build_platform_config(cfg, args.device)
    platform = build_platform(platform_cfg)
    _log(f"[run] platform grid={platform.grid_shape} sensors={len(platform.observer.sensor_ids)} "
         f"geometry={platform.metadata.get('geometry')}")

    out = run_named_experiment(experiment, platform, cfg.get("params", {}), args.seed)
    resolved = runio.resolved_config(
        cfg, seed=args.seed, device=args.device,
        platform_meta=platform.metadata, platform_config=platform_cfg.to_json(),
    )
    run_dir = runio.write_run(
        Path(args.out), experiment=experiment, seed=args.seed,
        resolved=resolved, result=out["result"], arrays=out["arrays"],
    )
    _log(f"[run] wrote {run_dir}")
    print(json.dumps({
        "status": "ok",
        "experiment": experiment,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "result": out["result"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
