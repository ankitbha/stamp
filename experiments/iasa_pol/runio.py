"""Provenance stamping and artifact I/O for the Task 10 experiment runner.

Every run records a resolved config (config + seed + git SHA + device/dtype +
torch/cuda versions) so results are reproducible from provenance alone, and the
primary Q / lag rule / fixed-zero mask / inventory version are recoverable
without inspecting fit quality.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.iasa.backend import runtime_provenance


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def resolved_config(config: dict[str, Any], *, seed: int, device: str, platform_meta: dict[str, Any],
                    platform_config: dict[str, Any]) -> dict[str, Any]:
    dev = torch.device(device)
    prov = runtime_provenance(dev, torch.float64)
    return {
        "config": config,
        "seed": int(seed),
        "device": device,
        "git_sha": git_sha(),
        "runtime": prov,
        "platform_config": platform_config,
        "platform_metadata": platform_meta,
        "inventory_version": platform_config.get("inventory_version"),
        "primary_background": "constant+linear+daily_sin1+daily_cos1 (declared)",
        "lag_rule": "declared_in_platform_config.lag_window_steps (not tuned to fit)",
    }


def write_run(out_dir: Path, *, experiment: str, seed: int, resolved: dict[str, Any],
              result: dict[str, Any], arrays: dict[str, np.ndarray]) -> Path:
    run_dir = Path(out_dir) / f"{experiment}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    if arrays:
        np.savez_compressed(run_dir / "arrays.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    return run_dir
