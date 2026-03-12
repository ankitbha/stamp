from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.abspath(".."))
import sim.polsim as polsim  # noqa: E402


Array = np.ndarray


def _to_numpy(x: Any) -> Array:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _extract_sensor_series(npz: Dict[str, Any], use_noisy: bool = True, traj: int = 0) -> Array:
    key = "sensor_noisy" if use_noisy and "sensor_noisy" in npz else "sensor_clean"
    if key not in npz:
        raise KeyError(f"Expected 'sensor_noisy' or 'sensor_clean' in dataset. Found keys: {list(npz.keys())}")

    arr = _to_numpy(npz[key]).astype(np.float32)
    if arr.ndim == 2:
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    if arr.ndim == 3:
        arr = arr[traj]
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    raise ValueError(f"Unsupported sensor series shape: {arr.shape}")


def _pearson_flat(a: Array, b: Array, eps: float = 1e-12) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= eps:
        return float("nan")
    return float(np.sum(a * b) / den)


def _mean_sensor_temporal_corr(pred_st: Array, obs_st: Array) -> float:
    cors = []
    for s in range(pred_st.shape[0]):
        c = _pearson_flat(pred_st[s], obs_st[s])
        if np.isfinite(c):
            cors.append(c)
    return float(np.mean(cors)) if cors else float("nan")


def _spatial_grad_corr(a_hw: Array, b_hw: Array) -> float:
    ax, ay = np.gradient(a_hw.astype(np.float64))
    bx, by = np.gradient(b_hw.astype(np.float64))
    ga = np.concatenate([ax.reshape(-1), ay.reshape(-1)])
    gb = np.concatenate([bx.reshape(-1), by.reshape(-1)])
    return _pearson_flat(ga, gb)


def _metrics(pred: Array, gt: Array, eps: float = 1e-12) -> Dict[str, float]:
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    gt_rms = float(np.sqrt(np.mean(gt ** 2)))
    nrmse = float(rmse / (gt_rms + eps))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
    }


def _load_pred_unknown(calib_npz: Dict[str, Any], key_hint: Optional[str] = None) -> Array:
    if key_hint:
        if key_hint not in calib_npz:
            raise KeyError(f"Requested key '{key_hint}' not in calibration file. Keys={list(calib_npz.keys())}")
        arr = _to_numpy(calib_npz[key_hint]).astype(np.float32)
        return arr[0] if arr.ndim == 3 else arr

    candidates = ["S_unknown", "s_unknown", "theta_S_unknown", "unknown", "S"]
    for k in candidates:
        if k in calib_npz:
            arr = _to_numpy(calib_npz[k]).astype(np.float32)
            return arr[0] if arr.ndim == 3 else arr

    for k in calib_npz.keys():
        arr = _to_numpy(calib_npz[k])
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
            return arr[0].astype(np.float32)

    raise KeyError(f"Could not infer predicted S_unknown key from calibration file keys={list(calib_npz.keys())}")


@dataclass
class EvalResult:
    calib_path: str
    unknown_error: Dict[str, float]
    sensor_error: Dict[str, float]
    sensor_corr_flat: float
    sensor_corr_mean_per_sensor: float
    unknown_corr_flat: float
    unknown_corr_grad: float


@dataclass
class EvalConfig:
    data_path: str = "data/pol_dataset.npz"
    calib_paths: List[str] = field(default_factory=lambda: [
        "calib_simgrad_pol_best.npz",
        "calib_stamp_pol_best.npz",
    ])
    pred_key: Optional[str] = None
    use_noisy: bool = False
    wind_seed: int = 123
    device: str = "cpu"


CFG = EvalConfig()


def evaluate_one(
    data_path: str,
    calib_path: str,
    *,
    use_noisy: bool,
    pred_key: Optional[str],
    wind_seed: int,
    device: str,
) -> EvalResult:
    data = dict(np.load(data_path, allow_pickle=True))
    calib = dict(np.load(calib_path, allow_pickle=True))

    s_unknown_true = _to_numpy(data["S_unknown_coarse"]).astype(np.float32)
    s_unknown_pred = _load_pred_unknown(calib, key_hint=pred_key)

    if s_unknown_pred.shape != s_unknown_true.shape:
        raise ValueError(
            f"Predicted S_unknown shape {s_unknown_pred.shape} does not match true {s_unknown_true.shape}."
        )

    sensor_obs = _extract_sensor_series(data, use_noisy=use_noisy)
    sensors_idx = _to_numpy(data["sensors_idx"]).astype(np.int64)

    H = int(len(_to_numpy(data["x"]))) if "x" in data else 40
    W = int(len(_to_numpy(data["y"]))) if "y" in data else 40

    root = os.path.abspath(os.path.join(os.path.dirname(data_path), ".."))
    sim_dir = os.path.join(root, "sim")

    torch_device = torch.device(device)
    grid = polsim.make_grid(Nx=H, Ny=W, src_dir=sim_dir, device=torch_device, dtype=torch.float32, load_sources=True)
    if "S_known" in data:
        grid.S_known = torch.from_numpy(_to_numpy(data["S_known"]).astype(np.float32)).to(torch_device)

    params = polsim.PolParams()
    if "params" in data and "param_names" in data:
        vals = _to_numpy(data["params"]).reshape(-1)
        names = [str(x) for x in _to_numpy(data["param_names"]).reshape(-1)]
        for n, v in zip(names, vals):
            if hasattr(params, n):
                setattr(params, n, float(v))

    dt = float(_to_numpy(data["dt"])) if "dt" in data else float(np.median(np.diff(_to_numpy(data["t"]).astype(float))))
    steps = int(_to_numpy(data["steps"])) if "steps" in data else int(sensor_obs.shape[1] - 1)
    save_every = int(_to_numpy(data["save_every"])) if "save_every" in data else 1

    out = polsim.rollout_pollution(
        S_unknown=torch.from_numpy(s_unknown_pred).to(torch_device),
        grid=grid,
        params=params,
        dt=dt,
        steps=steps,
        save_every=save_every,
        enforce_cfl=True,
        wind_seed=wind_seed,
        no_grad=True,
    )
    U = out["U"]
    ix = torch.as_tensor(sensors_idx[:, 1], device=torch_device, dtype=torch.long)
    iy = torch.as_tensor(sensors_idx[:, 0], device=torch_device, dtype=torch.long)
    sensor_pred = U[ix, iy, :].detach().cpu().numpy().astype(np.float32)

    T = min(sensor_pred.shape[1], sensor_obs.shape[1])
    sensor_pred = sensor_pred[:, :T]
    sensor_obs = sensor_obs[:, :T]

    return EvalResult(
        calib_path=calib_path,
        unknown_error=_metrics(s_unknown_pred, s_unknown_true),
        sensor_error=_metrics(sensor_pred, sensor_obs),
        sensor_corr_flat=_pearson_flat(sensor_pred, sensor_obs),
        sensor_corr_mean_per_sensor=_mean_sensor_temporal_corr(sensor_pred, sensor_obs),
        unknown_corr_flat=_pearson_flat(s_unknown_pred, s_unknown_true),
        unknown_corr_grad=_spatial_grad_corr(s_unknown_pred, s_unknown_true),
    )


def main() -> None:
    print("# Pollution calibration evaluation")
    print(f"dataset={CFG.data_path}")

    for cp in CFG.calib_paths:
        if not os.path.exists(cp):
            print(f"\n[{cp}] MISSING (skipped)")
            continue

        res = evaluate_one(
            data_path=CFG.data_path,
            calib_path=cp,
            use_noisy=CFG.use_noisy,
            pred_key=CFG.pred_key,
            wind_seed=CFG.wind_seed,
            device=CFG.device,
        )

        print(f"\n[{res.calib_path}]")
        print(
            "1) S_unknown error: "
            f"mse={res.unknown_error['mse']:.6g}, rmse={res.unknown_error['rmse']:.6g}, "
            f"mae={res.unknown_error['mae']:.6g}, nrmse={res.unknown_error['nrmse']:.6g}"
        )
        print(
            "2) Sensor error: "
            f"mse={res.sensor_error['mse']:.6g}, rmse={res.sensor_error['rmse']:.6g}, "
            f"mae={res.sensor_error['mae']:.6g}, nrmse={res.sensor_error['nrmse']:.6g}"
        )
        print(
            "3) Sensor spatio-temporal structure (correlation): "
            f"flat_pearson={res.sensor_corr_flat:.6g}, "
            f"mean_temporal_corr_per_sensor={res.sensor_corr_mean_per_sensor:.6g}"
        )
        print(
            "4) S_unknown spatial structure (correlation): "
            f"flat_pearson={res.unknown_corr_flat:.6g}, grad_field_corr={res.unknown_corr_grad:.6g}"
        )


if __name__ == "__main__":
    main()