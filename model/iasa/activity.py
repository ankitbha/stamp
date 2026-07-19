from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from model.iasa.backend import to_numpy


TRAFFIC_SLOT_HOURS: tuple[int, ...] = (0, 6, 12, 18)
POPULATION_COOKING_HOURS: tuple[int, ...] = (7, 13, 19)


@dataclass(frozen=True)
class ActivityProfile:
    source_names: list[str]
    timestamps: Any
    theta: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TemporalBasis:
    names: list[str]
    values: np.ndarray
    metadata: dict[str, Any]


def _hours_from_timestamps(timestamps: Any) -> np.ndarray:
    hours = []
    for ts in timestamps:
        if not hasattr(ts, "hour"):
            raise TypeError("timestamps must contain datetime-like values with an hour attribute")
        hours.append(int(ts.hour))
    return np.asarray(hours, dtype=np.int64)


def _circular_hour_distance(hours: np.ndarray, center: int) -> np.ndarray:
    delta = np.abs(hours.astype(np.int64) - int(center))
    return np.minimum(delta, 24 - delta).astype(np.float32)


def _traffic_slot_for_hours(hours: np.ndarray) -> np.ndarray:
    slots = np.asarray(TRAFFIC_SLOT_HOURS, dtype=np.int64)
    distances = np.stack([_circular_hour_distance(hours, int(slot)) for slot in slots], axis=1)
    return slots[np.argmin(distances, axis=1)]


def validate_activity_theta(theta: np.ndarray, *, nonnegative: bool = True) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=np.float32)
    if theta_arr.ndim not in (1, 2):
        raise ValueError("theta must have shape [K] or [T,K]")
    if not np.isfinite(theta_arr).all():
        raise ValueError("theta must contain only finite values")
    if nonnegative and np.any(theta_arr < 0):
        raise ValueError("theta must be nonnegative")
    return theta_arr


def validate_source_names(source_names: Sequence[str]) -> list[str]:
    names = [str(name) for name in source_names]
    if not names:
        raise ValueError("source_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("source_names must be unique")
    return names


def build_default_activity_profile(
    source_names: Sequence[str],
    timestamps: Any,
    *,
    seed: int = 0,
    industry_24h_fraction: float = 0.25,
) -> ActivityProfile:
    names = validate_source_names(source_names)
    hours = _hours_from_timestamps(timestamps)
    if not 0.0 <= float(industry_24h_fraction) <= 1.0:
        raise ValueError("industry_24h_fraction must be in [0,1]")

    theta = np.zeros((len(hours), len(names)), dtype=np.float32)
    traffic_slots = _traffic_slot_for_hours(hours)
    is_day = ((hours >= 7) & (hours <= 19)).astype(np.float32)
    industry_day_only = np.where(is_day > 0, 1.0, 0.35).astype(np.float32)
    industry_profile = (
        float(industry_24h_fraction) * np.ones_like(industry_day_only)
        + (1.0 - float(industry_24h_fraction)) * industry_day_only
    ).astype(np.float32)
    population_profile = np.full(len(hours), 0.2, dtype=np.float32)
    for peak_hour in POPULATION_COOKING_HOURS:
        distance = _circular_hour_distance(hours, peak_hour)
        population_profile += (0.8 * np.exp(-0.5 * (distance / 1.25) ** 2)).astype(np.float32)
    population_profile = np.clip(population_profile, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    brick_phase = int(rng.integers(0, 4)) if len(hours) else 0
    brick_block = (((np.arange(len(hours)) + brick_phase) // 12) % 2 == 0).astype(np.float32)
    brick_profile = (0.25 + 0.35 * brick_block).astype(np.float32)

    for i, name in enumerate(names):
        if name == "brick_kilns":
            theta[:, i] = brick_profile
        elif name == "industries":
            theta[:, i] = industry_profile
        elif name == "population_density":
            theta[:, i] = population_profile
        elif name.startswith("traffic_"):
            slot_hour = int(name.rsplit("_", 1)[1])
            theta[:, i] = (traffic_slots == slot_hour).astype(np.float32)

    metadata: dict[str, Any] = {
        "profile": "task4_default_proxy_activity",
        "seed": int(seed),
        "traffic_slot_hours": list(TRAFFIC_SLOT_HOURS),
        "traffic_assignment": "nearest_slot",
        "brick_kilns": {
            "baseline": 0.25,
            "block_increment": 0.35,
            "block_hours": 12,
            "phase": brick_phase,
        },
        "industries": {
            "day_hours": [7, 19],
            "night_fraction": 0.35,
            "industry_24h_fraction": float(industry_24h_fraction),
            "spatial_sampling": "metadata_only_v1",
        },
        "population_density": {
            "baseline": 0.2,
            "cooking_peak_hours": list(POPULATION_COOKING_HOURS),
            "peak_width_hours": 1.25,
        },
    }
    return ActivityProfile(source_names=names, timestamps=timestamps, theta=theta, metadata=metadata)


def build_theta_from_temporal_basis(
    source_names: Sequence[str],
    timestamps: Any,
    basis: TemporalBasis | np.ndarray,
    coefficients: np.ndarray,
    *,
    nonnegative: bool = True,
) -> ActivityProfile:
    names = validate_source_names(source_names)
    if isinstance(basis, TemporalBasis):
        basis_names = list(basis.names)
        basis_values = np.asarray(basis.values, dtype=np.float32)
        basis_metadata = dict(basis.metadata)
    else:
        basis_values = np.asarray(basis, dtype=np.float32)
        basis_names = [f"basis_{i}" for i in range(basis_values.shape[1] if basis_values.ndim == 2 else 0)]
        basis_metadata = {}
    coeff = np.asarray(coefficients, dtype=np.float32)
    if basis_values.ndim != 2:
        raise ValueError("basis must have shape [T,B]")
    if coeff.shape != (len(names), basis_values.shape[1]):
        raise ValueError("coefficients must have shape [K,B]")
    if len(timestamps) != basis_values.shape[0]:
        raise ValueError("timestamps length must match basis time dimension")
    if not np.isfinite(basis_values).all() or not np.isfinite(coeff).all():
        raise ValueError("basis and coefficients must contain only finite values")
    theta = to_numpy(
        torch.einsum(
            "tb,kb->tk",
            torch.as_tensor(basis_values, dtype=torch.float32),
            torch.as_tensor(coeff, dtype=torch.float32),
        )
    ).astype(np.float32)
    if nonnegative and np.any(theta < 0):
        raise ValueError("temporal-basis coefficients produced negative activity")
    metadata: dict[str, Any] = {
        "profile": "temporal_basis_coefficients",
        "basis_names": basis_names,
        "basis_metadata": basis_metadata,
        "coefficient_shape": list(coeff.shape),
    }
    return ActivityProfile(source_names=names, timestamps=timestamps, theta=theta, metadata=metadata)


def combine_inventory_sources(
    source_maps: np.ndarray,
    theta: np.ndarray,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    maps = np.asarray(source_maps, dtype=np.float32)
    theta_arr = validate_activity_theta(theta, nonnegative=nonnegative)
    if maps.ndim != 3:
        raise ValueError("source_maps must have shape [K,Nx,Ny]")
    if theta_arr.shape[-1] != maps.shape[0]:
        raise ValueError(
            f"theta has {theta_arr.shape[-1]} source columns but source_maps has {maps.shape[0]} maps"
        )
    if not np.isfinite(maps).all():
        raise ValueError("source_maps must contain only finite values")
    if nonnegative and np.any(maps < 0):
        raise ValueError("source_maps must be nonnegative")
    maps_t = torch.as_tensor(maps, dtype=torch.float32)
    theta_t = torch.as_tensor(theta_arr, dtype=torch.float32)
    if theta_arr.ndim == 1:
        return to_numpy(torch.einsum("k,kxy->xy", theta_t, maps_t)).astype(np.float32)
    return to_numpy(torch.einsum("tk,kxy->txy", theta_t, maps_t)).astype(np.float32)


__all__ = [
    "ActivityProfile",
    "POPULATION_COOKING_HOURS",
    "TRAFFIC_SLOT_HOURS",
    "TemporalBasis",
    "build_default_activity_profile",
    "build_theta_from_temporal_basis",
    "combine_inventory_sources",
    "validate_activity_theta",
    "validate_source_names",
]
