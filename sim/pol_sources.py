from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_NAMES: tuple[str, ...] = (
    "brick_kilns",
    "industries",
    "population_density",
    "traffic_00",
    "traffic_06",
    "traffic_12",
    "traffic_18",
)

SOURCE_FILES: dict[str, str] = {
    "brick_kilns": "brick_kilns_intensity_80x80.npy",
    "industries": "industries_intensity_80x80.npy",
    "population_density": "population_density_intensity_80x80.npy",
    "traffic_00": "traffic_00_intensity_80x80.npy",
    "traffic_06": "traffic_06_intensity_80x80.npy",
    "traffic_12": "traffic_12_intensity_80x80.npy",
    "traffic_18": "traffic_18_intensity_80x80.npy",
}

DEFAULT_CROP: tuple[slice, slice] = (slice(21, 61), slice(16, 56))


@dataclass
class PolSourceInventory:
    source_names: list[str]
    source_maps: np.ndarray
    source_matrix: np.ndarray
    source_activity_defaults: dict[str, Any]
    raw_metadata: dict[str, Any]


def _slice_bounds(slc: slice) -> tuple[int | None, int | None, int | None]:
    return slc.start, slc.stop, slc.step


def _normalize_cropped_source(cropped: np.ndarray, eps: float) -> tuple[np.ndarray, float, bool]:
    scale = float(np.percentile(cropped, 99))
    all_zero = bool(np.count_nonzero(cropped) == 0)
    if all_zero or scale <= eps:
        return np.zeros_like(cropped, dtype=np.float32), 1.0, all_zero
    return (cropped / (scale + eps)).astype(np.float32, copy=False), scale, all_zero


def load_pol_source_inventory(
    src_dir: str | Path = "./",
    *,
    crop: tuple[slice, slice] = DEFAULT_CROP,
    dtype: np.dtype = np.float32,
    eps: float = 1e-12,
) -> PolSourceInventory:
    """
    Load named pollution source inventories for the IASA path.

    Each source map is cropped to the 40x40 New Delhi window and normalized by
    its own cropped 99th percentile. Source categories are never summed here.
    """
    src_path = Path(src_dir)
    source_names = list(SOURCE_NAMES)
    maps: list[np.ndarray] = []
    scale_by_source: dict[str, float] = {}
    all_zero_sources: list[str] = []
    file_by_source: dict[str, str] = {}
    raw_stats: dict[str, dict[str, float | int]] = {}

    for name in source_names:
        file_name = SOURCE_FILES[name]
        path = src_path / file_name
        raw = np.load(path).astype(dtype, copy=False)
        cropped = raw[crop].astype(dtype, copy=False)
        normalized, scale, all_zero = _normalize_cropped_source(cropped, float(eps))
        maps.append(normalized)
        scale_by_source[name] = float(scale)
        file_by_source[name] = str(path)
        if all_zero:
            all_zero_sources.append(name)
        raw_stats[name] = {
            "raw_min": float(np.nanmin(raw)),
            "raw_max": float(np.nanmax(raw)),
            "cropped_min": float(np.nanmin(cropped)),
            "cropped_max": float(np.nanmax(cropped)),
            "cropped_p99": float(np.percentile(cropped, 99)),
            "cropped_nonzero": int(np.count_nonzero(cropped)),
        }

    source_maps = np.stack(maps, axis=0).astype(np.float32, copy=False)
    source_matrix = source_maps.reshape(len(source_names), -1).T.copy()

    raw_metadata: dict[str, Any] = {
        "source_files": file_by_source,
        "crop": {
            "axis0": _slice_bounds(crop[0]),
            "axis1": _slice_bounds(crop[1]),
        },
        "normalization": "per_source_cropped_p99",
        "scale_by_source": scale_by_source,
        "all_zero_sources": all_zero_sources,
        "raw_stats": raw_stats,
    }
    source_activity_defaults: dict[str, Any] = {
        "traffic_time_slices": ["traffic_00", "traffic_06", "traffic_12", "traffic_18"],
        "normalization": "per_source_cropped_p99",
    }

    return PolSourceInventory(
        source_names=source_names,
        source_maps=source_maps,
        source_matrix=source_matrix,
        source_activity_defaults=source_activity_defaults,
        raw_metadata=raw_metadata,
    )


__all__ = [
    "DEFAULT_CROP",
    "SOURCE_FILES",
    "SOURCE_NAMES",
    "PolSourceInventory",
    "load_pol_source_inventory",
]
