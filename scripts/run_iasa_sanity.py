#!/usr/bin/env python3
"""Minimal IASA sanity runner for the active pollution inventory path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "sim"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_activity(timestamps: Any, source_names: list[str]) -> np.ndarray:
    """Build a small deterministic temporal activity matrix [T,K]."""
    hours = np.asarray([int(ts.hour) for ts in timestamps], dtype=np.int64)
    activity = np.zeros((len(hours), len(source_names)), dtype=np.float32)
    for i, name in enumerate(source_names):
        if name == "brick_kilns":
            activity[:, i] = 0.25
        elif name == "industries":
            activity[:, i] = 0.5
        elif name == "population_density":
            activity[:, i] = 1.0
        elif name.startswith("traffic_"):
            slot_hour = int(name.rsplit("_", 1)[1])
            activity[:, i] = (hours == slot_hour).astype(np.float32)
    return activity


def _station_grid_indices(sensors_xy: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    lon = sensors_xy[:, 0].astype(np.float64)
    lat = sensors_xy[:, 1].astype(np.float64)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-12)
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-12)
    ix = np.rint((lon - np.nanmin(lon)) / lon_scale * (nx - 1)).astype(np.int64)
    iy = np.rint((lat - np.nanmin(lat)) / lat_scale * (ny - 1)).astype(np.int64)
    return np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)


def run_sanity(*, start: str, end: str) -> dict[str, Any]:
    from data.pol_weather import load_new_delhi_wind_data
    import sim.polsim as polsim

    inventory = polsim.load_pol_source_inventory(src_dir=SIM_DIR)
    grid = polsim.make_grid(Nx=40, Ny=40, src_dir=str(SIM_DIR), load_inventory=True)
    wind = load_new_delhi_wind_data(
        SIM_DIR / "govdata_1H_current.csv",
        SIM_DIR / "govdata_locations.csv",
        start=start,
        end=end,
    )

    if grid.source_names != inventory.source_names:
        raise RuntimeError("Grid and inventory source names disagree.")
    if grid.source_maps is None or grid.source_matrix is None or grid.source_metadata is None:
        raise RuntimeError("Grid inventory fields were not populated.")
    if inventory.raw_metadata.get("normalization") != "per_source_cropped_p99":
        raise RuntimeError("Expected per-source p99 source normalization.")
    if wind.observed_vectors.shape[:2] != (len(wind.station_ids), len(wind.timestamps)):
        raise RuntimeError("Wind vector shape does not match station/time axes.")

    activity = _default_activity(wind.timestamps, inventory.source_names)
    if activity.shape != (len(wind.timestamps), len(inventory.source_names)):
        raise RuntimeError("Activity matrix has unexpected shape.")
    if np.any(activity < 0):
        raise RuntimeError("Activity matrix must be nonnegative.")

    source_terms = np.einsum("tk,kxy->txy", activity, inventory.source_maps, optimize=True).astype(np.float32)
    if source_terms.shape != (len(wind.timestamps), grid.Nx, grid.Ny):
        raise RuntimeError("Inventory activity source terms have unexpected shape.")
    if not np.isfinite(source_terms).all():
        raise RuntimeError("Inventory activity source terms contain non-finite values.")

    ix, iy = _station_grid_indices(wind.sensors_xy, grid.Nx, grid.Ny)
    sensor_source_terms = source_terms[:, ix, iy].T
    pm25_mask_count = int(np.count_nonzero(wind.raw_pm25_mask))
    wind_mask_count = int(np.count_nonzero(wind.vector_mask[..., 0]))
    if pm25_mask_count == 0:
        raise RuntimeError("Sanity window has no observed pm25 values.")
    if wind_mask_count == 0:
        raise RuntimeError("Sanity window has no observed wind vectors.")
    if not np.isfinite(sensor_source_terms).all():
        raise RuntimeError("Sensor source-term matrix contains non-finite values.")

    return {
        "status": "ok",
        "gate": "task3a_minimal_iasa",
        "source_names": inventory.source_names,
        "source_maps_shape": list(inventory.source_maps.shape),
        "source_matrix_shape": list(inventory.source_matrix.shape),
        "activity_shape": list(activity.shape),
        "source_terms_shape": list(source_terms.shape),
        "sensor_source_terms_shape": list(sensor_source_terms.shape),
        "station_count": int(len(wind.station_ids)),
        "timestamp_count": int(len(wind.timestamps)),
        "observed_pm25_count": pm25_mask_count,
        "observed_wind_vector_count": wind_mask_count,
        "normalization": inventory.raw_metadata["normalization"],
        "all_zero_sources": inventory.raw_metadata["all_zero_sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-05-01 00:00:00+05:30")
    parser.add_argument("--end", default="2018-05-01 23:00:00+05:30")
    args = parser.parse_args()
    print(json.dumps(run_sanity(start=args.start, end=args.end), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
