#!/usr/bin/env python3
"""Read-only smoke check for the STAMP pollution runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "sim"
REQUIRED_MODULES = ("numpy", "torch", "pandas", "pykrige", "einops")
OPTIONAL_MODULES = ("scipy",)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_status(module_names: tuple[str, ...]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception:
            status[name] = False
        else:
            status[name] = True
    return status


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{name}={actual!r}; expected {expected!r}")


def main() -> None:
    dependency_status = _import_status(REQUIRED_MODULES + OPTIONAL_MODULES)
    missing_required = [name for name in REQUIRED_MODULES if not dependency_status[name]]
    if missing_required:
        raise RuntimeError(f"Missing required runtime modules: {missing_required}")

    import pandas as pd
    from data.pol_weather import load_new_delhi_wind_data, wind_to_transport_uv
    import sim.polsim as polsim

    inventory = polsim.load_pol_source_inventory(src_dir=SIM_DIR)
    grid = polsim.make_grid(
        Nx=40,
        Ny=40,
        src_dir=str(SIM_DIR),
        load_sources=False,
        load_inventory=True,
    )

    locations_path = SIM_DIR / "govdata_locations.csv"
    data_path = SIM_DIR / "govdata_1H_current.csv"

    locations = pd.read_csv(locations_path, index_col=0)
    data_header = pd.read_csv(data_path, nrows=0)
    station_ids = pd.read_csv(data_path, usecols=["monitor_id"])["monitor_id"]
    wind = load_new_delhi_wind_data(
        data_path,
        locations_path,
        start="2018-05-01 00:00:00+05:30",
        end="2018-05-01 23:00:00+05:30",
    )
    cardinal_ux, cardinal_vy = wind_to_transport_uv(
        pd.Series([0.0, 90.0, 180.0, 270.0]).to_numpy(),
        pd.Series([1.0, 1.0, 1.0, 1.0]).to_numpy(),
    )
    expected_cardinal_ux = np.asarray([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
    expected_cardinal_vy = np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    expected_sources = [
        "brick_kilns",
        "industries",
        "population_density",
        "traffic_00",
        "traffic_06",
        "traffic_12",
        "traffic_18",
    ]
    _assert_equal("source_names", inventory.source_names, expected_sources)
    _assert_equal("source_maps_shape", _shape(inventory.source_maps), (7, 40, 40))
    _assert_equal("source_matrix_shape", _shape(inventory.source_matrix), (1600, 7))
    _assert_equal("grid_shape", (grid.Nx, grid.Ny), (40, 40))
    _assert_equal("grid_source_names", grid.source_names, expected_sources)
    _assert_equal("grid_source_maps_shape", _shape(grid.source_maps), (7, 40, 40))
    _assert_equal("grid_source_matrix_shape", _shape(grid.source_matrix), (1600, 7))
    if grid.S_known is not None:
        raise RuntimeError("IASA inventory grid should not populate aggregate S_known.")
    if inventory.raw_metadata.get("normalization") != "per_source_cropped_p99":
        raise RuntimeError("Expected per-source p99 normalization metadata.")
    if "traffic_06" not in inventory.raw_metadata.get("all_zero_sources", []):
        raise RuntimeError("Expected traffic_06 to be recorded as an all-zero source.")
    _assert_equal("wind_smoke_station_count", len(wind.station_ids), 32)
    _assert_equal("wind_smoke_timestamp_count", len(wind.timestamps), 24)
    _assert_equal("wind_smoke_vector_shape", _shape(wind.observed_vectors), (32, 24, 2))
    if wind.vector_mask.dtype != np.bool_:
        raise RuntimeError("wind vector_mask must be boolean.")
    if not wind.vector_mask.any():
        raise RuntimeError("wind smoke window has no valid observed wind vectors.")
    if not np.isfinite(wind.observed_vectors[wind.vector_mask]).all():
        raise RuntimeError("wind smoke window contains non-finite observed vectors where vector_mask is valid.")
    np.testing.assert_allclose(cardinal_ux, expected_cardinal_ux, atol=1e-6)
    np.testing.assert_allclose(cardinal_vy, expected_cardinal_vy, atol=1e-6)

    print("# STAMP IASA runtime smoke check")
    print(f"repo_root: {REPO_ROOT}")
    print(f"dependency_status: {dependency_status}")
    print("polsim_import: ok")
    print(f"source_names: {inventory.source_names}")
    print(f"source_maps_shape: {_shape(inventory.source_maps)}")
    print(f"source_matrix_shape: {_shape(inventory.source_matrix)}")
    print(f"source_normalization: {inventory.raw_metadata['normalization']}")
    print(f"all_zero_sources: {inventory.raw_metadata['all_zero_sources']}")
    print(f"grid_shape: ({grid.Nx}, {grid.Ny})")
    print(f"grid_source_maps_shape: {_shape(grid.source_maps)}")
    print(f"location_rows: {len(locations)}")
    print(f"hourly_data_station_count: {station_ids.nunique()}")
    print(f"hourly_data_columns: {list(data_header.columns)}")
    print(f"wind_smoke_station_count: {len(wind.station_ids)}")
    print(f"wind_smoke_timestamp_count: {len(wind.timestamps)}")
    print(f"wind_smoke_vector_shape: {_shape(wind.observed_vectors)}")
    print(f"wind_cardinal_U_x: {[round(float(x), 6) for x in cardinal_ux]}")
    print(f"wind_cardinal_V_y: {[round(float(y), 6) for y in cardinal_vy]}")
    print("smoke_assertions: ok")


if __name__ == "__main__":
    main()
