#!/usr/bin/env python3
"""Read-only smoke check for the STAMP pollution runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


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


def main() -> None:
    dependency_status = _import_status(REQUIRED_MODULES + OPTIONAL_MODULES)
    missing_required = [name for name in REQUIRED_MODULES if not dependency_status[name]]
    if missing_required:
        raise RuntimeError(f"Missing required runtime modules: {missing_required}")

    import pandas as pd
    from data.pol_weather import load_new_delhi_wind_data, wind_to_transport_uv
    import sim.polsim as polsim

    source, raw_sources = polsim.load_known_sources_40x40(src_dir=str(SIM_DIR))
    grid = polsim.make_grid(
        Nx=40,
        Ny=40,
        src_dir=str(SIM_DIR),
        load_sources=True,
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

    print("# STAMP IASA runtime smoke check")
    print(f"repo_root: {REPO_ROOT}")
    print(f"dependency_status: {dependency_status}")
    print("polsim_import: ok")
    print(f"S_known_shape: {_shape(source)}")
    print(f"raw_source_keys: {sorted(raw_sources.keys())}")
    print(f"grid_shape: ({grid.Nx}, {grid.Ny})")
    print(f"grid_S_known_shape: {_shape(grid.S_known)}")
    print(f"location_rows: {len(locations)}")
    print(f"hourly_data_station_count: {station_ids.nunique()}")
    print(f"hourly_data_columns: {list(data_header.columns)}")
    print(f"wind_smoke_station_count: {len(wind.station_ids)}")
    print(f"wind_smoke_timestamp_count: {len(wind.timestamps)}")
    print(f"wind_smoke_vector_shape: {_shape(wind.observed_vectors)}")
    print(f"wind_cardinal_U_x: {[round(float(x), 6) for x in cardinal_ux]}")
    print(f"wind_cardinal_V_y: {[round(float(y), 6) for y in cardinal_vy]}")


if __name__ == "__main__":
    main()
