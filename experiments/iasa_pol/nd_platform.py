"""Shared New Delhi base platform for the Task 10 controlled experiment suite.

The paper fixes ONE platform -- one regulatory sensor geometry, one PM2.5/wind
record, one set of proxy inventories -- and each experiment varies exactly one
controlled axis on top of it. This module builds that platform and exposes the
factories (winds, sensor layouts, compact controlled sources, background bases,
synthetic ground-truth coefficients) the experiments compose.

Everything a fit consumes -- the primary background basis ``Q``, the lag rule,
the fixed-zero mask, and the inventory version -- is DECLARED here / in the
experiment config and stamped into provenance; none is selected from ``Y`` or a
recovery score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from model.iasa.activity import TemporalBasis, build_default_activity_profile
from model.iasa.background import BackgroundBasisConfig, build_background_basis
from model.iasa.response import DispersionConfig, Observer, ResponseConfig
from model.iasa.wind import (
    ar1_synthetic,
    constant_direction,
    diurnal_synthetic,
    multi_direction_synthetic,
    single_direction_synthetic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = REPO_ROOT / "sim"

# The four declared proxy source groups (paper inventory). Order is fixed and is
# the inventory column order used everywhere downstream.
DEFAULT_INVENTORY_VERSION = "new_delhi_proxy_v1"

PLATFORM_START = "2018-05-01 00:00:00+05:30"


@dataclass(frozen=True)
class PlatformConfig:
    """Declared platform geometry and lag rule (never tuned to fit quality)."""

    grid_shape: tuple[int, int] = (20, 20)
    T: int = 24
    lag_window_steps: int = 10
    substep_dt: float = 0.25
    kernel_truncation_radius: float = 3.0
    sigma_parallel: float = 0.7
    sigma_perp: float = 0.25
    min_dispersion_time: float = 0.25
    # Primary background basis (rank-4: constant, centered linear, first daily sin/cos).
    background_polynomial_degree: int = 1
    background_daily_harmonics: int = 1
    inventory_version: str = DEFAULT_INVENTORY_VERSION
    device: str = "cpu"
    response_dtype: str = "float32"

    def to_json(self) -> dict[str, Any]:
        return {
            "grid_shape": list(self.grid_shape),
            "T": self.T,
            "lag_window_steps": self.lag_window_steps,
            "substep_dt": self.substep_dt,
            "kernel_truncation_radius": self.kernel_truncation_radius,
            "sigma_parallel": self.sigma_parallel,
            "sigma_perp": self.sigma_perp,
            "min_dispersion_time": self.min_dispersion_time,
            "background_polynomial_degree": self.background_polynomial_degree,
            "background_daily_harmonics": self.background_daily_harmonics,
            "inventory_version": self.inventory_version,
            "device": self.device,
            "response_dtype": self.response_dtype,
        }


@dataclass
class Platform:
    """Shared base platform reused (identically) on both sides of every comparison."""

    config: PlatformConfig
    source_names: list[str]
    source_maps: np.ndarray  # [K, Nx, Ny] downsampled, normalized proxy inventory
    observer: Observer  # regulatory geometry mapped to distinct grid cells
    timestamps: np.ndarray  # [T] datetime64[h]
    response_config: ResponseConfig
    dispersion_config: DispersionConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.source_maps.shape[1]), int(self.source_maps.shape[2]))

    @property
    def n_sources(self) -> int:
        return int(self.source_maps.shape[0])

    def background_config(self, mode: str = "primary") -> BackgroundBasisConfig:
        return build_background_config(self.config, mode)


# --------------------------------------------------------------------------- #
# Inventory + geometry loading
# --------------------------------------------------------------------------- #
def _block_downsample(maps: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Downsample [K, H, W] maps to [K, nx, ny] by block-mean (exact divisor) or
    nearest-index sampling otherwise. Renormalize each map to a unit maximum."""
    k, h, w = maps.shape
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if nx == h and ny == w:
        out = maps.astype(np.float32, copy=True)
    elif h % nx == 0 and w % ny == 0:
        fx, fy = h // nx, w // ny
        out = maps.reshape(k, nx, fx, ny, fy).mean(axis=(2, 4)).astype(np.float32)
    else:
        ix = np.clip(np.round(np.linspace(0, h - 1, nx)).astype(np.int64), 0, h - 1)
        iy = np.clip(np.round(np.linspace(0, w - 1, ny)).astype(np.int64), 0, w - 1)
        out = maps[:, ix][:, :, iy].astype(np.float32)
    peaks = out.reshape(k, -1).max(axis=1)
    peaks = np.where(peaks > 1e-12, peaks, 1.0)
    return (out / peaks[:, None, None]).astype(np.float32)


def load_inventory_maps(grid_shape: tuple[int, int]) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Real normalized proxy inventory downsampled to the platform grid."""
    import sim.pol_sources as pol_sources

    inv = pol_sources.load_pol_source_inventory(src_dir=str(SIM_DIR))
    maps = _block_downsample(np.asarray(inv.source_maps, dtype=np.float32), grid_shape)
    meta = {
        "inventory_normalization": inv.raw_metadata.get("normalization"),
        "all_zero_sources": inv.raw_metadata.get("all_zero_sources", []),
        "native_shape": list(np.asarray(inv.source_maps).shape),
        "downsampled_shape": list(maps.shape),
    }
    return list(inv.source_names), maps, meta


def _regulatory_grid_cells(grid_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Map real regulatory station lon/lat to distinct grid cells (deduped)."""
    from data.pol_weather import load_new_delhi_wind_data

    wind = load_new_delhi_wind_data(
        SIM_DIR / "govdata_1H_current.csv",
        SIM_DIR / "govdata_locations.csv",
        start="2018-05-01 00:00:00+05:30",
        end="2018-05-01 05:00:00+05:30",
    )
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    lon = np.asarray(wind.sensors_xy[:, 0], dtype=np.float64)
    lat = np.asarray(wind.sensors_xy[:, 1], dtype=np.float64)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-12)
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-12)
    ix = np.clip(np.rint((lon - np.nanmin(lon)) / lon_scale * (nx - 1)).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint((lat - np.nanmin(lat)) / lat_scale * (ny - 1)).astype(np.int64), 0, ny - 1)
    seen: dict[tuple[int, int], int] = {}
    cells: list[tuple[int, int]] = []
    ids: list[str] = []
    station_index: list[int] = []  # retained ORIGINAL station index per deduped cell
    for s, (cx, cy) in enumerate(zip(ix.tolist(), iy.tolist())):
        key = (cx, cy)
        if key in seen:
            continue
        seen[key] = s
        cells.append(key)
        ids.append(f"reg_{cx}_{cy}")
        station_index.append(s)
    xy = np.asarray(cells, dtype=np.float32)
    return xy, ids, station_index


# --------------------------------------------------------------------------- #
# Factories used by the experiments
# --------------------------------------------------------------------------- #
def compact_source(grid_shape: tuple[int, int], center: tuple[float, float], sigma: float = 1.2) -> np.ndarray:
    """A compact normalized Gaussian source blob (controlled-geometry primitive)."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    src = np.exp(-0.5 * (dx * dx + dy * dy) / float(sigma * sigma)).astype(np.float32)
    src[src < 0.05] = 0.0
    return (src / max(float(src.max()), 1e-12)).astype(np.float32)


def make_wind(kind: str, T: int, *, seed: int = 0, speed: float = 1.0, **kwargs: Any):
    """Wind regime factory covering the Experiment 4 axis.

    ``real`` requires the imputed New Delhi wind product (or observed fallback)
    and is resolved lazily so the synthetic regimes never touch disk.
    """
    start = kwargs.pop("start", PLATFORM_START)
    if kind == "constant":
        return constant_direction(length=T, vx=speed, vy=0.0, start=start)
    if kind == "single":
        return single_direction_synthetic(
            length=T, speed=speed, direction_degrees=float(kwargs.get("direction_degrees", 20.0)), start=start
        )
    if kind == "diurnal":
        return diurnal_synthetic(length=T, base_vx=speed, start=start, **kwargs)
    if kind == "ar1":
        return ar1_synthetic(length=T, seed=seed, start=start, **kwargs)
    if kind == "multi":
        return multi_direction_synthetic(length=T, speed=speed, seed=seed, start=start, **kwargs)
    if kind == "real":
        from model.iasa.wind import real_new_delhi_wind_sequence

        seq = real_new_delhi_wind_sequence(
            SIM_DIR / "govdata_1H_current.csv",
            SIM_DIR / "govdata_locations.csv",
            allow_observed_fallback=True,
        )
        return _truncate_wind(seq, T)
    raise ValueError(f"unknown wind kind {kind!r}")


def _truncate_wind(seq, T: int):
    from model.iasa.wind import WindSequence

    T = min(int(T), int(seq.vx.shape[0]))
    return WindSequence(
        timestamps=seq.timestamps[:T],
        vx=np.asarray(seq.vx[:T], dtype=np.float32),
        vy=np.asarray(seq.vy[:T], dtype=np.float32),
        provider=seq.provider,
        metadata=dict(seq.metadata),
    )


def sensor_layout(kind: str, grid_shape: tuple[int, int], *, n: int = 6, seed: int = 0,
                  regulatory: Observer | None = None) -> Observer:
    """Sensor-geometry factory covering the Experiment 4 layout axis."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if kind == "regulatory":
        if regulatory is None:
            raise ValueError("regulatory layout requires the platform's base observer")
        return regulatory
    rng = np.random.default_rng(seed)
    if kind == "random":
        xs = rng.integers(1, nx - 1, size=n)
        ys = rng.integers(1, ny - 1, size=n)
    elif kind == "downwind":
        # Cluster sensors on the downwind (east) half, spread across y.
        xs = rng.integers(nx // 2, nx - 1, size=n)
        ys = np.linspace(2, ny - 2, n).round().astype(np.int64)
    else:
        raise ValueError(f"unknown sensor layout {kind!r}")
    cells = {}
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        cells[(int(cx), int(cy))] = True
    xy = np.asarray(list(cells.keys()), dtype=np.float32)
    ids = [f"{kind}_{int(cx)}_{int(cy)}" for cx, cy in cells]
    return Observer(sensor_ids=ids, sensor_xy=xy)


def build_background_config(cfg: PlatformConfig, mode: str) -> BackgroundBasisConfig:
    """Declared background bases for Experiment 3.

    ``none`` -> empty; ``primary`` -> rank-4 (constant, linear, daily sin/cos);
    ``redundant``/``stress`` bases are constructed in the experiment from the
    primary Q so they share span / add a source-like column.
    """
    if mode == "none":
        return BackgroundBasisConfig(include_constant=False)
    if mode in ("primary", "redundant", "stress"):
        return BackgroundBasisConfig(
            include_constant=True,
            temporal_polynomial_degree=cfg.background_polynomial_degree,
            daily_harmonics=cfg.background_daily_harmonics,
            max_background_rank=8,
            basis_mode="normal" if mode == "primary" else "stress",
        )
    raise ValueError(f"unknown background mode {mode!r}")


def default_basis(kind: str, T: int) -> TemporalBasis:
    """Declared temporal basis. ``impulse_constant`` is the controlled default;
    ``multi`` gives the diurnal/block/day-night bases for temporal recovery."""
    if kind == "impulse_constant":
        values = np.zeros((T, 2), dtype=np.float32)
        values[min(2, T - 1), 0] = 1.0
        values[:, 1] = 1.0
        return TemporalBasis(names=["impulse_t2", "constant"], values=values,
                             metadata={"kind": kind})
    if kind == "constant":
        return TemporalBasis(names=["constant"], values=np.ones((T, 1), dtype=np.float32),
                             metadata={"kind": kind})
    if kind == "multi":
        hours = np.arange(T) % 24
        diurnal = np.exp(-0.5 * ((hours - 8) / 2.0) ** 2) + np.exp(-0.5 * ((hours - 18) / 2.0) ** 2)
        block = (hours < 12).astype(np.float32)
        day_night = ((hours >= 7) & (hours <= 19)).astype(np.float32) * 0.65 + 0.35
        values = np.stack([diurnal, block, day_night], axis=1).astype(np.float32)
        return TemporalBasis(names=["diurnal", "block", "day_night"], values=values,
                             metadata={"kind": kind})
    raise ValueError(f"unknown basis kind {kind!r}")


def synthetic_coefficients(n_sources: int, n_basis: int, *, seed: int = 0,
                           zero_fraction: float = 0.0) -> np.ndarray:
    """Deterministic nonnegative ground-truth source-basis coefficients c (source-major)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.3, 1.2, size=n_sources * n_basis).astype(np.float64)
    if zero_fraction > 0.0:
        k = int(round(zero_fraction * c.shape[0]))
        c[rng.choice(c.shape[0], size=k, replace=False)] = 0.0
    return c


# --------------------------------------------------------------------------- #
# Platform builder
# --------------------------------------------------------------------------- #
def build_platform(cfg: PlatformConfig | None = None) -> Platform:
    cfg = cfg or PlatformConfig()
    source_names, source_maps, inv_meta = load_inventory_maps(cfg.grid_shape)
    station_index: list[int] = []
    try:
        xy, ids, station_index = _regulatory_grid_cells(cfg.grid_shape)
        geometry = "regulatory_new_delhi"
    except Exception as exc:  # pragma: no cover - only if station CSVs are missing
        # Deterministic fallback so the platform is always constructible offline.
        rng = np.random.default_rng(0)
        nx, ny = cfg.grid_shape
        cells = {(int(x), int(y)) for x, y in zip(rng.integers(1, nx - 1, 8), rng.integers(1, ny - 1, 8))}
        xy = np.asarray(sorted(cells), dtype=np.float32)
        ids = [f"fallback_{int(x)}_{int(y)}" for x, y in xy]
        station_index = list(range(len(ids)))
        geometry = f"synthetic_fallback ({exc})"
    observer = Observer(sensor_ids=ids, sensor_xy=xy)
    timestamps = np.datetime64("2018-05-01T00:00") + np.arange(cfg.T) * np.timedelta64(1, "h")
    response_config = ResponseConfig(
        dt=1.0, lag_window_steps=cfg.lag_window_steps, substep_dt=cfg.substep_dt,
        kernel_truncation_radius=cfg.kernel_truncation_radius,
        device=cfg.device, response_dtype=cfg.response_dtype,
    )
    dispersion_config = DispersionConfig(
        sigma_parallel=cfg.sigma_parallel, sigma_perp=cfg.sigma_perp,
        min_dispersion_time=cfg.min_dispersion_time,
    )
    metadata = {
        "inventory_version": cfg.inventory_version,
        "geometry": geometry,
        "n_regulatory_sensors": len(ids),
        # Original raw-station index for each (deduped) regulatory sensor, so observed
        # PM2.5 (in original station order) can be joined to the correct sensor.
        "regulatory_station_index": list(station_index),
        **inv_meta,
    }
    return Platform(
        config=cfg, source_names=list(source_names), source_maps=source_maps,
        observer=observer, timestamps=timestamps, response_config=response_config,
        dispersion_config=dispersion_config, metadata=metadata,
    )
