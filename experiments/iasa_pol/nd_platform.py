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

from model.iasa.activity import (
    POPULATION_COOKING_HOURS,
    TRAFFIC_SLOT_HOURS,
    TemporalBasis,
    build_default_activity_profile,
)
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
    """Reduce [K, H, W] maps to [K, nx, ny] preserving the DECLARED own-p99
    normalization applied upstream in ``pol_sources.load_pol_source_inventory``.

    At native resolution (nx==H) the maps pass through unchanged. When a smaller
    grid is requested we block-mean (exact divisor) or nearest-index sample; a
    local average of p99-normalized cells stays in the same p99 units. We do NOT
    re-normalize to unit maximum -- doing so would silently override the paper's
    per-source cropped-p99 normalization (would change relative source scales,
    hence sigma_J / visibility / coefficients)."""
    k, h, w = maps.shape
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if nx == h and ny == w:
        return maps.astype(np.float32, copy=True)
    if h % nx == 0 and w % ny == 0:
        fx, fy = h // nx, w // ny
        return maps.reshape(k, nx, fx, ny, fy).mean(axis=(2, 4)).astype(np.float32)
    ix = np.clip(np.round(np.linspace(0, h - 1, nx)).astype(np.int64), 0, h - 1)
    iy = np.clip(np.round(np.linspace(0, w - 1, ny)).astype(np.int64), 0, w - 1)
    return maps[:, ix][:, :, iy].astype(np.float32)


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


def make_wind(kind: str, T: int, *, seed: int = 0, speed: float = 1.0,
              grid_shape: tuple[int, int] | None = None, **kwargs: Any):
    """Wind regime factory covering the Experiment 4 axis.

    ``real`` returns the real New Delhi GRIDDED wind field (a spatially varying
    ``GriddedWindSampler`` built from observed station vectors via the adopted
    kernel coordinate-query imputer) when ``grid_shape`` is given -- this is the
    "imputed grid wind field" the response operator consumes and avoids the
    city-level zero-fill fallback. Without ``grid_shape`` it degrades to the
    city-level real sequence. Synthetic regimes never touch disk.
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
        if grid_shape is not None:
            return real_gridded_wind_sampler(grid_shape, T, start=start, seed=seed)
        from model.iasa.wind import real_new_delhi_wind_sequence

        seq = real_new_delhi_wind_sequence(
            SIM_DIR / "govdata_1H_current.csv",
            SIM_DIR / "govdata_locations.csv",
            allow_observed_fallback=True,
        )
        return _truncate_wind(seq, T)
    raise ValueError(f"unknown wind kind {kind!r}")


def real_gridded_wind_sampler(grid_shape: tuple[int, int], T: int, *,
                              start: str = PLATFORM_START, seed: int = 0):
    """Real New Delhi gridded wind field over a T-hour window as a GriddedWindSampler.

    Uses observed station transport vectors interpolated to every grid cell by the
    adopted kernel coordinate-query imputer (the field the response operator was
    designed to consume). The sampler clamps t_index/position, so a short/masked
    window is handled gracefully. Returns the GriddedWindSampler."""
    import pandas as pd

    from model.iasa.response import GriddedWindSampler
    from model.iasa.wind import gridded_new_delhi_wind_field

    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(hours=int(T) - 1)
    field = gridded_new_delhi_wind_field(
        SIM_DIR / "govdata_1H_current.csv",
        SIM_DIR / "govdata_locations.csv",
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        start=str(start_ts), end=str(end_ts), seed=seed,
    )
    sampler = GriddedWindSampler.from_gridded_wind_field(field)
    return sampler


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


def four_group_inventory(platform: "Platform") -> tuple[list[str], np.ndarray]:
    """The paper's FOUR source groups. The inventory ships traffic as four
    time-slot spatial maps (traffic_00/06/12/18); the paper models traffic as a
    SINGLE road-network source with a time-invariant congestion pattern and slot
    temporal bases, so we collapse the four slot maps into one road map (their
    mean). Returns (group_names[4], group_maps[4, Nx, Ny]) in p99 units."""
    names = list(platform.source_names)
    maps = np.asarray(platform.source_maps, dtype=np.float32)
    traffic_idx = [i for i, n in enumerate(names) if n.startswith("traffic_")]
    other_idx = [i for i, n in enumerate(names) if not n.startswith("traffic_")]
    group_names = [names[i] for i in other_idx] + (["traffic"] if traffic_idx else [])
    group_maps = [maps[i] for i in other_idx]
    if traffic_idx:
        group_maps.append(maps[traffic_idx].mean(axis=0))
    return group_names, np.stack(group_maps, axis=0).astype(np.float32)


def paper_temporal_bases(timestamps: Any, group_names: Sequence[str]) -> tuple[TemporalBasis, list[list[int]], tuple[int, ...]]:
    """Per-group declared temporal bases and the fixed-zero mask F0.

    Builds a combined basis whose components are: the four traffic nearest-slot
    indicators (00/06/12/18), a brick-kiln alternating-12h-block component, an
    industries day/night component, and a population cooking-peaks component. Each
    group is admissible ONLY on its own component(s); F0 fixes every non-admissible
    (group, component) coefficient to zero (paper 7.evaluation Inventories).

    Returns (TemporalBasis[T, B], admissible_components_per_group,
    fixed_zero_indices) with columns ordered source-major/basis-minor (J = K*B).
    """
    hours = _hours_of_day(timestamps)
    T = len(hours)
    comps: list[np.ndarray] = []
    comp_names: list[str] = []
    comp_owner: list[str] = []  # group name that owns each component

    slots = np.asarray(TRAFFIC_SLOT_HOURS, dtype=np.int64)
    nearest = slots[np.argmin(np.abs(((hours[:, None] - slots[None, :] + 12) % 24) - 12), axis=1)]
    for h0 in TRAFFIC_SLOT_HOURS:
        comps.append((nearest == h0).astype(np.float32))
        comp_names.append(f"traffic_slot_{h0:02d}")
        comp_owner.append("traffic")

    block = (((np.arange(T)) // 12) % 2 == 0).astype(np.float32)
    comps.append(0.25 + 0.35 * block); comp_names.append("kiln_block_12h"); comp_owner.append("brick_kilns")

    day_night = np.where((hours >= 7) & (hours <= 19), 1.0, 0.35).astype(np.float32)
    comps.append(day_night); comp_names.append("industry_day_night"); comp_owner.append("industries")

    cooking = np.full(T, 0.2, dtype=np.float32)
    for peak in POPULATION_COOKING_HOURS:
        d = np.abs(((hours - peak + 12) % 24) - 12)
        cooking += 0.8 * np.exp(-0.5 * (d / 1.25) ** 2)
    comps.append(np.clip(cooking, 0.0, 1.0)); comp_names.append("population_cooking"); comp_owner.append("population_density")

    values = np.stack(comps, axis=1).astype(np.float32)  # [T, B]
    B = values.shape[1]
    basis = TemporalBasis(names=comp_names, values=values,
                          metadata={"kind": "paper_per_group_bases", "component_owner": comp_owner})

    admissible: list[list[int]] = []
    fixed_zero: list[int] = []
    for k, gname in enumerate(group_names):
        own = [b for b in range(B) if comp_owner[b] == gname]
        if not own:  # a group with no declared component keeps a flat fallback (its slot-0)
            own = [0]
        admissible.append(own)
        for b in range(B):
            if b not in own:
                fixed_zero.append(k * B + b)  # source-major/basis-minor column index
    return basis, admissible, tuple(sorted(fixed_zero))


def krige_initial_condition(grid_shape: tuple[int, int], station_xy: np.ndarray,
                            station_values: np.ndarray, *, length_scale: float | None = None) -> np.ndarray:
    """Spatially interpolate first-hour station PM2.5 onto the response grid to form
    the initial concentration field U0 (a Gaussian-kernel ordinary-kriging surrogate,
    dependency-free and consistent with the repo's kernel wind imputer).

    The station values come from the loader's Pusa-AVERAGED 32-sensor layout (the two
    Pusa monitors are already merged upstream in ``load_new_delhi_wind_data``), so the
    field is Pusa-averaged by construction -- not the drop-one behavior of the legacy
    ``polsim.build_U0_from_govdata_kriging``. Nonnegative in, nonnegative out."""
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    xy = np.asarray(station_xy, dtype=np.float64).reshape(-1, 2)
    vals = np.asarray(station_values, dtype=np.float64).reshape(-1)
    if xy.shape[0] == 0:
        return np.zeros((nx, ny), dtype=np.float32)
    ell = float(length_scale) if length_scale is not None else max(2.0, min(nx, ny) / 6.0)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    acc = np.zeros((nx, ny), dtype=np.float64)
    wsum = np.zeros((nx, ny), dtype=np.float64)
    for (sx, sy), v in zip(xy, vals):
        w = np.exp(-0.5 * ((xs - sx) ** 2 + (ys - sy) ** 2) / (ell * ell))
        acc += w * v
        wsum += w
    U0 = acc / np.maximum(wsum, 1e-12)
    return np.clip(U0, 0.0, None).astype(np.float32)


def _hours_of_day(timestamps: Any) -> np.ndarray:
    ts = np.asarray(timestamps)
    if np.issubdtype(ts.dtype, np.datetime64):
        return ((ts.astype("datetime64[h]").astype(np.int64)) % 24).astype(np.int64)
    return (np.arange(len(ts)) % 24).astype(np.int64)


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
