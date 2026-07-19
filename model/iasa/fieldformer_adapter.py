"""FieldFormer coordinate-query wind imputer adapter (Task 9A).

Wraps the vendored coordinate-query FieldFormer (``baselines/fieldformer``) behind
the ``CoordinateQueryImputer`` protocol used by ``build_gridded_wind_field``, so a
trained FieldFormer can drive the gridded wind field exactly like the default
kernel imputer -- with no change to the response/diagnostics/fitting APIs.

FieldFormer for IASA wind predicts a 2-vector ``(Ux, Vy)`` field, so the wrapped
model must have ``out_dim == 2``. The upstream trained checkpoints are scalar
pollution/heat/SWE fields; a 2-vector wind checkpoint must be trained before this
imputer produces meaningful wind (see ``baselines/fieldformer/README.md``). The
default IASA wind imputer therefore remains ``KernelCoordinateQueryImputer``; this
adapter is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from baselines.fieldformer.model import (
    FieldFormerCoordinateQuery,
    SplitAwareSparseNeighborIndexer,
    load_fieldformer_checkpoint,
)


@dataclass
class FieldFormerCoordinateQueryImputer:
    """Coordinate-query imputer backed by a trained FieldFormer model.

    Implements ``query(coords_xy, t_index, station_coords, station_vectors,
    station_mask) -> [n, 2]`` (the ``CoordinateQueryImputer`` protocol). Spatial
    coordinates are min-max normalized to roughly ``[0, 1]`` from the station
    extent (or fixed ``domain_bounds``), and time is normalized by ``T - 1``;
    these are the conventions a wind checkpoint must be trained under.
    """

    model: Any
    time_radius: int = 3
    k_neighbors: int = 32
    domain_bounds: tuple[float, float, float, float] | None = None
    device: str = "cpu"
    name: str = "fieldformer_coordinate_query"

    def __post_init__(self) -> None:
        out_dim = int(getattr(self.model, "out_dim", -1))
        if out_dim != 2:
            raise ValueError(
                f"FieldFormer wind imputer requires a model with out_dim=2 (Ux,Vy); got out_dim={out_dim}. "
                "Train a 2-vector wind checkpoint (see baselines/fieldformer/README.md)."
            )
        self.model.to(self.device)
        self.model.eval()

    def _normalize_xy(self, coords: np.ndarray, stations: np.ndarray) -> np.ndarray:
        if self.domain_bounds is not None:
            xmin, xmax, ymin, ymax = self.domain_bounds
        else:
            xmin, ymin = float(stations[:, 0].min()), float(stations[:, 1].min())
            xmax, ymax = float(stations[:, 0].max()), float(stations[:, 1].max())
        sx = max(xmax - xmin, 1e-9)
        sy = max(ymax - ymin, 1e-9)
        out = np.empty_like(coords, dtype=np.float64)
        out[:, 0] = (coords[:, 0] - xmin) / sx
        out[:, 1] = (coords[:, 1] - ymin) / sy
        return out

    def query(
        self,
        coords_xy: np.ndarray,
        t_index: int,
        station_coords: np.ndarray,
        station_vectors: np.ndarray,
        station_mask: np.ndarray,
    ) -> np.ndarray:
        stations = np.asarray(station_coords, dtype=np.float64)
        vectors = np.asarray(station_vectors, dtype=np.float32)  # [S, T, 2]
        mask = np.asarray(station_mask, dtype=bool)  # [S, T]
        coords = np.asarray(coords_xy, dtype=np.float64)
        S, T = vectors.shape[0], vectors.shape[1]
        if vectors.shape != (S, T, 2):
            raise ValueError("station_vectors must have shape [S, T, 2] for wind (Ux, Vy)")
        if not mask[:, int(t_index)].any() and not mask.any():
            raise ValueError("no observed station vectors available for FieldFormer query")

        device = self.device
        stations_norm = self._normalize_xy(stations, stations)
        coords_norm = self._normalize_xy(coords, stations)
        t_grid_np = np.arange(T, dtype=np.float64) / max(T - 1, 1)

        sensors_xy = torch.as_tensor(stations_norm, dtype=torch.float32, device=device)
        t_grid = torch.as_tensor(t_grid_np, dtype=torch.float32, device=device)

        # Observation tuples in s-major order (lin = s*T + k), matching the indexer.
        sx = np.repeat(stations_norm[:, 0], T)
        sy = np.repeat(stations_norm[:, 1], T)
        st = np.tile(t_grid_np, S)
        obs_coords = torch.as_tensor(np.stack([sx, sy, st], axis=1), dtype=torch.float32, device=device)
        obs_vals = torch.as_tensor(vectors.reshape(S * T, 2), dtype=torch.float32, device=device)
        allowed = np.flatnonzero(mask.reshape(-1))
        allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=device)

        indexer = SplitAwareSparseNeighborIndexer(
            sensors_xy, t_grid, time_radius=self.time_radius, k_neighbors=self.k_neighbors,
            allowed_indices=allowed_t,
        )

        t_q = float(t_grid_np[int(t_index)])
        xyt_q = torch.as_tensor(
            np.column_stack([coords_norm[:, 0], coords_norm[:, 1], np.full(coords.shape[0], t_q)]),
            dtype=torch.float32, device=device,
        )
        nb_idx = indexer.gather_continuous_neighbors(xyt_q)
        with torch.no_grad():
            out = self.model.forward_continuous(xyt_q, obs_coords, obs_vals, nb_idx)
        out = out.reshape(coords.shape[0], 2)
        result = out.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("FieldFormer produced non-finite wind vectors")
        return result


def build_fieldformer_wind_imputer(
    checkpoint_path: str | None = None,
    *,
    model: Any = None,
    d_model: int = 128,
    nhead: int = 4,
    layers: int = 3,
    d_ff: int = 256,
    time_radius: int = 3,
    k_neighbors: int = 32,
    domain_bounds: tuple[float, float, float, float] | None = None,
    device: str = "cpu",
    use_ema: bool = True,
) -> FieldFormerCoordinateQueryImputer:
    """Build a FieldFormer wind imputer from a checkpoint or an explicit model.

    Provide ``checkpoint_path`` to load a trained 2-vector wind checkpoint, or a
    prebuilt ``model`` (e.g. for smoke tests). Exactly one must be given.
    """

    if (checkpoint_path is None) == (model is None):
        raise ValueError("provide exactly one of checkpoint_path or model")
    if checkpoint_path is not None:
        model = load_fieldformer_checkpoint(
            checkpoint_path, d_model=d_model, nhead=nhead, layers=layers, d_ff=d_ff,
            out_dim=2, device=device, use_ema=use_ema,
        )
    return FieldFormerCoordinateQueryImputer(
        model=model, time_radius=time_radius, k_neighbors=k_neighbors,
        domain_bounds=domain_bounds, device=device,
    )


def build_untrained_wind_model(d_model: int = 32, nhead: int = 4, layers: int = 2, d_ff: int = 64) -> FieldFormerCoordinateQuery:
    """Construct an UNTRAINED 2-vector FieldFormer for plumbing/smoke checks only.

    Output is meaningless until a wind checkpoint is trained; use only to exercise
    the coordinate-query plumbing (shapes, protocol conformance), never as a
    scientific wind product.
    """

    return FieldFormerCoordinateQuery(d_model=d_model, nhead=nhead, layers=layers, d_ff=d_ff, out_dim=2)


__all__ = [
    "FieldFormerCoordinateQueryImputer",
    "build_fieldformer_wind_imputer",
    "build_untrained_wind_model",
]
