"""Vendored inference-only copy of the coordinate-query FieldFormer model.

Provenance: vendored from the sibling FieldFormer research repository at
``/scratch/ab9738/fieldformer`` (fieldformer_core/scripts/ffag_polsparse_train.py
and fieldformer_core/scripts/sparse_neighbor_indexer.py). Only the inference
pieces are copied -- the transformer model, its coordinate-query forward pass, and
the sparse sensor-time neighbor indexer. Training loops, dataset loaders, and CLI
utilities are intentionally omitted. State-dict parameter names are preserved so
checkpoints trained in the upstream repository load unchanged.

FieldFormer (bhardwaj2025fieldformer) is a coordinate-query model: given sparse
masked station observations ``{(z_i, t, u_i, m_i)}`` it predicts field values at
arbitrary query coordinates, implementing the paper's
``w_hat_t(x_g) = f_omega(x_g, t; stations)`` (paper eq. fieldformer_wind_field).
For IASA wind, the field is a 2-vector ``(Ux, Vy)`` so ``out_dim`` must be 2; the
upstream pollution/heat/SWE checkpoints are scalar (``out_dim=1``), so a 2-vector
wind checkpoint must be trained before this model produces meaningful wind.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.backends.cuda import sdp_kernel


class SplitAwareSparseNeighborIndexer:
    """Sparse sensor-time neighbor indexer with an optional visible-index set.

    If ``allowed_indices`` is supplied, observed and continuous queries can only
    gather neighbors whose linear indices ``lin = s*Nt + k`` are in that set, so
    masked (held-out) observations are never read as context.
    """

    def __init__(
        self,
        sensors_xy: torch.Tensor,
        t_grid: torch.Tensor,
        time_radius: int,
        k_neighbors: int,
        allowed_indices: torch.Tensor | None = None,
    ):
        self.sensors_xy = sensors_xy
        self.t_grid = t_grid
        self.S = sensors_xy.shape[0]
        self.Nt = t_grid.shape[0]
        self.time_radius = int(time_radius)
        self.k_neighbors = int(k_neighbors)
        self.allowed_indices: torch.Tensor | None = None
        self.allowed_mask: torch.Tensor | None = None
        self.fallback_index: torch.Tensor | None = None

        sensor_ids = torch.arange(self.S, dtype=torch.long)
        offsets = torch.arange(-self.time_radius, self.time_radius + 1, dtype=torch.long)
        s_mesh, dt_mesh = torch.meshgrid(sensor_ids, offsets, indexing="ij")
        self.base_sensor = s_mesh.reshape(-1)
        self.base_dt = dt_mesh.reshape(-1)

        if allowed_indices is not None:
            self.set_allowed_indices(allowed_indices)

    def set_allowed_indices(self, allowed_indices: torch.Tensor | None) -> None:
        if allowed_indices is None:
            self.allowed_indices = None
            self.allowed_mask = None
            self.fallback_index = None
            return
        allowed = allowed_indices.detach().long().flatten()
        if allowed.numel() == 0:
            raise ValueError("allowed_indices must contain at least one observation")
        mask = torch.zeros(self.S * self.Nt, dtype=torch.bool, device=allowed.device)
        mask[allowed] = True
        self.allowed_indices = allowed
        self.allowed_mask = mask
        self.fallback_index = allowed[0]

    def lin_to_sk(self, lin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return lin // self.Nt, lin % self.Nt

    def sk_to_lin(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return s * self.Nt + k

    def _filter_and_pad(self, lin_nb: torch.Tensor, lin_q: torch.Tensor | None = None, exclude_self: bool = False) -> torch.Tensor:
        valid = torch.ones_like(lin_nb, dtype=torch.bool)
        if self.allowed_mask is not None:
            valid &= self.allowed_mask.to(lin_nb.device)[lin_nb]
        if exclude_self and lin_q is not None:
            valid &= lin_nb != lin_q[:, None]

        cand_count = lin_nb.shape[1]
        order = torch.arange(cand_count, device=lin_nb.device).unsqueeze(0).expand_as(lin_nb)
        sort_key = torch.where(valid, order, order.new_full(order.shape, cand_count + 1))
        take = torch.argsort(sort_key, dim=1)[:, : min(self.k_neighbors, cand_count)]
        lin_nb = torch.gather(lin_nb, 1, take)
        valid = torch.gather(valid, 1, take)

        has_valid = valid.any(dim=1, keepdim=True)
        first_valid_pos = valid.long().argmax(dim=1, keepdim=True)
        row_fallback = torch.gather(lin_nb, 1, first_valid_pos)
        if self.allowed_indices is not None:
            allowed = self.allowed_indices.to(lin_nb.device)
            primary = allowed[0].expand_as(row_fallback)
            if lin_q is not None and allowed.numel() > 1:
                secondary = allowed[1].expand_as(row_fallback)
                global_fallback = torch.where(primary == lin_q[:, None], secondary, primary)
            else:
                global_fallback = primary
        elif lin_q is not None:
            global_fallback = lin_q[:, None]
        else:
            global_fallback = lin_nb[:, :1]
        row_fallback = torch.where(has_valid, row_fallback, global_fallback)
        lin_nb = torch.where(valid, lin_nb, row_fallback.expand_as(lin_nb))

        if lin_nb.shape[1] < self.k_neighbors:
            pad = lin_nb[:, -1:].expand(-1, self.k_neighbors - lin_nb.shape[1])
            lin_nb = torch.cat([lin_nb, pad], dim=1)
        return lin_nb

    def gather_observed_neighbors(self, lin_q: torch.Tensor, exclude_self: bool = True) -> torch.Tensor:
        _, k_q = self.lin_to_sk(lin_q)
        bsz = lin_q.shape[0]
        s_nb = self.base_sensor.to(lin_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_q[:, None] + self.base_dt.to(lin_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        return self._filter_and_pad(lin_nb, lin_q=lin_q, exclude_self=exclude_self)

    def gather_continuous_neighbors(self, xyt_q: torch.Tensor) -> torch.Tensor:
        t_q = xyt_q[:, 2]
        dist = torch.abs(t_q[:, None] - self.t_grid[None, :].to(xyt_q.device))
        k_hat = torch.argmin(dist, dim=1)
        bsz = xyt_q.shape[0]
        s_nb = self.base_sensor.to(xyt_q.device).unsqueeze(0).expand(bsz, -1)
        k_nb = (k_hat[:, None] + self.base_dt.to(xyt_q.device)[None, :]).clamp_(0, self.Nt - 1)
        lin_nb = self.sk_to_lin(s_nb, k_nb)
        return self._filter_and_pad(lin_nb)


class FieldFormerCoordinateQuery(nn.Module):
    """Coordinate-query FieldFormer (upstream ``FieldFormerSparsePollution``).

    ``forward_continuous(xyt_q, obs_coords, obs_vals, nb_idx)`` predicts field
    values at arbitrary query coordinates ``xyt_q`` from sparse observations.
    ``out_dim`` is the field vector dimension (1 for scalar upstream checkpoints,
    2 for IASA wind ``(Ux, Vy)``).
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 3, d_ff: int = 256, out_dim: int = 1):
        super().__init__()
        self.out_dim = int(out_dim)
        self.log_gammas = nn.Parameter(torch.zeros(3))
        self.input_proj = nn.Linear(3 + self.out_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.out_dim),
        )

    def _forward_tokens(self, xyt_q: torch.Tensor, nb_xyt: torch.Tensor, nb_vals: torch.Tensor) -> torch.Tensor:
        rel = nb_xyt - xyt_q[:, None, :]
        rel = rel * torch.exp(self.log_gammas)[None, None, :]
        if nb_vals.ndim == 2:
            nb_vals = nb_vals[..., None]
        mu = nb_vals.mean(dim=1, keepdim=True)
        sigma = nb_vals.std(dim=1, keepdim=True).clamp_min(1e-3)
        nb_vals_norm = (nb_vals - mu) / sigma
        tokens = torch.cat([rel, nb_vals_norm], dim=-1)

        kernel_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else nullcontext()
        with kernel_ctx, amp_ctx:
            h = self.input_proj(tokens)
            h = self.encoder(h)
            u_std_res = self.head(h.mean(dim=1))
        out = u_std_res * sigma.squeeze(1) + mu.squeeze(1)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def forward_observed(self, q_lin: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(obs_coords[q_lin], obs_coords[nb_idx], obs_vals[nb_idx])

    def forward_continuous(self, xyt_q: torch.Tensor, obs_coords: torch.Tensor, obs_vals: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
        return self._forward_tokens(xyt_q, obs_coords[nb_idx], obs_vals[nb_idx])


# Upstream alias so pre-existing references / documentation resolve.
FieldFormerSparsePollution = FieldFormerCoordinateQuery


def load_fieldformer_checkpoint(
    path: str,
    *,
    d_model: int = 128,
    nhead: int = 4,
    layers: int = 3,
    d_ff: int = 256,
    out_dim: int = 2,
    device: str = "cpu",
    use_ema: bool = True,
    strict: bool = True,
) -> FieldFormerCoordinateQuery:
    """Load a trained coordinate-query FieldFormer checkpoint for inference.

    Mirrors the upstream ``maybe_load_checkpoint`` loading contract: accepts a raw
    state-dict or a dict containing ``model_state_dict`` (and optionally
    ``ema_model_state_dict``, preferred when ``use_ema``). ``out_dim`` defaults to
    2 for wind ``(Ux, Vy)``.
    """

    model = FieldFormerCoordinateQuery(d_model=d_model, nhead=nhead, layers=layers, d_ff=d_ff, out_dim=out_dim)
    ckpt = torch.load(path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        if use_ema and "ema_model_state_dict" in ckpt and ckpt["ema_model_state_dict"]:
            state = ckpt["ema_model_state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "FieldFormerCoordinateQuery",
    "FieldFormerSparsePollution",
    "SplitAwareSparseNeighborIndexer",
    "load_fieldformer_checkpoint",
]
