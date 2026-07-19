# Vendored FieldFormer (coordinate-query, inference-only)

This directory vendors the inference pieces of the coordinate-query FieldFormer
model used as the IASA wind imputer (paper eq. `fieldformer_wind_field`,
`w_hat_t(x_g) = f_omega(x_g, t; stations)`).

## Provenance

Copied from the sibling FieldFormer research repository at
`/scratch/ab9738/fieldformer`:

- `fieldformer_core/scripts/ffag_polsparse_train.py` → `FieldFormerCoordinateQuery`
  (upstream `FieldFormerSparsePollution`).
- `fieldformer_core/scripts/sparse_neighbor_indexer.py` →
  `SplitAwareSparseNeighborIndexer`.

Only inference is vendored (model + neighbor indexer + checkpoint loader).
Training loops, dataset loaders, and CLI utilities are intentionally omitted.
State-dict parameter names are preserved, so upstream checkpoints load unchanged.

## Status: activation requires a trained 2-vector wind checkpoint

FieldFormer is a coordinate-query model; for IASA **wind** the field is a
2-vector `(Ux, Vy)`, so `out_dim = 2`. The upstream trained checkpoints
(`ffag_polsparse_best.pt`, `ffag_heatsparse_best.pt`, `ffag_swesparse_best.pt`)
are **scalar** (`out_dim = 1`) pollution/heat/SWE fields — there is **no wind
checkpoint**. Until a 2-vector `(Ux, Vy)` coordinate-query checkpoint is trained
on the government wind data, the FieldFormer path produces untrained output.

Accordingly, the **default** IASA wind imputer remains the kernel
coordinate-query interpolator (`KernelCoordinateQueryImputer`); FieldFormer is
opt-in via `model/iasa/fieldformer_adapter.py`
(`FieldFormerCoordinateQueryImputer`) once a checkpoint exists.

## How to activate

1. Train a coordinate-query FieldFormer with `out_dim=2` on the gov wind
   transport vectors (upstream training script, adapted for the 2-vector target).
2. `from model.iasa.fieldformer_adapter import build_fieldformer_wind_imputer`
   and pass the resulting imputer to `build_gridded_wind_field(..., imputer=...)`.

## Caveat for training/inference: neighbor selection is not spatially local

`SplitAwareSparseNeighborIndexer._filter_and_pad` keeps the first `k_neighbors`
valid candidates in **s-major enumeration order (lowest sensor index first), not
by spatial distance**. This is the upstream FieldFormer design, vendored
faithfully. When the candidate set (`S * (2*time_radius + 1)` sensor-time tuples)
exceeds `k_neighbors`, only the lowest-index sensors feed the transformer and
spatial selectivity relies entirely on the learned `log_gammas`. For New Delhi
(~32 stations) the adapter default `k_neighbors=32` with `time_radius=3` yields
224 candidates, so raise `k_neighbors` to cover the station-time window (or
accept the learned-attention regime) — and **train the wind checkpoint under the
same `k_neighbors`/`time_radius` the adapter queries with**, since the two must
match for the learned attention to behave as trained.
