"""Vendored inference-only FieldFormer coordinate-query model for IASA wind.

See ``model.py`` for provenance. A 2-vector (Ux, Vy) wind checkpoint must be
trained before this model produces meaningful wind; the default IASA wind imputer
remains the kernel coordinate-query interpolator.
"""

from baselines.fieldformer.model import (
    FieldFormerCoordinateQuery,
    FieldFormerSparsePollution,
    SplitAwareSparseNeighborIndexer,
    load_fieldformer_checkpoint,
)

__all__ = [
    "FieldFormerCoordinateQuery",
    "FieldFormerSparsePollution",
    "SplitAwareSparseNeighborIndexer",
    "load_fieldformer_checkpoint",
]
