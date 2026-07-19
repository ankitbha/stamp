"""Identifiability-aware source apportionment primitives."""

from model.iasa.backend import (
    ENSEMBLE_KINDS,
    INVERSE_DTYPE,
    RESPONSE_DTYPE,
    runtime_provenance,
    tensor_hash,
    to_numpy,
    validate_ensemble_kind,
)
from model.iasa.background import BackgroundBasisConfig, BackgroundBasisResult, build_background_basis
from model.iasa.projection import (
    BackgroundProjector,
    ProjectionConfig,
    ProjectionResult,
    fit_background_projector,
    project_response_and_observations,
)

from model.iasa.response import (
    BOUNDARY_MODE,
    RESPONSE_IMPLEMENTATION,
    CityWindSampler,
    DispersionConfig,
    Observer,
    ResponseConfig,
    ResponseMatrixResult,
    WindSampler,
    build_lagged_response_matrix,
)

__all__ = [
    "BOUNDARY_MODE",
    "ENSEMBLE_KINDS",
    "INVERSE_DTYPE",
    "RESPONSE_DTYPE",
    "RESPONSE_IMPLEMENTATION",
    "runtime_provenance",
    "tensor_hash",
    "to_numpy",
    "validate_ensemble_kind",
    "BackgroundBasisConfig",
    "BackgroundBasisResult",
    "BackgroundProjector",
    "CityWindSampler",
    "DispersionConfig",
    "Observer",
    "ProjectionConfig",
    "ProjectionResult",
    "ResponseConfig",
    "ResponseMatrixResult",
    "WindSampler",
    "build_background_basis",
    "build_lagged_response_matrix",
    "fit_background_projector",
    "project_response_and_observations",
]
