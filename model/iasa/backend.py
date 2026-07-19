"""Shared PyTorch backend helpers for the IASA numeric core (Task 6A).

PyTorch is the sole computational backend for response, background, projection,
diagnostics, and fitting. NumPy/pandas are permitted only at CSV/NPZ ingestion
and serialization boundaries. This module centralizes device/dtype resolution,
the single sanctioned tensor->NumPy boundary, runtime/data provenance, and
ensemble-kind tagging so downstream modules stay consistent.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


DEFAULT_DEVICE = "cpu"
# Inverse diagnostics/fitting use float64 by default; response construction may
# use float32 or float64.
INVERSE_DTYPE = torch.float64
RESPONSE_DTYPE = torch.float32

ENSEMBLE_KINDS = ("transport", "inventory")

_DTYPE_ALIASES: dict[Any, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "f32": torch.float32,
    "f64": torch.float64,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device(DEFAULT_DEVICE)
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype | None, *, default: torch.dtype = INVERSE_DTYPE) -> torch.dtype:
    if dtype is None:
        return default
    resolved = _DTYPE_ALIASES.get(dtype)
    if resolved is None:
        raise ValueError(f"unsupported dtype {dtype!r}; use 'float32' or 'float64'")
    return resolved


def dtype_name(dtype: torch.dtype) -> str:
    return "float64" if dtype == torch.float64 else "float32"


def as_tensor(
    data: Any,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    default_dtype: torch.dtype = INVERSE_DTYPE,
) -> torch.Tensor:
    """Ingestion-boundary conversion of array-like data to a device tensor."""

    resolved_dtype = resolve_dtype(dtype, default=default_dtype)
    resolved_device = resolve_device(device)
    if isinstance(data, torch.Tensor):
        return data.to(device=resolved_device, dtype=resolved_dtype)
    array = np.asarray(data)
    return torch.as_tensor(array, dtype=resolved_dtype, device=resolved_device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """The only sanctioned tensor->NumPy boundary."""

    if not isinstance(tensor, torch.Tensor):
        return np.asarray(tensor)
    return tensor.detach().cpu().numpy()


def runtime_provenance(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "device": str(device),
        "dtype": dtype_name(dtype),
        "torch_version": str(torch.__version__),
        "cuda_version": (None if torch.version.cuda is None else str(torch.version.cuda)),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def tensor_hash(tensor: torch.Tensor | np.ndarray) -> str:
    """Stable content hash over contiguous CPU float64 bytes for provenance."""

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().to(torch.float64).contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(tensor, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def names_hash(names: Any) -> str:
    digest = hashlib.sha256()
    digest.update(" ".join(str(name) for name in names).encode("utf-8"))
    return digest.hexdigest()


def validate_ensemble_kind(kind: str) -> str:
    if kind not in ENSEMBLE_KINDS:
        raise ValueError(f"ensemble_kind must be one of {ENSEMBLE_KINDS}; got {kind!r}")
    return kind


__all__ = [
    "DEFAULT_DEVICE",
    "ENSEMBLE_KINDS",
    "INVERSE_DTYPE",
    "RESPONSE_DTYPE",
    "as_tensor",
    "dtype_name",
    "names_hash",
    "resolve_device",
    "resolve_dtype",
    "runtime_provenance",
    "tensor_hash",
    "to_numpy",
    "validate_ensemble_kind",
]
