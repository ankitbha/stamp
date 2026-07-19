from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.backend import (  # noqa: E402
    ENSEMBLE_KINDS,
    as_tensor,
    dtype_name,
    names_hash,
    resolve_device,
    resolve_dtype,
    runtime_provenance,
    tensor_hash,
    to_numpy,
    validate_ensemble_kind,
)


class BackendTests(unittest.TestCase):
    def test_device_and_dtype_resolution(self) -> None:
        self.assertEqual(resolve_device(None).type, "cpu")
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertEqual(resolve_dtype(None), torch.float64)
        self.assertEqual(resolve_dtype("float32"), torch.float32)
        self.assertEqual(resolve_dtype("float64"), torch.float64)
        self.assertEqual(resolve_dtype(None, default=torch.float32), torch.float32)
        self.assertEqual(dtype_name(torch.float64), "float64")
        self.assertEqual(dtype_name(torch.float32), "float32")
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            resolve_dtype("float16")

    def test_as_tensor_and_to_numpy_roundtrip(self) -> None:
        array = np.arange(6, dtype=np.float64).reshape(2, 3)
        tensor = as_tensor(array, device="cpu", dtype="float32")
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, "cpu")
        back = to_numpy(tensor)
        self.assertIsInstance(back, np.ndarray)
        np.testing.assert_allclose(back, array.astype(np.float32))

    def test_runtime_provenance_fields(self) -> None:
        prov = runtime_provenance(resolve_device("cpu"), torch.float64)
        for key in ("device", "dtype", "torch_version", "cuda_version", "cuda_available"):
            self.assertIn(key, prov)
        self.assertEqual(prov["device"], "cpu")
        self.assertEqual(prov["dtype"], "float64")
        self.assertIsInstance(prov["cuda_available"], bool)

    def test_hashes_are_stable_and_content_sensitive(self) -> None:
        a = np.arange(9, dtype=np.float64).reshape(3, 3)
        b = a.copy()
        c = a + 1.0
        self.assertEqual(tensor_hash(a), tensor_hash(b))
        self.assertEqual(tensor_hash(a), tensor_hash(torch.as_tensor(a)))
        self.assertNotEqual(tensor_hash(a), tensor_hash(c))
        self.assertEqual(names_hash(["x", "y"]), names_hash(["x", "y"]))
        self.assertNotEqual(names_hash(["x", "y"]), names_hash(["y", "x"]))

    def test_validate_ensemble_kind(self) -> None:
        for kind in ENSEMBLE_KINDS:
            self.assertEqual(validate_ensemble_kind(kind), kind)
        with self.assertRaisesRegex(ValueError, "ensemble_kind"):
            validate_ensemble_kind("mixed")


if __name__ == "__main__":
    unittest.main()
