from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.iasa.wind import (  # noqa: E402
    ar1_synthetic,
    constant_direction,
    diurnal_synthetic,
    multi_direction_synthetic,
    real_new_delhi_wind_sequence,
    single_direction_synthetic,
)
from data.pol_weather import load_new_delhi_wind_data, save_imputed_wind_product, transport_uv_to_wind  # noqa: E402


SIM_DIR = REPO_ROOT / "sim"


class IasaWindTests(unittest.TestCase):
    def test_real_new_delhi_provider_reads_imputed_product(self) -> None:
        wind = load_new_delhi_wind_data(
            SIM_DIR / "govdata_1H_current.csv",
            SIM_DIR / "govdata_locations.csv",
            start="2018-05-01 00:00:00+05:30",
            end="2018-05-01 23:00:00+05:30",
        )
        imputed_vectors = wind.observed_vectors.copy()
        imputed_wd, imputed_ws = transport_uv_to_wind(imputed_vectors[..., 0], imputed_vectors[..., 1])
        city = imputed_vectors.mean(axis=0)
        imputed = {
            "imputed_U_x": imputed_vectors[..., 0],
            "imputed_V_y": imputed_vectors[..., 1],
            "imputed_WD": imputed_wd,
            "imputed_WS": imputed_ws,
            "Vx": city[:, 0].astype(np.float32),
            "Vy": city[:, 1].astype(np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            product_path = Path(tmp) / "wind_product.npz"
            save_imputed_wind_product(wind, imputed, product_path, {"test": True})
            seq = real_new_delhi_wind_sequence(
                SIM_DIR / "govdata_1H_current.csv",
                SIM_DIR / "govdata_locations.csv",
                imputed_product_path=product_path,
                start="2018-05-01 00:00:00+05:30",
                end="2018-05-01 23:00:00+05:30",
            )

        self.assertEqual(seq.provider, "real_imputed_new_delhi")
        self.assertEqual(seq.vx.shape, (24,))
        self.assertEqual(seq.vy.shape, (24,))
        np.testing.assert_allclose(seq.vx, city[:, 0], atol=1e-6)
        np.testing.assert_allclose(seq.vy, city[:, 1], atol=1e-6)
        self.assertEqual(seq.metadata["aggregation"], "imputed_product_city_level")

    def test_real_new_delhi_provider_observed_fallback_is_explicitly_labeled(self) -> None:
        seq = real_new_delhi_wind_sequence(
            SIM_DIR / "govdata_1H_current.csv",
            SIM_DIR / "govdata_locations.csv",
            imputed_product_path=SIM_DIR / "missing_imputed_product.npz",
            start="2018-05-01 00:00:00+05:30",
            end="2018-05-01 23:00:00+05:30",
            allow_observed_fallback=True,
        )
        self.assertEqual(seq.provider, "real_observed_new_delhi")
        self.assertEqual(seq.vx.shape, (24,))
        self.assertEqual(seq.vy.shape, (24,))
        self.assertTrue(np.isfinite(seq.vx).all())
        self.assertTrue(np.isfinite(seq.vy).all())
        self.assertEqual(seq.metadata["aggregation"], "observed_station_mean_city_level")

    def test_constant_and_single_direction_shapes(self) -> None:
        constant = constant_direction(length=5, vx=2.0, vy=-1.0)
        self.assertEqual(constant.provider, "constant_direction")
        np.testing.assert_allclose(constant.vx, np.full(5, 2.0, dtype=np.float32))
        np.testing.assert_allclose(constant.vy, np.full(5, -1.0, dtype=np.float32))

        single = single_direction_synthetic(length=4, speed=1.0, direction_degrees=0.0)
        self.assertEqual(single.provider, "single_direction_synthetic")
        np.testing.assert_allclose(single.vx, np.ones(4, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(single.vy, np.zeros(4, dtype=np.float32), atol=1e-6)

    def test_diurnal_provider_shape_and_metadata(self) -> None:
        seq = diurnal_synthetic(length=24, base_vx=1.0, base_vy=0.5, amplitude=0.25)
        self.assertEqual(seq.provider, "diurnal_synthetic")
        self.assertEqual(seq.vx.shape, (24,))
        self.assertEqual(seq.metadata["amplitude"], 0.25)
        self.assertGreater(float(seq.vx.max() - seq.vx.min()), 0.0)

    def test_seeded_synthetic_providers_are_reproducible(self) -> None:
        ar1_a = ar1_synthetic(length=12, seed=11)
        ar1_b = ar1_synthetic(length=12, seed=11)
        np.testing.assert_allclose(ar1_a.vx, ar1_b.vx)
        np.testing.assert_allclose(ar1_a.vy, ar1_b.vy)

        multi_a = multi_direction_synthetic(length=12, seed=7)
        multi_b = multi_direction_synthetic(length=12, seed=7)
        np.testing.assert_allclose(multi_a.vx, multi_b.vx)
        np.testing.assert_allclose(multi_a.vy, multi_b.vy)
        self.assertEqual(multi_a.provider, "multi_direction_synthetic")


if __name__ == "__main__":
    unittest.main()
