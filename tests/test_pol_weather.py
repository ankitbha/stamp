from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pol_weather import (  # noqa: E402
    ImputeFormerWindConfig,
    WindData,
    _make_model,
    assert_imputed_product_complete,
    impute_wind_with_checkpoint,
    load_new_delhi_wind_data,
    save_imputed_wind_product,
    train_wind_imputeformer,
    transport_uv_to_wind,
    wind_to_transport_uv,
)


def _synthetic_wind(all_masks_valid: bool = True) -> WindData:
    timestamps = pd.date_range("2018-05-01 00:00:00+05:30", periods=4, freq="h")
    raw_WD = np.asarray([[0.0, np.nan, 90.0, 180.0], [270.0, 0.0, np.nan, 180.0]], dtype=np.float32)
    raw_WS = np.asarray([[1.0, np.nan, 2.0, 1.0], [1.5, 1.0, np.nan, 2.0]], dtype=np.float32)
    raw_pm25 = np.asarray([[25.0, 26.0, np.nan, 27.0], [30.0, np.nan, 31.0, 32.0]], dtype=np.float32)
    raw_WD_mask = np.isfinite(raw_WD)
    raw_WS_mask = np.isfinite(raw_WS)
    raw_pm25_mask = np.isfinite(raw_pm25)
    if not all_masks_valid:
        raw_WD_mask[:] = False
        raw_WS_mask[:] = False
        raw_pm25_mask[:] = False
    ux, vy = wind_to_transport_uv(raw_WD, raw_WS)
    vector_mask_2d = raw_WD_mask & raw_WS_mask & np.isfinite(ux) & np.isfinite(vy)
    observed_vectors = np.stack([np.nan_to_num(ux, nan=0.0), np.nan_to_num(vy, nan=0.0)], axis=-1).astype(np.float32)
    vector_mask = np.repeat(vector_mask_2d[..., None], 2, axis=-1)
    return WindData(
        station_ids=np.asarray(["station_a", "station_b"], dtype="<U64"),
        timestamps=timestamps,
        sensors_xy=np.asarray([[77.1, 28.6], [77.2, 28.7]], dtype=np.float32),
        location_names=np.asarray(["A", "B"], dtype="<U256"),
        raw_WD=raw_WD,
        raw_WS=raw_WS,
        raw_pm25=raw_pm25,
        raw_WD_mask=raw_WD_mask,
        raw_WS_mask=raw_WS_mask,
        raw_pm25_mask=raw_pm25_mask,
        observed_vectors=observed_vectors,
        vector_mask=vector_mask,
        merged_sensors={},
        source_columns=("WD", "WS", "pm25"),
        source_data_csv="sim/govdata_1H_current.csv",
        source_locations_csv="sim/govdata_locations.csv",
    )


def _synthetic_imputed(wind: WindData) -> dict[str, np.ndarray]:
    fill = np.asarray([0.25, -0.5], dtype=np.float32)
    imputed_vectors = np.where(wind.vector_mask, wind.observed_vectors, fill).astype(np.float32)
    imputed_WD, imputed_WS = transport_uv_to_wind(imputed_vectors[..., 0], imputed_vectors[..., 1])
    city = imputed_vectors.mean(axis=0)
    return {
        "imputed_U_x": imputed_vectors[..., 0],
        "imputed_V_y": imputed_vectors[..., 1],
        "imputed_WD": imputed_WD,
        "imputed_WS": imputed_WS,
        "Vx": city[:, 0].astype(np.float32),
        "Vy": city[:, 1].astype(np.float32),
    }


def _small_cfg(**overrides: object) -> ImputeFormerWindConfig:
    values = dict(
        windows=3,
        window_stride=1,
        batch_size=1,
        val_batch_size=1,
        epochs=1,
        mask_rate=0.5,
        input_embedding_dim=4,
        learnable_embedding_dim=4,
        num_layers=1,
        num_temporal_heads=2,
        dim_proj=2,
        dropout=0.0,
        patience=1,
        train_frac=0.5,
        val_frac=0.25,
    )
    values.update(overrides)
    return ImputeFormerWindConfig(**values)


class WindConversionTests(unittest.TestCase):
    def test_cardinal_conversion_and_roundtrip(self) -> None:
        wd = np.asarray([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
        ws = np.ones(4, dtype=np.float32)
        ux, vy = wind_to_transport_uv(wd, ws)
        np.testing.assert_allclose(ux, np.asarray([0.0, -1.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(vy, np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)

        wd_in = np.asarray([0.0, 45.0, 90.0, 180.0, 270.0, 315.0], dtype=np.float32)
        ws_in = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        ux, vy = wind_to_transport_uv(wd_in, ws_in)
        wd_out, ws_out = transport_uv_to_wind(ux, vy)
        circular_diff = np.abs(((wd_out - wd_in + 180.0) % 360.0) - 180.0)
        np.testing.assert_allclose(circular_diff, np.zeros_like(wd_in), atol=1e-4)
        np.testing.assert_allclose(ws_out, ws_in, atol=1e-5)


class WindLoaderTests(unittest.TestCase):
    def test_short_window_loader_preserves_new_delhi_alignment(self) -> None:
        wind = load_new_delhi_wind_data(
            REPO_ROOT / "sim" / "govdata_1H_current.csv",
            REPO_ROOT / "sim" / "govdata_locations.csv",
            start="2018-05-01 00:00:00+05:30",
            end="2018-05-01 23:00:00+05:30",
        )
        self.assertEqual(len(wind.station_ids), 32)
        self.assertEqual(len(wind.timestamps), 24)
        self.assertTrue(wind.timestamps.is_monotonic_increasing)
        self.assertEqual(wind.raw_WD.shape, (32, 24))
        self.assertEqual(wind.raw_WS.shape, (32, 24))
        self.assertEqual(wind.raw_pm25.shape, (32, 24))
        self.assertEqual(wind.observed_vectors.shape, (32, 24, 2))
        self.assertEqual(wind.vector_mask.dtype, np.bool_)
        self.assertIn("Pusa_averaged", wind.merged_sensors)


class WindProductValidationTests(unittest.TestCase):
    def _write_payload(self, directory: Path, name: str, payload: dict[str, np.ndarray]) -> Path:
        path = directory / name
        np.savez_compressed(path, **payload)
        return path

    def test_product_validator_accepts_schema_and_rejects_bad_products(self) -> None:
        wind = _synthetic_wind()
        imputed = _synthetic_imputed(wind)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            product_path = tmp_path / "wind_product.npz"
            save_imputed_wind_product(wind, imputed, product_path, {"test": True})
            assert_imputed_product_complete(product_path, expected_station_count=2, expected_timestamp_count=4)

            with np.load(product_path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            self.assertTrue(np.array_equal(payload["raw_WD_missing_mask"], ~payload["raw_WD_mask"]))
            self.assertEqual(str(payload["mask_convention"][0]).count("valid/observed"), 1)
            json.loads(str(payload["imputation_config"][0]))

            missing_payload = dict(payload)
            del missing_payload["mask_convention"]
            missing_path = self._write_payload(tmp_path, "missing_key.npz", missing_payload)
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                assert_imputed_product_complete(missing_path)

            shape_payload = dict(payload)
            shape_payload["Vx"] = shape_payload["Vx"][:-1]
            shape_path = self._write_payload(tmp_path, "bad_shape.npz", shape_payload)
            with self.assertRaisesRegex(ValueError, "Vx must have shape"):
                assert_imputed_product_complete(shape_path)

            finite_payload = dict(payload)
            bad_ws = finite_payload["imputed_WS"].copy()
            bad_ws[0, 0] = np.nan
            finite_payload["imputed_WS"] = bad_ws
            finite_path = self._write_payload(tmp_path, "bad_finite.npz", finite_payload)
            with self.assertRaisesRegex(ValueError, "non-finite values"):
                assert_imputed_product_complete(finite_path)

            polarity_payload = dict(payload)
            bad_missing = polarity_payload["raw_WD_missing_mask"].copy()
            bad_missing[0, 0] = ~bad_missing[0, 0]
            polarity_payload["raw_WD_missing_mask"] = bad_missing
            polarity_path = self._write_payload(tmp_path, "bad_polarity.npz", polarity_payload)
            with self.assertRaisesRegex(ValueError, "logical inverse"):
                assert_imputed_product_complete(polarity_path)


class CheckpointAndTrainingGuardTests(unittest.TestCase):
    def _write_checkpoint(self, path: Path, wind: WindData, cfg: ImputeFormerWindConfig) -> None:
        model = _make_model(len(wind.station_ids), cfg.windows, 2, cfg, torch.device("cpu"))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "meta": {
                    "variant": "imputeformer_wind",
                    "val_mean": [0.0, 0.0],
                    "val_std": [1.0, 1.0],
                    "normalizes_values": True,
                    "channel_names": ["U_x", "V_y"],
                    "best_val_rmse": 1.0,
                    "windows": cfg.windows,
                    "num_nodes": len(wind.station_ids),
                },
            },
            path,
        )

    def test_checkpoint_config_is_authoritative_for_imputation(self) -> None:
        wind = _synthetic_wind()
        cfg = _small_cfg(input_embedding_dim=6, learnable_embedding_dim=6)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            self._write_checkpoint(checkpoint_path, wind, cfg)
            imputed = impute_wind_with_checkpoint(wind, checkpoint_path, cfg=None, device="cpu")
            self.assertEqual(imputed["imputed_WD"].shape, (2, 4))
            self.assertEqual(imputed["Vx"].shape, (4,))

            bad_cfg = replace(cfg, input_embedding_dim=4)
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                impute_wind_with_checkpoint(wind, checkpoint_path, cfg=bad_cfg, device="cpu")

    def test_training_rejects_invalid_configs_and_missing_observations(self) -> None:
        wind = _synthetic_wind()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            with self.assertRaisesRegex(ValueError, "epochs >= 1"):
                train_wind_imputeformer(wind, checkpoint_path, _small_cfg(epochs=0), device="cpu")
            with self.assertRaisesRegex(ValueError, "0 < mask_rate"):
                train_wind_imputeformer(wind, checkpoint_path, _small_cfg(mask_rate=0.0), device="cpu")
            with self.assertRaisesRegex(ValueError, "positive batch sizes"):
                train_wind_imputeformer(wind, checkpoint_path, _small_cfg(batch_size=0), device="cpu")

            no_observed_wind = _synthetic_wind(all_masks_valid=False)
            with self.assertRaisesRegex(ValueError, "No observed wind vectors"):
                train_wind_imputeformer(no_observed_wind, checkpoint_path, _small_cfg(), device="cpu")


if __name__ == "__main__":
    unittest.main()
