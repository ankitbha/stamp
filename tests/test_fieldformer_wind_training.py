from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_fieldformer_wind as tfw  # noqa: E402
from baselines.fieldformer.model import load_fieldformer_checkpoint  # noqa: E402
from model.iasa.fieldformer_adapter import FieldFormerCoordinateQueryImputer  # noqa: E402
from model.iasa.wind import build_gridded_wind_field  # noqa: E402


def synth_stations(S=6, T=8, nx=12, ny=12, seed=0):
    """A small smooth station field so a tiny model has signal to fit."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(1.0, nx - 2.0, size=S)
    ys = rng.uniform(1.0, ny - 2.0, size=S)
    grid_xy = np.stack([xs, ys], axis=1).astype(np.float64)
    vectors = np.zeros((S, T, 2), dtype=np.float32)
    for s in range(S):
        for k in range(T):
            # Smooth spatial + temporal transport field.
            vectors[s, k, 0] = 1.0 + 0.1 * xs[s] + 0.05 * k
            vectors[s, k, 1] = 0.2 * ys[s] - 0.05 * k
    mask = np.ones((S, T), dtype=bool)
    mask[0, 0] = False  # a couple of genuine gaps
    mask[S - 1, T - 1] = False
    return grid_xy, vectors, mask


def tiny_cfg(ckpt_dir, **kw):
    base = dict(
        d_model=16, nhead=2, layers=1, d_ff=32, k_neighbors=8, time_radius=2,
        epochs=2, batch_size=64, lr=1e-3, patience=5, seed=0,
        checkpoint_dir=str(ckpt_dir), run_name="test_wind", device="cpu",
    )
    base.update(kw)
    return tfw.WindTrainConfig(**base)


class FieldFormerWindTrainingTests(unittest.TestCase):
    def test_smoke_train_save_load_drive(self) -> None:
        grid_xy, vectors, mask = synth_stations()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp)
            sup = tfw.build_supervision(
                grid_xy, vectors, mask, holdout_indices=[1], k_neighbors=cfg.k_neighbors,
                time_radius=cfg.time_radius, device="cpu",
            )
            report = tfw.train_from_supervision(sup, cfg, provenance={"test": True})
            self.assertTrue(Path(report["best_checkpoint"]).exists())
            self.assertGreaterEqual(report["epochs_run"], 1)
            self.assertTrue(np.isfinite(report["best_val_rmse"]))

            # Load as a 2-vector checkpoint and build the imputer.
            model = load_fieldformer_checkpoint(
                report["best_checkpoint"], d_model=cfg.d_model, nhead=cfg.nhead,
                layers=cfg.layers, d_ff=cfg.d_ff, out_dim=2, device="cpu", use_ema=True,
            )
            self.assertEqual(int(model.out_dim), 2)
            imputer = FieldFormerCoordinateQueryImputer(
                model=model, time_radius=cfg.time_radius, k_neighbors=cfg.k_neighbors, device="cpu",
            )
            # Drive the gridded wind field end to end.
            timestamps = np.datetime64("2026-06-01T00:00") + np.arange(vectors.shape[1]) * np.timedelta64(1, "h")
            gwf = build_gridded_wind_field(
                grid_xy, vectors, mask, timestamps, (6, 6), imputer=imputer,
            )
            self.assertEqual(gwf.physical_field.shape, (vectors.shape[1], 6, 6, 2))
            self.assertTrue(np.isfinite(gwf.physical_field).all())

    def test_holdout_excluded_from_neighbors(self) -> None:
        grid_xy, vectors, mask = synth_stations()
        S, T = vectors.shape[0], vectors.shape[1]
        hold = [2, 4]
        sup = tfw.build_supervision(
            grid_xy, vectors, mask, holdout_indices=hold, k_neighbors=8, time_radius=2, device="cpu",
        )
        allowed_mask = sup["indexer"].allowed_mask
        # Every held-out station's linear indices must be disallowed as neighbors.
        for s in hold:
            for k in range(T):
                self.assertFalse(bool(allowed_mask[s * T + k]), f"held-out lin {s}*{T}+{k} leaked")
        # Train tuples are s-major lin=s*T+k and only for non-holdout observed cells.
        train_set = set(int(v) for v in sup["train_lins"].tolist())
        for lin in train_set:
            self.assertNotIn(lin // T, hold)
            self.assertTrue(mask[lin // T, lin % T])
        # Val tuples are exactly the observed held-out cells.
        val_set = set(int(v) for v in sup["val_lins"].tolist())
        expected_val = {s * T + k for s in hold for k in range(T) if mask[s, k]}
        self.assertEqual(val_set, expected_val)

    def test_checkpoint_roundtrip_strict(self) -> None:
        grid_xy, vectors, mask = synth_stations()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp)
            sup = tfw.build_supervision(
                grid_xy, vectors, mask, holdout_indices=[1], k_neighbors=cfg.k_neighbors,
                time_radius=cfg.time_radius, device="cpu",
            )
            report = tfw.train_from_supervision(sup, cfg)
            ckpt = torch.load(report["best_checkpoint"], map_location="cpu")
            for key in ("model_state_dict", "ema_model_state_dict", "config", "epoch", "best_val_rmse", "provenance"):
                self.assertIn(key, ckpt)
            fresh = tfw.FieldFormerCoordinateQuery(cfg.d_model, cfg.nhead, cfg.layers, cfg.d_ff, out_dim=2)
            fresh.load_state_dict(ckpt["ema_model_state_dict"], strict=True)  # must not raise

    def test_early_stopping(self) -> None:
        stop = tfw.EarlyStopping(patience=3)
        stop.step(1.0)
        self.assertFalse(stop.stopped)
        for _ in range(3):  # three non-improving epochs
            stop.step(1.0)
        self.assertTrue(stop.stopped)

    def test_determinism(self) -> None:
        grid_xy, vectors, mask = synth_stations()
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            sup1 = tfw.build_supervision(grid_xy, vectors, mask, holdout_indices=[1], k_neighbors=8, time_radius=2, device="cpu")
            sup2 = tfw.build_supervision(grid_xy, vectors, mask, holdout_indices=[1], k_neighbors=8, time_radius=2, device="cpu")
            r1 = tfw.train_from_supervision(sup1, tiny_cfg(tmp1))
            r2 = tfw.train_from_supervision(sup2, tiny_cfg(tmp2))
            self.assertAlmostEqual(r1["best_val_rmse"], r2["best_val_rmse"], places=5)

    def test_out_dim_guard(self) -> None:
        # The adapter must reject a scalar model.
        scalar = tfw.FieldFormerCoordinateQuery(8, 2, 1, 16, out_dim=1)
        with self.assertRaises(ValueError):
            FieldFormerCoordinateQueryImputer(model=scalar, device="cpu")


if __name__ == "__main__":
    unittest.main()
