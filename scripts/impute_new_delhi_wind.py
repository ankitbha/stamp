#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pol_weather import (  # noqa: E402
    ImputeFormerWindConfig,
    assert_imputed_product_complete,
    impute_wind_with_checkpoint,
    load_new_delhi_wind_data,
    save_imputed_wind_product,
    train_wind_imputeformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImputeFormer and save imputed New Delhi wind products.")
    parser.add_argument("--data-csv", type=Path, default=REPO_ROOT / "sim" / "govdata_1H_current.csv")
    parser.add_argument("--locations-csv", type=Path, default=REPO_ROOT / "sim" / "govdata_locations.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "new_delhi_wind_imputed.npz")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "imputeformer_wind_best.pt")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-train", action="store_true", help="Load an existing checkpoint and only run imputation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ImputeFormerWindConfig(
        windows=args.windows,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_rate=args.mask_rate,
        patience=args.patience,
        seed=args.seed,
    )
    wind = load_new_delhi_wind_data(args.data_csv, args.locations_csv, start=args.start, end=args.end)
    print(
        "[load] stations={} timestamps={} first={} last={}".format(
            len(wind.station_ids),
            len(wind.timestamps),
            wind.timestamps[0].isoformat(),
            wind.timestamps[-1].isoformat(),
        )
    )
    print(
        "[mask] raw WD valid={} WS valid={} vector valid={}".format(
            int(wind.raw_WD_mask.sum()),
            int(wind.raw_WS_mask.sum()),
            int(wind.vector_mask[..., 0].sum()),
        )
    )

    if args.skip_train:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"--skip-train requested but checkpoint does not exist: {args.checkpoint}")
    else:
        info = train_wind_imputeformer(wind, args.checkpoint, cfg, device=args.device)
        print(f"[train] best_val_rmse={info['best_val_rmse']:.6f} checkpoint={info['checkpoint_path']}")

    imputed = impute_wind_with_checkpoint(wind, args.checkpoint, cfg=None, device=args.device)
    save_imputed_wind_product(
        wind,
        imputed,
        args.output,
        imputation_config={
            **asdict(cfg),
            "checkpoint": str(args.checkpoint),
            "checkpoint_config_source": "checkpoint",
            "start": args.start,
            "end": args.end,
            "skip_train": bool(args.skip_train),
        },
    )
    assert_imputed_product_complete(
        args.output,
        expected_station_count=len(wind.station_ids),
        expected_timestamp_count=len(wind.timestamps),
    )
    print(f"[save] {args.output}")
    print(f"[done] Vx/Vy length={imputed['Vx'].shape[0]}")


if __name__ == "__main__":
    main()
