#!/usr/bin/env python3
"""Chronos-2 masked reconstruction on joint and single-ticker OHLCV tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.classifiers.chronos2.multivariate import (
    prepare_multivariate,
)
from futures_foundation.finetune.classifiers.chronos2.ssl_stages import train_mask


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument(
        "--parent", type=Path,
        default=ROOT / "temp/chronos2_small_3min/trainer/finetuned-ckpt")
    value.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "temp/chronos2_small_3min/mask")
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--context-length", type=int, default=256)
    value.add_argument("--prediction-length", type=int, default=32)
    value.add_argument("--timeframes", default="3min")
    value.add_argument("--mask-ratio", type=float, default=0.40)
    value.add_argument("--epochs", type=int, default=2)
    value.add_argument("--steps", type=int, default=10)
    value.add_argument("--batch-windows", type=int, default=1)
    value.add_argument("--gradient-accumulation", type=int, default=8)
    value.add_argument("--lr", type=float, default=5e-5)
    value.add_argument("--patience", type=int, default=3)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--single-ticker-probability", type=float, default=0.25)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.parent.is_dir():
        raise SystemExit(f"parent adapter is missing: {args.parent}")
    timeframes = tuple(
        value.strip() for value in args.timeframes.split(",") if value.strip())
    if not timeframes or len(set(timeframes)) != len(timeframes):
        raise SystemExit("timeframes must be a non-empty unique list")
    if set(timeframes) - set(TIMEFRAMES):
        raise SystemExit(f"unsupported timeframes: {sorted(set(timeframes) - set(TIMEFRAMES))}")
    prepared = {
        timeframe: prepare_multivariate(
            args.data_dir,
            repo_root=ROOT,
            tickers=TICKERS,
            timeframe=timeframe,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            report_path=args.out_dir / f"data_preflight_{timeframe}.json",
        )
        for timeframe in timeframes
    }
    for timeframe, item in prepared.items():
        print(
            f"[chronos2-mask:{timeframe}] loaded joint matrix {item.train.shape} "
            f"finite={item.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.preflight_only:
        return
    report = train_mask(
        prepared,
        parent=args.parent,
        out_dir=args.out_dir,
        device=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_windows=args.batch_windows,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        single_ticker_probability=args.single_ticker_probability,
        resume=args.resume,
    )
    print(
        f"[chronos2-mask] PASS best_val={report['best_val_loss']:.6f} "
        f"checkpoint={report['checkpoint']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
