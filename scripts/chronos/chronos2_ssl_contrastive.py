#!/usr/bin/env python3
"""Temporal contrastive refinement across Chronos-2 OHLCV task groups."""
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
from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
    train_contrastive,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument(
        "--parent", type=Path,
        default=ROOT / "temp/chronos2_small_3min/mask/checkpoint")
    value.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "temp/chronos2_small_3min/contrastive")
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--context-length", type=int, default=256)
    value.add_argument("--prediction-length", type=int, default=32)
    value.add_argument("--timeframes", default="3min")
    value.add_argument("--pos-deltas", default="2,16,64")
    value.add_argument("--far-min", type=int, default=512)
    value.add_argument("--temperature", type=float, default=0.10)
    value.add_argument("--noise", type=float, default=0.02)
    value.add_argument("--scale", type=float, default=0.10)
    value.add_argument("--time-mask", type=float, default=0.0)
    value.add_argument("--projection-dim", type=int, default=128)
    value.add_argument(
        "--regime-key", choices=("kaufman", "temporal"), default="kaufman")
    value.add_argument("--kaufman-chop", type=float, default=0.25)
    value.add_argument("--kaufman-trend", type=float, default=0.50)
    value.add_argument("--kaufman-length", type=int, default=64)
    value.add_argument("--epochs", type=int, default=2)
    value.add_argument("--steps", type=int, default=4)
    value.add_argument("--batch-windows", type=int, default=16)
    value.add_argument("--gradient-accumulation", type=int, default=2)
    value.add_argument("--lr", type=float, default=2e-5)
    value.add_argument("--patience", type=int, default=3)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--single-ticker-probability", type=float, default=0.25)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.parent.is_dir():
        raise SystemExit(f"mask parent adapter is missing: {args.parent}")
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
            f"[chronos2-contrastive:{timeframe}] loaded joint matrix {item.train.shape} "
            f"finite={item.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.preflight_only:
        return
    deltas = tuple(int(value) for value in args.pos_deltas.split(",") if value)
    report = train_contrastive(
        prepared,
        parent=args.parent,
        out_dir=args.out_dir,
        device=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        pos_deltas=deltas,
        far_min=args.far_min,
        temperature=args.temperature,
        noise=args.noise,
        scale=args.scale,
        time_mask=args.time_mask,
        projection_dim=args.projection_dim,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_windows=args.batch_windows,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        regime_key=args.regime_key,
        kaufman_chop=args.kaufman_chop,
        kaufman_trend=args.kaufman_trend,
        kaufman_length=args.kaufman_length,
        single_ticker_probability=args.single_ticker_probability,
        resume=args.resume,
    )
    print(
        f"[chronos2-contrastive] PASS best_val={report['best_val_loss']:.6f} "
        f"checkpoint={report['checkpoint']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
