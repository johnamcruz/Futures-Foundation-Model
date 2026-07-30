#!/usr/bin/env python3
"""Self-supervised Chronos-2 compression/stable/expansion refinement."""
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
    train_volatility_contrastive,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument(
        "--parent", type=Path,
        default=(
            ROOT
            / "temp/chronos2_small_36stream"
            / "contrastive_kaufman_full/checkpoint"))
    value.add_argument(
        "--out-dir", type=Path,
        default=(
            ROOT
            / "temp/chronos2_small_36stream/volatility_contrastive"))
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--context-length", type=int, default=256)
    value.add_argument("--prediction-length", type=int, default=32)
    value.add_argument("--timeframes", default="1min,3min,5min,15min")
    value.add_argument("--dynamics-length", type=int, default=64)
    value.add_argument("--lower-quantile", type=float, default=0.25)
    value.add_argument("--upper-quantile", type=float, default=0.75)
    value.add_argument("--threshold-samples", type=int, default=50_000)
    value.add_argument("--kaufman-retention-weight", type=float, default=1.0)
    value.add_argument("--kaufman-chop", type=float, default=0.25)
    value.add_argument("--kaufman-trend", type=float, default=0.50)
    value.add_argument("--kaufman-length", type=int, default=64)
    value.add_argument("--temperature", type=float, default=0.10)
    value.add_argument("--noise", type=float, default=0.02)
    value.add_argument("--scale", type=float, default=0.10)
    value.add_argument("--time-mask", type=float, default=0.0)
    value.add_argument("--projection-dim", type=int, default=128)
    value.add_argument("--epochs", type=int, default=20)
    value.add_argument("--steps", type=int, default=100)
    value.add_argument("--batch-windows", type=int, default=64)
    value.add_argument("--gradient-accumulation", type=int, default=1)
    value.add_argument("--lr", type=float, default=2e-5)
    value.add_argument("--patience", type=int, default=5)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.parent.is_dir():
        raise SystemExit(f"masked parent adapter is missing: {args.parent}")
    timeframes = tuple(
        item.strip() for item in args.timeframes.split(",") if item.strip())
    if not timeframes or len(set(timeframes)) != len(timeframes):
        raise SystemExit("timeframes must be a non-empty unique list")
    unsupported = set(timeframes) - set(TIMEFRAMES)
    if unsupported:
        raise SystemExit(f"unsupported timeframes: {sorted(unsupported)}")
    prepared = {
        timeframe: prepare_multivariate(
            args.data_dir,
            repo_root=ROOT,
            tickers=TICKERS,
            timeframe=timeframe,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            report_path=(
                args.out_dir / f"data_preflight_{timeframe}.json"),
        )
        for timeframe in timeframes
    }
    for timeframe, item in prepared.items():
        print(
            f"[chronos2-volatility:{timeframe}] "
            f"loaded joint matrix {item.train.shape} "
            f"finite={item.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.preflight_only:
        return
    report = train_volatility_contrastive(
        prepared,
        parent=args.parent,
        out_dir=args.out_dir,
        device=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
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
        volatility_length=args.dynamics_length,
        volatility_lower_quantile=args.lower_quantile,
        volatility_upper_quantile=args.upper_quantile,
        volatility_threshold_samples=args.threshold_samples,
        kaufman_retention_weight=args.kaufman_retention_weight,
        kaufman_chop=args.kaufman_chop,
        kaufman_trend=args.kaufman_trend,
        kaufman_length=args.kaufman_length,
        resume=args.resume,
    )
    print(
        f"[chronos2-volatility] PASS "
        f"best_val={report['best_val_loss']:.6f} "
        f"checkpoint={report['checkpoint']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
