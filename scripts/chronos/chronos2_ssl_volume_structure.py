#!/usr/bin/env python3
"""Causal Chronos-2 Volume-Structure SSL adapter refinement.

This stage is deliberately self-supervised and uses completed OHLCV only.  It
combines masked volume/price patch reconstruction with participation,
concentration/dispersion, displacement-volume, and temporal-order objectives.
The temporary objective heads are discarded; the output is a LoRA adapter
whose frozen embeddings can be probed for causal OHLCV volume structure. Bars
cannot reconstruct a literal intrabar exchange volume profile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.finetune.classifiers.chronos2.multivariate import (
    prepare_multivariate,
)
from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
    preflight_volume_structure_ssl,
    train_volume_structure_ssl,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument(
        "--parent", type=Path,
        default=ROOT / "temp/chronos2_small_36stream/mask_full/checkpoint")
    value.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "temp/chronos2_small_36stream/volume_structure_ssl")
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--context-length", type=int, default=256)
    value.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    value.add_argument("--mask-ratio", type=float, default=0.25)
    value.add_argument("--projection-dim", type=int, default=128)
    value.add_argument("--temperature", type=float, default=0.10)
    value.add_argument("--noise", type=float, default=0.02)
    value.add_argument("--scale", type=float, default=0.10)
    value.add_argument("--epochs", type=int, default=60)
    value.add_argument("--steps", type=int, default=100)
    value.add_argument("--batch-windows", type=int, default=32)
    value.add_argument("--gradient-accumulation", type=int, default=1)
    value.add_argument("--lr", type=float, default=1e-5)
    value.add_argument("--weight-decay", type=float, default=0.05)
    value.add_argument("--patience", type=int, default=8)
    value.add_argument("--threshold-samples", type=int, default=4096)
    value.add_argument("--validation-windows-per-stream", type=int, default=16)
    value.add_argument("--price-bins", type=int, default=16)
    value.add_argument("--reconstruction-weight", type=float, default=1.0)
    value.add_argument("--participation-weight", type=float, default=1.0)
    value.add_argument("--concentration-weight", type=float, default=1.0)
    value.add_argument("--displacement-weight", type=float, default=1.0)
    value.add_argument("--temporal-weight", type=float, default=0.5)
    value.add_argument("--adapter-retention-weight", type=float, default=0.1)
    value.add_argument("--log-every-steps", type=int, default=10)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.parent.is_dir():
        raise SystemExit(f"parent adapter is missing: {args.parent}")
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
            prediction_length=1,
            report_path=args.out_dir / f"data_preflight_{timeframe}.json",
        )
        for timeframe in timeframes
    }
    for timeframe, item in prepared.items():
        print(
            f"[chronos2-volume-structure:{timeframe}] "
            f"loaded OHLCV matrix {item.train.shape} "
            f"finite={item.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.preflight_only:
        report = preflight_volume_structure_ssl(
            prepared,
            context_length=args.context_length,
            threshold_samples=args.threshold_samples,
            validation_windows_per_stream=args.validation_windows_per_stream,
            price_bins=args.price_bins,
        )
        path = args.out_dir / "volume_structure_preflight_only.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        print(
            f"[chronos2-volume-structure] PREFLIGHT PASS report={path}",
            flush=True,
        )
        return
    report = train_volume_structure_ssl(
        prepared,
        parent=args.parent,
        out_dir=args.out_dir,
        device=args.device,
        context_length=args.context_length,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_windows=args.batch_windows,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        noise=args.noise,
        scale=args.scale,
        mask_ratio=args.mask_ratio,
        threshold_samples=args.threshold_samples,
        validation_windows_per_stream=args.validation_windows_per_stream,
        price_bins=args.price_bins,
        reconstruction_weight=args.reconstruction_weight,
        participation_weight=args.participation_weight,
        concentration_weight=args.concentration_weight,
        displacement_weight=args.displacement_weight,
        temporal_weight=args.temporal_weight,
        adapter_retention_weight=args.adapter_retention_weight,
        log_every_steps=args.log_every_steps,
        seed=args.seed,
        resume=args.resume,
    )
    print(
        f"[chronos2-volume-structure] PASS "
        f"best_val={report['best_val_loss']:.6f} "
        f"checkpoint={report['checkpoint']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
