#!/usr/bin/env python3
"""Causal balanced-Kaufman refinement of the promoted Volume SSL adapter.

Kaufman mode is the production path.  It authenticates the completed
Volume-Structure v3 parent and balances causal chop against efficient trend
windows. Trend direction is deliberately ignored: up and down are one
direction-agnostic state. Promotion uses only native Chronos REG geometry
after temporary projection heads have been discarded. ``temporal`` remains
available solely as the historical contrastive path and does not receive the
Kaufman promotion gates.
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
    preflight_balanced_kaufman_ssl,
    train_balanced_kaufman_ssl,
    train_contrastive,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
VOLUME_V3_DIR = (
    ROOT / "temp/chronos2_small_36stream/volume_structure_ssl_v3_seed0")
VOLUME_V3_CHECKPOINT = VOLUME_V3_DIR / "checkpoint"
VOLUME_V3_REPORT = VOLUME_V3_DIR / "report.json"
BALANCED_KAUFMAN_OUT = (
    ROOT / "temp/chronos2_small_36stream/contrastive_balanced_kaufman_v1")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument(
        "--parent", type=Path,
        default=VOLUME_V3_CHECKPOINT,
        help=(
            "canonical completed Volume-Structure v3 LoRA checkpoint; "
            "Kaufman mode refuses any other parent"
        ),
    )
    value.add_argument(
        "--parent-report", type=Path, default=VOLUME_V3_REPORT,
        help="completion report authenticating the Volume-Structure parent",
    )
    value.add_argument(
        "--base-snapshot",
        type=Path,
        default=None,
        help=(
            "pinned Chronos2-small Hugging Face snapshot directory; "
            "mandatory in Kaufman mode"
        ),
    )
    value.add_argument(
        "--out-dir", type=Path,
        default=BALANCED_KAUFMAN_OUT)
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--context-length", type=int, default=256)
    value.add_argument("--prediction-length", type=int, default=32)
    value.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    value.add_argument("--pos-deltas", default="2,16,64")
    value.add_argument("--far-min", type=int, default=512)
    value.add_argument("--temperature", type=float, default=0.10)
    value.add_argument("--noise", type=float, default=0.02)
    value.add_argument("--scale", type=float, default=0.10)
    value.add_argument("--time-mask", type=float, default=0.0)
    value.add_argument("--projection-dim", type=int, default=128)
    value.add_argument("--head-auxiliary-weight", type=float, default=0.25)
    value.add_argument(
        "--regime-key", choices=("kaufman", "temporal"), default="kaufman")
    value.add_argument(
        "--kaufman-chop", type=float, default=0.25,
        help="fixed ER upper bound for the direction-agnostic chop state",
    )
    value.add_argument(
        "--kaufman-trend", type=float, default=0.50,
        help="fixed ER lower bound for efficient trend, combining up and down",
    )
    value.add_argument(
        "--kaufman-length", type=int, default=64,
        help="fixed completed-bar lookback used by the Kaufman ER teacher",
    )
    value.add_argument("--epochs", type=int, default=60)
    value.add_argument("--steps", type=int, default=100)
    value.add_argument("--batch-windows", type=int, default=32)
    value.add_argument("--gradient-accumulation", type=int, default=1)
    value.add_argument("--lr", type=float, default=5e-6)
    value.add_argument("--weight-decay", type=float, default=0.05)
    value.add_argument("--patience", type=int, default=8)
    value.add_argument("--adapter-retention-weight", type=float, default=0.1)
    value.add_argument("--native-promotion-margin", type=float, default=1e-4)
    value.add_argument("--validation-windows-per-state", type=int, default=16)
    value.add_argument("--log-every-steps", type=int, default=10)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--single-ticker-probability", type=float, default=0.25)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def _write_preflight(path: Path, report: dict) -> None:
    """Publish one structured preflight report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parser().parse_args()
    if not args.parent.is_dir():
        raise SystemExit(f"parent adapter is missing: {args.parent}")
    if args.regime_key == "kaufman":
        if args.base_snapshot is None or not args.base_snapshot.is_dir():
            raise SystemExit(
                "Kaufman mode requires --base-snapshot pointing to the "
                "pinned Chronos2-small snapshot directory"
            )
        if args.parent.resolve() != VOLUME_V3_CHECKPOINT.resolve():
            raise SystemExit(
                "Kaufman mode is pinned to the canonical completed "
                f"Volume-Structure v3 checkpoint: {VOLUME_V3_CHECKPOINT}"
            )
        if args.parent_report.resolve() != VOLUME_V3_REPORT.resolve():
            raise SystemExit(
                "Kaufman mode is pinned to the canonical Volume-Structure "
                f"v3 completion report: {VOLUME_V3_REPORT}"
            )
        if not args.parent_report.is_file():
            raise SystemExit(
                f"Volume-Structure completion report is missing: {args.parent_report}")
        fixed_kaufman = (
            args.kaufman_chop == 0.25
            and args.kaufman_trend == 0.50
            and args.kaufman_length == 64
        )
        if not fixed_kaufman:
            raise SystemExit(
                "balanced Kaufman mode fixes ER length=64, chop<=0.25, "
                "and direction-agnostic efficient trend>=0.50"
            )
    else:
        print(
            "[chronos2-contrastive] LEGACY temporal mode selected; "
            "balanced Kaufman sampling, native checkpoint promotion, and "
            "Volume-parent authentication are disabled",
            flush=True,
        )
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
            # Kaufman authentication belongs to the balanced stage. Do not
            # mutate its output directory before the Volume report and pinned
            # base have passed that gate. Legacy temporal mode retains its
            # historical per-timeframe reports.
            report_path=(
                None if args.regime_key == "kaufman"
                else args.out_dir / f"data_preflight_{timeframe}.json"
            ),
        )
        for timeframe in timeframes
    }
    for timeframe, item in prepared.items():
        print(
            f"[chronos2-contrastive:{timeframe}] loaded joint matrix {item.train.shape} "
            f"finite={item.report['finite_fraction']:.4f}",
            flush=True,
        )
    if args.regime_key == "kaufman":
        if args.preflight_only:
            report = preflight_balanced_kaufman_ssl(
                prepared,
                parent=args.parent,
                parent_report=args.parent_report,
                base_snapshot=args.base_snapshot,
                context_length=args.context_length,
                kaufman_length=args.kaufman_length,
                kaufman_chop=args.kaufman_chop,
                kaufman_trend=args.kaufman_trend,
                validation_windows_per_state=(
                    args.validation_windows_per_state),
            )
            path = args.out_dir / "balanced_kaufman_preflight_only.json"
            _write_preflight(path, report)
            print(
                f"[chronos2-balanced-kaufman] PREFLIGHT PASS report={path}",
                flush=True,
            )
            return
        report = train_balanced_kaufman_ssl(
            prepared,
            parent=args.parent,
            parent_report=args.parent_report,
            base_snapshot=args.base_snapshot,
            out_dir=args.out_dir,
            device=args.device,
            context_length=args.context_length,
            kaufman_length=args.kaufman_length,
            kaufman_chop=args.kaufman_chop,
            kaufman_trend=args.kaufman_trend,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            batch_windows=args.batch_windows,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            projection_dim=args.projection_dim,
            head_auxiliary_weight=args.head_auxiliary_weight,
            temperature=args.temperature,
            noise=args.noise,
            scale=args.scale,
            validation_windows_per_state=args.validation_windows_per_state,
            adapter_retention_weight=args.adapter_retention_weight,
            native_promotion_margin=args.native_promotion_margin,
            log_every_steps=args.log_every_steps,
            seed=args.seed,
            resume=args.resume,
        )
        print(
            f"[chronos2-balanced-kaufman] PASS "
            f"best_val={report['best_val_loss']:.6f} "
            f"checkpoint={report['checkpoint']['path']}",
            flush=True,
        )
        return

    if args.preflight_only:
        report = {
            "schema": "ffm_chronos2_temporal_contrastive_preflight_v1",
            "status": "pass",
            "mode": "legacy_temporal",
            "parent": str(args.parent.resolve()),
            "base_snapshot": (
                None if args.base_snapshot is None
                else str(args.base_snapshot.resolve())
            ),
            "timeframes": list(timeframes),
            "data_contracts": {
                timeframe: item.report
                for timeframe, item in prepared.items()
            },
            "limitations": [
                "legacy temporal mode has no balanced Kaufman native gate",
                "legacy temporal mode does not authenticate a Volume parent report",
            ],
        }
        path = args.out_dir / "temporal_preflight_only.json"
        _write_preflight(path, report)
        print(
            f"[chronos2-contrastive] LEGACY PREFLIGHT PASS report={path}",
            flush=True,
        )
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
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        regime_key="temporal",
        single_ticker_probability=args.single_ticker_probability,
        resume=args.resume,
    )
    print(
        f"[chronos2-contrastive] LEGACY PASS "
        f"best_val={report['best_val_loss']:.6f} "
        f"checkpoint={report['checkpoint']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
