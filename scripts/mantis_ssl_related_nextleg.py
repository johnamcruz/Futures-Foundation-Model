#!/usr/bin/env python3
"""Run the opt-in Mantis related-series NextLeg experiment locally.

The primary stream owns every target. Same-ticker timeframes and a same-timeframe sibling are
closed-bar-aligned context only. Use a distinct output path; this produces a composite
``mantis-related-v1`` checkpoint rather than replacing the incumbent plain Mantis checkpoint.
"""
from __future__ import annotations

import argparse
import os

from futures_foundation.finetune import ssl


def _csv(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(description="Mantis causal related-series NextLeg")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--warm-ckpt", default=os.environ.get("BACKBONE_CKPT"), required=False)
    parser.add_argument("--out", default=os.environ.get(
        "OUT_PATH", "temp/related_nextleg/mantis_related_nextleg.pt"))
    parser.add_argument("--tickers", default=os.environ.get("TICKERS", "NQ,ES"))
    parser.add_argument("--tfs", default=os.environ.get("TFS", "1min,3min,5min,15min"))
    parser.add_argument("--related-tfs", default=os.environ.get(
        "RELATED_TFS", "1min,3min,5min,15min"))
    parser.add_argument(
        "--tf-pairs", default=os.environ.get("RELATED_TF_PAIRS", "1min=5min,3min=15min"),
        help=("paired timeframe contexts; '=' is bidirectional, ':' directional, '+' multiple. "
              "Example: 1min=5min,3min=15min"))
    parser.add_argument(
        "--siblings", default=os.environ.get("RELATED_SIBLINGS", "0"),
        help="cross-instrument context; disabled by default to isolate timeframe-pair value")
    parser.add_argument("--heads", type=int, default=int(os.environ.get("RELATED_HEADS", "4")))
    parser.add_argument("--dropout", type=float,
                        default=float(os.environ.get("RELATED_DROPOUT", "0")))
    parser.add_argument("--max-gap-factor", type=float,
                        default=float(os.environ.get("RELATED_MAX_GAP_FACTOR", "2")))
    parser.add_argument("--related-control", choices=("real", "shuffle", "random", "drop"),
                        default=os.environ.get("RELATED_CONTROL", "real"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "60")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "200")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "128")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    parser.add_argument("--device", default=os.environ.get("DEVICE"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--holdout-start", default=os.environ.get("HOLDOUT_START", "2026-01-01"))
    parser.add_argument("--controls", default=os.environ.get("CONTROLS", "shuffle,random"))
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    if not args.warm_ckpt:
        parser.error("--warm-ckpt (or BACKBONE_CKPT) is required; related fusion refines NextLeg")
    if not os.path.isfile(args.warm_ckpt):
        parser.error(f"warm checkpoint not found: {args.warm_ckpt}")
    if os.path.abspath(args.warm_ckpt) == os.path.abspath(args.out):
        parser.error("--out must differ from --warm-ckpt; never overwrite the incumbent")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[related-nextleg] primary targets; causal contexts only", flush=True)
    print(f"[related-nextleg] tfs={args.related_tfs} pairs={args.tf_pairs} "
          f"siblings={args.siblings} "
          f"heads={args.heads} output={args.out}", flush=True)
    ssl.loop_ssl(
        data_dir=args.data_dir, tickers=_csv(args.tickers), tfs=_csv(args.tfs),
        controls=_csv(args.controls), out_path=args.out, probe=not args.no_probe,
        holdout_start=args.holdout_start, pretext="related_nextleg",
        backbone_ckpt=args.warm_ckpt, related_tfs=_csv(args.related_tfs),
        related_tf_pairs=args.tf_pairs,
        related_siblings=args.siblings, related_heads=args.heads,
        related_dropout=args.dropout, related_max_gap_factor=args.max_gap_factor,
        related_control=args.related_control,
        epochs=args.epochs, steps_per_epoch=args.steps, batch=args.batch, lr=args.lr,
        device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
