#!/usr/bin/env python3
"""Fit the official MV3 market-context decoder on the public 9x4 corpus.

This reuses Probe Atlas's balanced, causal corpus and embedding cache.  The
encoder is frozen.  The saved decoder exposes the four exact MV-v3 states and
is cryptographically bound to the selected encoder checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.market_context import sha256
from futures_foundation.market_context_training import (
    fit_market_context_decoder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "mantis_ssl_mv_v3.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--labels",
        type=Path,
        default=(
            ROOT
            / "temp"
            / "clean_ssl_pre2026_lora"
            / "probe_atlas"
            / "trend_lifecycle_labels_pre2026.npz"
        ),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=ROOT / "temp" / "market_context_mv3_embeddings.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints" / "mantis_ssl_mv_v3_context.npz",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--fit-end", default="2023-01-01")
    parser.add_argument("--calibration-start", default="2023-01-01")
    parser.add_argument("--calibration-end", default="2024-01-01")
    parser.add_argument("--evaluation-start", default="2025-01-01")
    parser.add_argument("--evaluation-end", default="2026-01-01")
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"market-context encoder not found: {args.checkpoint}")
    if not args.labels.is_file():
        raise FileNotFoundError(
            f"market-context corpus labels not found: {args.labels}")
    if args.batch_size <= 0:
        raise ValueError("market-context batch size must be positive")

    # Probe Atlas reads its contract from the environment at import time.
    os.environ.update({
        "FFM_ROOT": str(ROOT),
        "DATA_DIR": str(args.data_dir),
        "CKPT_NAME": args.checkpoint.name,
        "CKPT_PATH": str(args.checkpoint),
        "CKPT_SHA256": sha256(args.checkpoint),
        "EMB_CACHE": str(args.embedding_cache),
        "TREND_LABELS": str(args.labels),
        "ATLAS_BATCH": str(args.batch_size),
    })
    if args.device is not None:
        os.environ["DEVICE"] = args.device
    import probe_atlas

    bars_by_stream, keys, fields = probe_atlas._load_pool()
    embeddings = probe_atlas._embeddings(bars_by_stream, keys)
    valid = fields["mv_state"] >= 0
    decoder, report = fit_market_context_decoder(
        embeddings[valid],
        fields["mv_state"][valid],
        fields["timestamp"][valid],
        fields["stream"][valid],
        encoder_sha256=sha256(args.checkpoint),
        fit_end=args.fit_end,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        seed=args.seed,
    )
    report.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "decoder": str(args.output),
        "embedding_cache": str(args.embedding_cache),
        "corpus": str(args.labels),
        "corpus_pool_sha256": probe_atlas._pool_sha256(keys),
        "scope": "9_tickers_x_4_timeframes",
        "strategy_inputs": False,
        "causal_window_end": "decision_candle",
        "future_label_start": "decision_candle_plus_1",
    })
    decoder.save(args.output)
    report_path = (
        args.report
        if args.report is not None
        else args.output.with_suffix(".report.json")
    )
    _atomic_json(report_path, report)
    print(f"[market-context] decoder -> {args.output}", flush=True)
    print(f"[market-context] report  -> {report_path}", flush=True)
    for name, row in report["evaluation"]["classes"].items():
        print(
            f"  {name:>18}: AUC={row['auc']:.4f} "
            f"shuffle={row['shuffle_auc']:.4f} "
            f"lift={row['real_minus_shuffle_auc']:+.4f} "
            f"worst={row['worst_stream_auc']}",
            flush=True,
        )
    print(f"[market-context] PASS={report['pass']}", flush=True)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
