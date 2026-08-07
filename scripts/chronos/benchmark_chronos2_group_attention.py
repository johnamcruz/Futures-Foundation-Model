#!/usr/bin/env python3
"""PROTOTYPE benchmark for optional split Chronos-2 group attention."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _windows(path: Path, *, count: int, context: int) -> np.ndarray:
    frame = pd.read_csv(
        path,
        usecols=["open", "high", "low", "close", "volume"],
    )
    bars = frame[["open", "high", "low", "close", "volume"]].to_numpy(
        np.float32)
    endpoints = np.arange(len(bars) - count, len(bars), dtype=np.int64)
    offsets = np.arange(context - 1, -1, -1, dtype=np.int64)
    rows = endpoints[:, None] - offsets[None, :]
    return np.transpose(bars[rows], (0, 2, 1))


def _child(args) -> int:
    from futures_foundation.finetune.classifiers.chronos2._embed_worker_fast import (
        embed_window_chunks,
    )

    values = np.load(args.input, allow_pickle=False)
    chunks = tuple(np.array_split(values, args.chunks))
    if args.mode == "fast":
        os.environ["FFM_CHRONOS2_FAST_GROUP_ATTENTION"] = "1"
    started = perf_counter()
    output = np.concatenate(list(embed_window_chunks(
        chunks,
        checkpoint=args.checkpoint,
        device=args.device,
        batch=args.batch_windows * values.shape[1],
        pool="reg",
        context_length=values.shape[-1],
    )))
    elapsed = perf_counter() - started
    np.save(args.output, output)
    Path(args.timing).write_text(json.dumps({"seconds": elapsed}) + "\n")
    return 0


def _compare(args) -> int:
    windows = _windows(args.bars, count=args.windows, context=args.context)
    with tempfile.TemporaryDirectory(prefix="ffm-fast-group-attention-") as raw:
        directory = Path(raw)
        source = directory / "windows.npy"
        np.save(source, windows)
        results = {}
        for mode in ("legacy", "fast"):
            output = directory / f"{mode}.npy"
            timing = directory / f"{mode}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode", mode,
                    "--input", str(source),
                    "--output", str(output),
                    "--timing", str(timing),
                    "--checkpoint", str(args.checkpoint),
                    "--device", args.device,
                    "--batch-windows", str(args.batch_windows),
                    "--chunks", str(args.chunks),
                ],
                check=True,
            )
            results[mode] = {
                "values": np.load(output, allow_pickle=False),
                "seconds": json.loads(timing.read_text())["seconds"],
            }
        legacy = results["legacy"]
        fast = results["fast"]
        report = {
            "schema": "chronos2-fast-group-attention-prototype-v1",
            "device": args.device,
            "windows": args.windows,
            "chunks": args.chunks,
            "batch_windows": args.batch_windows,
            "float32_exact": bool(np.array_equal(
                legacy["values"], fast["values"])),
            "float16_exact": bool(np.array_equal(
                legacy["values"].astype(np.float16),
                fast["values"].astype(np.float16),
            )),
            "max_abs_delta": float(np.max(np.abs(
                legacy["values"] - fast["values"]))),
            "legacy_seconds": legacy["seconds"],
            "fast_seconds": fast["seconds"],
            "speedup": legacy["seconds"] / fast["seconds"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["float16_exact"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "legacy", "fast"), default="compare")
    parser.add_argument("--bars", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        if args.bars is None:
            parser.error("--bars is required in compare mode")
        return _compare(args)
    return _child(args)


if __name__ == "__main__":
    raise SystemExit(main())
