#!/usr/bin/env python3
"""Generate the fixed pre-holdout trend-lifecycle corpus used by Probe Atlas."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("FFM_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from futures_foundation.pipeline._primitives import compute_atr  # noqa: E402
from futures_foundation.primitives.detection import detect_fractal_zigzag_pivots  # noqa: E402
from trend_lifecycle import label_trend_lifecycle  # noqa: E402


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUTPUT = Path(os.environ.get(
    "TREND_LABELS", ROOT / "temp" / "trend_lifecycle_labels.npz"))
LABEL_END = os.environ.get("TREND_LABEL_END", "2026-01-01").strip()
SWING = {None: 0, "HH": 1, "HL": 2, "LH": 3, "LL": 4}
KIND = {None: 0, "start": 1, "continue": 2, "end": 3, "start_end": 4}


def _trend_magnitude(labels, atr):
    """Attach target-side maturity, extent, and resolution to lifecycle groups."""
    groups = []
    current = []
    for index, record in enumerate(labels):
        if record["role_kind"] == "start":
            if current:
                groups.append((current, True))
            current = [index]
        elif record["role_kind"] == "continue" and current:
            current.append(index)
    if current:
        groups.append((current, False))
    for record in labels:
        record.update(leg_idx=0, run_len=0, extent_atr=0.0, resolved=0)
    for members, resolved in groups:
        start = labels[members[0]]
        scale = atr[start["confirm"]]
        scale = scale if np.isfinite(scale) and scale > 0 else np.nan
        prices = np.asarray([labels[index]["px"] for index in members])
        extent = (float(np.nanmax(np.abs(prices - start["px"])) / scale)
                  if np.isfinite(scale) else 0.0)
        for leg, index in enumerate(members, 1):
            labels[index].update(
                leg_idx=leg, run_len=len(members), extent_atr=extent,
                resolved=int(resolved))
    return labels


def main() -> None:
    columns = {name: [] for name in (
        "ticker", "tf", "origin", "confirm", "ts", "direction", "role",
        "swing", "trend_dir", "is_start", "ended", "kind", "px",
        "leg_idx", "run_len", "extent_atr", "resolved")}
    for ticker in TICKERS:
        for timeframe in TIMEFRAMES:
            frame = pd.read_csv(
                DATA_DIR / f"{ticker}_{timeframe}.csv",
                usecols=["datetime", "open", "high", "low", "close", "volume"])
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
            if LABEL_END:
                frame = frame[frame["datetime"] < pd.Timestamp(LABEL_END, tz="UTC")]
            frame = frame.reset_index(drop=True)
            timestamps = frame["datetime"].dt.tz_localize(None).to_numpy()
            o, h, l, c = (frame[name].to_numpy(float)
                          for name in ("open", "high", "low", "close"))
            pivots = detect_fractal_zigzag_pivots(
                o, h, l, c, k=2, min_leg_atr=1.25, atr_period=20)
            labels = _trend_magnitude(
                label_trend_lifecycle(h, l, pivots), compute_atr(h, l, c, 20))
            counts = Counter(record["kind"] for record in labels)
            print(f"[{ticker}@{timeframe}] pivots={len(labels):6d} "
                  f"start={counts.get('start', 0):5d} "
                  f"start_end={counts.get('start_end', 0):5d} "
                  f"continue={counts.get('continue', 0):5d} "
                  f"end={counts.get('end', 0):5d}", flush=True)
            for record in labels:
                columns["ticker"].append(ticker)
                columns["tf"].append(timeframe)
                columns["origin"].append(record["origin"])
                columns["confirm"].append(record["confirm"])
                columns["ts"].append(timestamps[record["confirm"]])
                columns["direction"].append(record["direction"])
                columns["role"].append(int(record["role"] == "high"))
                columns["swing"].append(SWING[record["swing_type"]])
                columns["trend_dir"].append(record["trend_dir"] or 0)
                columns["is_start"].append(int(record["role_kind"] == "start"))
                columns["ended"].append(int(record["ended"]))
                columns["kind"].append(KIND[record["kind"]])
                for name in ("px", "leg_idx", "run_len", "extent_atr", "resolved"):
                    columns[name].append(record[name])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        ticker=np.asarray(columns["ticker"]), tf=np.asarray(columns["tf"]),
        origin=np.asarray(columns["origin"], np.int64),
        confirm=np.asarray(columns["confirm"], np.int64),
        ts=np.asarray(columns["ts"], dtype="datetime64[ns]"),
        direction=np.asarray(columns["direction"], np.int8),
        role=np.asarray(columns["role"], np.int8),
        swing=np.asarray(columns["swing"], np.int8),
        trend_dir=np.asarray(columns["trend_dir"], np.int8),
        is_start=np.asarray(columns["is_start"], np.int8),
        ended=np.asarray(columns["ended"], np.int8),
        kind=np.asarray(columns["kind"], np.int8),
        px=np.asarray(columns["px"], np.float64),
        leg_idx=np.asarray(columns["leg_idx"], np.int32),
        run_len=np.asarray(columns["run_len"], np.int32),
        extent_atr=np.asarray(columns["extent_atr"], np.float32),
        resolved=np.asarray(columns["resolved"], np.int8),
    )
    print(f"wrote {OUTPUT} ({len(columns['ticker']):,} pivots)", flush=True)


if __name__ == "__main__":
    main()
