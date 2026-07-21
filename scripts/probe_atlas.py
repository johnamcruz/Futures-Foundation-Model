#!/usr/bin/env python3
"""Strategy-agnostic capability atlas for FFM encoder checkpoints.

The Atlas evaluates frozen embeddings on a balanced 9-ticker x 4-timeframe
market-structure corpus.  Inputs are causal 128-bar OHLCV windows ending at a
confirmed structural pivot.  Targets measure information retention and generic
future market behavior; there are no entries, stops, R targets, position rules,
or imports from the private strategies repository.

Environment contract (normally set by ``mantis_ssl_clean_pipeline.py``):
  CKPT_PATH, CKPT_NAME, CKPT_SHA256, EMB_CACHE, ATLAS_OUT, ATLAS_BATCH,
  DEVICE, DATA_DIR, TREND_LABELS, FFM_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("FFM_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CKPT_NAME = os.environ.get("CKPT_NAME", "mantis_ssl_nextleg.pt")
CKPT_PATH = os.environ.get("CKPT_PATH") or next(
    (str(path) for path in (
        ROOT / "checkpoints" / CKPT_NAME,
        ROOT / "models" / CKPT_NAME,
        ROOT / "AI_Models" / CKPT_NAME,
    ) if path.exists()), None)
EMB_CACHE = Path(os.environ.get(
    "EMB_CACHE", ROOT / "temp" / f"probe_atlas_{Path(CKPT_NAME).stem}.npy"))
CORPUS = Path(os.environ.get(
    "TREND_LABELS", ROOT / "temp" / "trend_lifecycle_labels.npz"))
ATLAS_OUT = os.environ.get("ATLAS_OUT")
ATLAS_BATCH = int(os.environ.get("ATLAS_BATCH", "512"))
ATLAS_CHUNK = int(os.environ.get("ATLAS_CHUNK", "32768"))
TRAIN_PER_STREAM = int(os.environ.get("ATLAS_TRAIN_PER_STREAM", "6000"))
EVAL_PER_STREAM = int(os.environ.get("ATLAS_EVAL_PER_STREAM", "3000"))
DEVICE = os.environ.get("DEVICE")

TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
FIT_END = pd.Timestamp("2024-01-01", tz="UTC")
EVAL_START = pd.Timestamp("2025-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW = 128
FORWARD = 20
VOL_FORWARD = 50
ATR_PERIOD = 20


def _even_sample(rows: np.ndarray, limit: int) -> np.ndarray:
    """Deterministically retain temporal coverage without favoring long streams."""
    rows = np.asarray(rows, np.int64)
    if len(rows) <= limit:
        return rows
    return rows[np.linspace(0, len(rows) - 1, limit, dtype=np.int64)]


def _rolling_percentile(values: np.ndarray, width: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(width, min_periods=width // 4).apply(
        lambda window: ((window[:-1] < window[-1]).mean()
                        if len(window) > 1 else np.nan), raw=True).to_numpy()


def _load_bars(ticker: str, timeframe: str) -> dict:
    from futures_foundation.pipeline._primitives import compute_atr

    frame = pd.read_csv(
        DATA_DIR / f"{ticker}_{timeframe}.csv",
        usecols=["datetime", "open", "high", "low", "close", "volume"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[frame["datetime"] < EVAL_END].reset_index(drop=True)
    timestamps = pd.DatetimeIndex(frame["datetime"])
    o, h, l, c = (frame[name].to_numpy(float)
                  for name in ("open", "high", "low", "close"))
    volume = frame["volume"].to_numpy(float)
    return {
        "ts": timestamps, "o": o, "h": h, "l": l, "c": c, "v": volume,
        "atr": compute_atr(h, l, c, ATR_PERIOD),
    }


def _stream_fields(bars: dict) -> dict[str, np.ndarray]:
    """Causal retention fields plus bounded, target-side future fields."""
    atr, volume, close, high, low = (bars[name]
                                     for name in ("atr", "v", "c", "h", "l"))
    atr100 = pd.Series(atr).rolling(100).mean().to_numpy()
    volume_series = pd.Series(volume)
    day = bars["ts"].normalize()
    day_high = pd.Series(high).groupby(day).cummax().to_numpy()
    day_low = pd.Series(low).groupby(day).cummin().to_numpy()
    forward_return = np.full(len(close), np.nan)
    forward_return[:-FORWARD] = ((close[FORWARD:] - close[:-FORWARD])
                                 / np.where(atr[:-FORWARD] > 0, atr[:-FORWARD], np.nan))
    atr_forward = np.full(len(close), np.nan)
    atr_forward[:-VOL_FORWARD] = (
        atr[VOL_FORWARD:] / np.where(atr[:-VOL_FORWARD] > 0, atr[:-VOL_FORWARD], np.nan))
    return {
        "atr_pct": _rolling_percentile(atr, 2000),
        "squeeze": atr / np.where(atr100 > 0, atr100, np.nan),
        "vol_z": ((volume_series - volume_series.rolling(500).mean())
                  / volume_series.rolling(500).std()).to_numpy(),
        "day_pos": ((close - day_low)
                    / np.where(day_high > day_low, day_high - day_low, np.nan)),
        "hour": bars["ts"].hour.to_numpy().astype(float),
        "forward_return": forward_return,
        "atr_forward": atr_forward,
    }


def _load_pool() -> tuple[dict[tuple[str, str], dict], list[tuple], dict[str, np.ndarray]]:
    """Build the same balanced corpus for every checkpoint and every machine."""
    if not CORPUS.is_file():
        raise FileNotFoundError(f"trend lifecycle corpus not found: {CORPUS}")
    corpus = np.load(CORPUS, allow_pickle=False)
    required = {"ticker", "tf", "confirm", "ts", "trend_dir", "is_start", "ended"}
    missing = required - set(corpus.files)
    if missing:
        raise RuntimeError(f"trend lifecycle corpus missing fields: {sorted(missing)}")

    all_keys: list[tuple] = []
    rows_by_field: dict[str, list] = {
        "timestamp": [], "trend_dir": [], "is_start": [], "ended": [],
        "atr_pct": [], "squeeze": [], "vol_z": [], "day_pos": [], "hour": [],
        "forward_return": [], "atr_forward": [], "stream": [],
    }
    bars_by_stream = {}
    corpus_ticker = corpus["ticker"].astype(str)
    corpus_tf = corpus["tf"].astype(str)
    corpus_ts = pd.to_datetime(corpus["ts"], utc=True)

    for ticker in TICKERS:
        for timeframe in TIMEFRAMES:
            bars = _load_bars(ticker, timeframe)
            bars_by_stream[(ticker, timeframe)] = bars
            fields = _stream_fields(bars)
            selected = np.where((corpus_ticker == ticker) & (corpus_tf == timeframe))[0]
            confirms = np.asarray(corpus["confirm"][selected], np.int64)
            timestamps = pd.DatetimeIndex(corpus_ts[selected])
            valid = ((confirms >= WINDOW - 1)
                     & (confirms + VOL_FORWARD < len(bars["c"])))
            selected, confirms, timestamps = selected[valid], confirms[valid], timestamps[valid]
            if len(selected):
                actual = bars["ts"].asi8[confirms]
                expected = timestamps.asi8
                if not np.array_equal(actual, expected):
                    mismatch = int(np.flatnonzero(actual != expected)[0])
                    raise RuntimeError(
                        f"corpus/bar index mismatch for {ticker}@{timeframe} at row {mismatch}")

            train_rows = _even_sample(np.where(timestamps < FIT_END)[0], TRAIN_PER_STREAM)
            eval_rows = _even_sample(
                np.where((timestamps >= EVAL_START) & (timestamps < EVAL_END))[0],
                EVAL_PER_STREAM)
            keep = np.concatenate([train_rows, eval_rows])
            stream_id = f"{ticker}@{timeframe}"
            for local in keep:
                corpus_row = int(selected[local])
                confirm = int(confirms[local])
                all_keys.append((ticker, timeframe, confirm))
                rows_by_field["timestamp"].append(timestamps[local])
                rows_by_field["trend_dir"].append(int(corpus["trend_dir"][corpus_row]))
                rows_by_field["is_start"].append(bool(corpus["is_start"][corpus_row]))
                rows_by_field["ended"].append(bool(corpus["ended"][corpus_row]))
                rows_by_field["stream"].append(stream_id)
                for name in ("atr_pct", "squeeze", "vol_z", "day_pos", "hour",
                             "forward_return", "atr_forward"):
                    rows_by_field[name].append(fields[name][confirm])
            print(f"[pool] {stream_id}: fit={len(train_rows):,} eval={len(eval_rows):,}",
                  flush=True)
    if not all_keys:
        raise RuntimeError("Probe Atlas corpus is empty")
    arrays = {name: np.asarray(values) for name, values in rows_by_field.items()}
    return bars_by_stream, all_keys, arrays


def _pool_sha256(keys: list[tuple]) -> str:
    digest = hashlib.sha256()
    for ticker, timeframe, confirm in keys:
        digest.update(f"{ticker}@{timeframe}:{confirm}\n".encode())
    return digest.hexdigest()


def _context_chunks(bars_by_stream: dict, keys: list[tuple]):
    for start in range(0, len(keys), ATLAS_CHUNK):
        block = []
        for ticker, timeframe, confirm in keys[start:start + ATLAS_CHUNK]:
            bars = bars_by_stream[(ticker, timeframe)]
            slc = slice(confirm - WINDOW + 1, confirm + 1)
            window = np.stack([bars[name][slc] for name in ("o", "h", "l", "c", "v")])
            block.append(np.nan_to_num(np.asarray(window, np.float32)))
        yield np.stack(block)


def _embeddings(bars_by_stream: dict, keys: list[tuple]) -> np.ndarray:
    identity_path = Path(str(EMB_CACHE) + ".pool.json")
    identity = {"schema": "ffm_probe_atlas_pool_v1", "rows": len(keys),
                "pool_sha256": _pool_sha256(keys), "window": WINDOW,
                "train_per_stream": TRAIN_PER_STREAM, "eval_per_stream": EVAL_PER_STREAM}
    if EMB_CACHE.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise RuntimeError(f"embedding cache has different pool identity: {EMB_CACHE}")
        cached = np.load(EMB_CACHE, mmap_mode="r")
        if len(cached) != len(keys):
            raise RuntimeError(f"embedding/pool mismatch {len(cached)} vs {len(keys)}")
        print(f"[emb-cache] HIT {EMB_CACHE.name} ({len(cached):,})", flush=True)
        return cached
    if not CKPT_PATH or not Path(CKPT_PATH).is_file():
        raise FileNotFoundError(f"checkpoint not found for {CKPT_NAME}: {CKPT_PATH}")
    from futures_foundation.finetune.pretext._torch.common import embed_window_chunks

    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(EMB_CACHE) + ".partial.npy")
    encoded = embed_window_chunks(
        _context_chunks(bars_by_stream, keys), ckpt=CKPT_PATH,
        device=DEVICE, batch=ATLAS_BATCH)
    cache = None
    written = 0
    for block in encoded:
        if cache is None:
            cache = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16, shape=(len(keys), block.shape[1]))
        stop = written + len(block)
        cache[written:stop] = block.astype(np.float16)
        written = stop
        print(f"[emb-cache] {written:,}/{len(keys):,} ({written / len(keys):.1%})", flush=True)
    if cache is None or written != len(keys):
        raise RuntimeError(f"incomplete Atlas embedding cache: {written}/{len(keys)}")
    cache.flush()
    del cache
    os.replace(temporary, EMB_CACHE)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return np.load(EMB_CACHE, mmap_mode="r")


def _fit_probe(name: str, family: str, labels: np.ndarray, valid: np.ndarray,
               train: np.ndarray, evaluate: np.ndarray, embeddings: np.ndarray,
               streams: np.ndarray) -> dict | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fit = train & valid
    test = evaluate & valid
    binary = np.asarray(labels, int)
    if (fit.sum() < 2000 or test.sum() < 500
            or binary[fit].std() == 0 or binary[test].std() == 0):
        print(f"  {name:>26}: skipped (degenerate)", flush=True)
        return None
    fit_rows = np.where(fit)[0]
    if len(fit_rows) > 150_000:
        fit_rows = np.sort(np.random.default_rng(0).choice(
            fit_rows, 150_000, replace=False))
    classifier = LogisticRegression(max_iter=1000, C=0.1).fit(
        np.asarray(embeddings[fit_rows], np.float32), binary[fit_rows])
    test_rows = np.where(test)[0]
    probability = classifier.predict_proba(
        np.asarray(embeddings[test_rows], np.float32))[:, 1]
    pooled_auc = roc_auc_score(binary[test], probability)
    per_stream = {}
    test_streams = streams[test_rows]
    for stream in sorted(set(test_streams)):
        rows = test_streams == stream
        if rows.sum() >= 100 and binary[test_rows][rows].std() > 0:
            per_stream[stream] = round(float(
                roc_auc_score(binary[test_rows][rows], probability[rows])), 4)
    result = {
        "family": family, "auc": round(float(pooled_auc), 4),
        "pos_rate": round(float(binary[test].mean()), 4), "n_eval": int(test.sum()),
        "per_stream_auc": per_stream,
        "worst_stream_auc": min(per_stream.values()) if per_stream else None,
    }
    print(f"  {name:>26} [{family}] AUC={pooled_auc:.4f} "
          f"worst={result['worst_stream_auc']} pos={binary[test].mean():.2%}", flush=True)
    return result


def main() -> dict:
    bars_by_stream, keys, fields = _load_pool()
    embeddings = _embeddings(bars_by_stream, keys)
    timestamps = pd.DatetimeIndex(fields["timestamp"])
    train = np.asarray(timestamps < FIT_END)
    evaluate = np.asarray((timestamps >= EVAL_START) & (timestamps < EVAL_END))
    common = np.isfinite(fields["atr_pct"])
    magnitude_cut = np.nanmedian(np.abs(fields["forward_return"][train & common]))
    probes = {
        "ret_vol_regime": ("retention", fields["atr_pct"]
                           > np.nanmedian(fields["atr_pct"][train & common]), common),
        "ret_squeeze": ("retention", fields["squeeze"]
                        < np.nanquantile(fields["squeeze"][train & common], 1 / 3), common),
        "ret_vol_surge": ("retention", fields["vol_z"] > 1.0,
                          common & np.isfinite(fields["vol_z"])),
        "ret_day_position": ("retention", fields["day_pos"]
                             > np.nanmedian(fields["day_pos"][train & common]),
                             common & np.isfinite(fields["day_pos"])),
        "ret_ny_session": ("retention", (fields["hour"] >= 13) & (fields["hour"] < 20),
                           common),
        "pred_fwd_direction": ("prediction", fields["forward_return"] > 0,
                               common & np.isfinite(fields["forward_return"])),
        "pred_fwd_large_move": ("prediction", np.abs(fields["forward_return"]) > magnitude_cut,
                                common & np.isfinite(fields["forward_return"])),
        "pred_vol_expand": ("prediction", fields["atr_forward"] > 1.2,
                            common & np.isfinite(fields["atr_forward"])),
        "pred_persistent_trend_start": (
            "prediction", fields["is_start"] & ~fields["ended"], common & fields["is_start"]),
    }
    results = {}
    for name, (family, labels, valid) in probes.items():
        result = _fit_probe(
            name, family, labels, valid, train, evaluate, embeddings, fields["stream"])
        if result is not None:
            results[name] = result

    weak = sorted((row["auc"], name) for name, row in results.items()
                  if row["family"] == "retention")[:2]
    gaps = sorted((row["auc"], name) for name, row in results.items()
                  if row["family"] == "prediction")[:3]
    print(f"\n[atlas] weakest retention: {weak}\n[atlas] biggest prediction gaps: {gaps}",
          flush=True)
    if ATLAS_OUT:
        payload = {
            "schema": "ffm_probe_atlas_v2", "scope": "9x4_strategy_agnostic",
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "checkpoint": CKPT_NAME, "checkpoint_path": CKPT_PATH,
            "checkpoint_sha256": os.environ.get("CKPT_SHA256"),
            "embedding_cache": str(EMB_CACHE), "fit": "<2024", "eval": "2025",
            "pool_rows": len(keys), "probes": results,
            "weakest_retention": weak, "biggest_prediction_gaps": gaps,
        }
        destination = Path(ATLAS_OUT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, destination)
        print(f"[atlas] result -> {destination}", flush=True)
    return results


if __name__ == "__main__":
    main()
