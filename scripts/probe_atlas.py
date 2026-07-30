#!/usr/bin/env python3
"""Strategy-agnostic capability atlas for FFM encoder checkpoints.

The Atlas evaluates frozen embeddings on a balanced 9-ticker x 4-timeframe
market-structure corpus.  Inputs are causal 128-bar OHLCV windows ending at a
confirmed structural pivot.  Targets measure information retention and generic
future market behavior; there are no entries, stops, R targets, position rules,
or imports from the private strategies repository.

Environment contract (normally set by ``mantis_ssl_clean_pipeline.py``):
  CKPT_PATH, CKPT_NAME, CKPT_SHA256, EMB_CACHE, ATLAS_OUT, ATLAS_BATCH,
  DEVICE, DATA_DIR, TREND_LABELS, FFM_ROOT. ``ATLAS_BACKBONE=chronos2``
  selects the Chronos-2 streaming embedder; the default remains Mantis.
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
LOAD_CKPT_PATH = os.environ.get(
    "ATLAS_LOAD_CHECKPOINT_PATH", CKPT_PATH)
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
BACKBONE = os.environ.get("ATLAS_BACKBONE", "mantis").strip().lower()
CONTROL = os.environ.get("ATLAS_CONTROL", "real").strip().lower()
POOL = os.environ.get(
    "ATLAS_POOL",
    "reg" if BACKBONE == "chronos2" else "encoder_default",
).strip().lower()
STAGE_REPORT_SHA256 = os.environ.get("ATLAS_STAGE_REPORT_SHA256")
PARENT_CHECKPOINT_SHA256 = os.environ.get(
    "ATLAS_PARENT_CHECKPOINT_SHA256")
DATA_IDENTITY_SHA256 = os.environ.get("ATLAS_DATA_IDENTITY_SHA256")
BASE_REVISION = os.environ.get("ATLAS_BASE_REVISION")
BASE_WEIGHTS_SHA256 = os.environ.get("ATLAS_BASE_WEIGHTS_SHA256")
BASE_CONFIG_SHA256 = os.environ.get("ATLAS_BASE_CONFIG_SHA256")

TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
FIT_END = pd.Timestamp("2024-01-01", tz="UTC")
EVAL_START = pd.Timestamp("2025-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW = int(os.environ.get("ATLAS_WINDOW", "128"))
PROBE_HORIZONS = tuple(
    int(value) for value in os.environ.get(
        "ATLAS_HORIZONS", "5,10,20,50").split(",") if value)
FORWARD = 20
VOL_FORWARD = 50
ATR_PERIOD = 20
MV_HORIZON = 20
MV_SCALE_LOOKBACK = 64
MV_MOMENTUM_THRESHOLD = 0.5
MV_EXPANSION_THRESHOLD = 1.1
ATLAS_SCHEMA = "ffm_probe_atlas_v5"
TARGET_SCHEMA = "ffm_probe_atlas_targets_v2_multihorizon_direction"
FORBIDDEN_NONCAUSAL_PROBES = frozenset({
    "pred_persistent_trend_start",
})
MAX_FORWARD = max((FORWARD, VOL_FORWARD, MV_HORIZON, *PROBE_HORIZONS))
MULTI_HORIZON_FIELDS = tuple(
    name
    for horizon in PROBE_HORIZONS
    for name in (
        f"trend_strength_{horizon}",
        f"range_expansion_{horizon}",
        f"future_direction_{horizon}",
    )
)

if BACKBONE not in {"mantis", "chronos2"}:
    raise ValueError("ATLAS_BACKBONE must be 'mantis' or 'chronos2'")
if CONTROL not in {"real", "shuffle", "random"}:
    raise ValueError("ATLAS_CONTROL must be 'real', 'shuffle', or 'random'")
if (
    (BACKBONE == "chronos2" and POOL not in {"reg", "mean_context"})
    or (BACKBONE == "mantis" and POOL != "encoder_default")
):
    raise ValueError("ATLAS_POOL is incompatible with ATLAS_BACKBONE")
if WINDOW < 2:
    raise ValueError("ATLAS_WINDOW must be >=2")
if (
    not PROBE_HORIZONS
    or any(value <= 0 for value in PROBE_HORIZONS)
    or len(set(PROBE_HORIZONS)) != len(PROBE_HORIZONS)
):
    raise ValueError("ATLAS_HORIZONS must be a non-empty unique positive list")


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


def _future_rolling(values: np.ndarray, width: int, reducer: str) -> np.ndarray:
    """Aggregate exactly ``values[i + 1:i + width + 1]`` at decision row ``i``."""
    shifted = pd.Series(np.asarray(values, float)).shift(-1)
    rolling = shifted.rolling(width, min_periods=width)
    aggregated = getattr(rolling, reducer)()
    return aggregated.shift(-(width - 1)).to_numpy()


def _future_direction(close: np.ndarray, horizon: int) -> np.ndarray:
    """Signed close displacement at exactly t+h; future target only."""
    close = np.asarray(close, float)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("direction horizon must be positive")
    direction = np.full(len(close), np.nan, dtype=np.float32)
    if horizon < len(close):
        direction[:-horizon] = (
            close[horizon:] > close[:-horizon]).astype(np.float32)
    return direction


def _momentum_volatility_fields(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    horizon: int = MV_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact headless MV-v3 targets for frozen-encoder transfer probes.

    The current and prior 63 completed candle ranges define the causal scale.
    Targets use only the next 20 completed bars and therefore remain disjoint
    from every input window ending at the decision row.
    """
    candle_range = np.asarray(high, float) - np.asarray(low, float)
    causal_scale = (
        pd.Series(candle_range)
        .rolling(MV_SCALE_LOOKBACK, min_periods=MV_SCALE_LOOKBACK)
        .median()
        .to_numpy()
    )
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("momentum-volatility horizon must be positive")
    future_range = _future_rolling(candle_range, horizon, "median")
    close = np.asarray(close, float)
    step = np.r_[np.nan, np.abs(np.diff(close))]
    path_length = _future_rolling(step, horizon, "sum")
    displacement = np.full(len(close), np.nan)
    displacement[:-horizon] = np.abs(
        close[horizon:] - close[:-horizon])
    strength = displacement / np.where(path_length > 0, path_length, np.nan)
    expansion = future_range / np.where(causal_scale > 0, causal_scale, np.nan)

    state = np.full(len(close), -1, np.int8)
    valid = np.isfinite(strength) & np.isfinite(expansion) & (expansion > 0)
    directional = strength >= MV_MOMENTUM_THRESHOLD
    expanding = expansion >= MV_EXPANSION_THRESHOLD
    state[valid & directional & expanding] = 0
    state[valid & directional & ~expanding] = 1
    state[valid & ~directional & expanding] = 2
    state[valid & ~directional & ~expanding] = 3
    return strength, expansion, state


def _load_bars(ticker: str, timeframe: str) -> dict:
    from futures_foundation.pipeline._primitives import compute_atr

    source_path = DATA_DIR / f"{ticker}_{timeframe}.csv"
    manifest_path = source_path.with_suffix(".csv.manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"source manifest missing for Probe Atlas: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source_sha256 = manifest.get("output_sha256")
    if not source_sha256:
        raise RuntimeError(
            f"source manifest has no output_sha256: {manifest_path}")
    frame = pd.read_csv(
        source_path,
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
        "source_sha256": str(source_sha256),
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
    mv_strength, mv_expansion, mv_state = _momentum_volatility_fields(
        high, low, close)
    result = {
        "atr_pct": _rolling_percentile(atr, 2000),
        "squeeze": atr / np.where(atr100 > 0, atr100, np.nan),
        "vol_z": ((volume_series - volume_series.rolling(500).mean())
                  / volume_series.rolling(500).std()).to_numpy(),
        "day_pos": ((close - day_low)
                    / np.where(day_high > day_low, day_high - day_low, np.nan)),
        "hour": bars["ts"].hour.to_numpy().astype(float),
        "forward_return": forward_return,
        "atr_forward": atr_forward,
        "mv_strength": mv_strength,
        "mv_expansion": mv_expansion,
        "mv_state": mv_state,
    }
    for horizon in PROBE_HORIZONS:
        strength, expansion, _ = _momentum_volatility_fields(
            high, low, close, horizon=horizon)
        result[f"trend_strength_{horizon}"] = strength
        result[f"range_expansion_{horizon}"] = expansion
        result[f"future_direction_{horizon}"] = _future_direction(
            close, horizon)
    return result


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
        "forward_return": [], "atr_forward": [], "mv_strength": [],
        "mv_expansion": [], "mv_state": [], "stream": [],
    }
    rows_by_field.update({name: [] for name in MULTI_HORIZON_FIELDS})
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
                     & (confirms + MAX_FORWARD < len(bars["c"])))
            selected, confirms, timestamps = selected[valid], confirms[valid], timestamps[valid]
            if len(selected):
                actual = bars["ts"].asi8[confirms]
                expected = timestamps.asi8
                if not np.array_equal(actual, expected):
                    mismatch = int(np.flatnonzero(actual != expected)[0])
                    raise RuntimeError(
                        f"corpus/bar index mismatch for {ticker}@{timeframe} at row {mismatch}")

            duration = pd.Timedelta(timeframe)
            decision_close = bars["ts"][confirms] + duration
            target_close = bars["ts"][confirms + MAX_FORWARD] + duration
            train_rows = _even_sample(np.where(
                (decision_close < FIT_END) & (target_close < FIT_END)
            )[0], TRAIN_PER_STREAM)
            eval_rows = _even_sample(
                np.where(
                    (decision_close >= EVAL_START)
                    & (target_close < EVAL_END)
                )[0],
                EVAL_PER_STREAM)
            keep = np.concatenate([train_rows, eval_rows])
            stream_id = f"{ticker}@{timeframe}"
            for local in keep:
                corpus_row = int(selected[local])
                confirm = int(confirms[local])
                all_keys.append((ticker, timeframe, confirm))
                rows_by_field["timestamp"].append(decision_close[local])
                rows_by_field["trend_dir"].append(int(corpus["trend_dir"][corpus_row]))
                rows_by_field["is_start"].append(bool(corpus["is_start"][corpus_row]))
                rows_by_field["ended"].append(bool(corpus["ended"][corpus_row]))
                rows_by_field["stream"].append(stream_id)
                for name in ("atr_pct", "squeeze", "vol_z", "day_pos", "hour",
                             "forward_return", "atr_forward", "mv_strength",
                             "mv_expansion", "mv_state", *MULTI_HORIZON_FIELDS):
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(bars_by_stream: dict) -> str:
    payload = {
        f"{ticker}@{timeframe}": bars["source_sha256"]
        for (ticker, timeframe), bars in sorted(bars_by_stream.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _embedder_sha256() -> str:
    if BACKBONE == "chronos2":
        from futures_foundation.finetune.classifiers.chronos2 import (
            _embed_worker,
        )
        module_path = Path(str(_embed_worker.__file__)).resolve()
    else:
        from futures_foundation.finetune.pretext._torch import common
        module_path = Path(str(common.__file__)).resolve()
    if not module_path.is_file():
        raise RuntimeError("Probe Atlas embedder source is unavailable")
    return _file_sha256(module_path)


def _controlled_window(window: np.ndarray, key: tuple) -> np.ndarray:
    """Deterministic input-only REAL/SHUFFLE/RANDOM Atlas controls."""
    if CONTROL == "real":
        return window
    seed = int.from_bytes(
        hashlib.sha256(f"{key}|{CONTROL}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if CONTROL == "shuffle":
        # Preserve cross-channel bar alignment while destroying temporal order.
        return window[:, rng.permutation(window.shape[1])]
    return rng.standard_normal(window.shape).astype(np.float32)


def _context_chunks(bars_by_stream: dict, keys: list[tuple]):
    for start in range(0, len(keys), ATLAS_CHUNK):
        block = []
        for ticker, timeframe, confirm in keys[start:start + ATLAS_CHUNK]:
            bars = bars_by_stream[(ticker, timeframe)]
            slc = slice(confirm - WINDOW + 1, confirm + 1)
            window = np.stack([bars[name][slc] for name in ("o", "h", "l", "c", "v")])
            clean = np.nan_to_num(np.asarray(window, np.float32))
            block.append(_controlled_window(
                clean, (ticker, timeframe, confirm)))
        yield np.stack(block)


def _embeddings(bars_by_stream: dict, keys: list[tuple]) -> np.ndarray:
    identity_path = Path(str(EMB_CACHE) + ".pool.json")
    identity = {
        "schema": "ffm_probe_atlas_pool_v2",
        "target_schema": TARGET_SCHEMA,
        "rows": len(keys),
        "pool_sha256": _pool_sha256(keys),
        "source_sha256": _source_sha256(bars_by_stream),
        "label_corpus_sha256": _file_sha256(CORPUS),
        "backbone": BACKBONE,
        "checkpoint_sha256": os.environ.get("CKPT_SHA256"),
        "stage_report_sha256": STAGE_REPORT_SHA256,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "data_identity_sha256": DATA_IDENTITY_SHA256,
        "base_revision": BASE_REVISION,
        "base_weights_sha256": BASE_WEIGHTS_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "window": WINDOW,
        "horizons": list(PROBE_HORIZONS),
        "pool": POOL,
        "control": CONTROL,
        "device": DEVICE,
        "train_per_stream": TRAIN_PER_STREAM,
        "eval_per_stream": EVAL_PER_STREAM,
        "atlas_code_sha256": _file_sha256(Path(__file__).resolve()),
        "embedder_code_sha256": _embedder_sha256(),
    }
    if EMB_CACHE.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise RuntimeError(f"embedding cache has different pool identity: {EMB_CACHE}")
        cached = np.load(EMB_CACHE, mmap_mode="r")
        if len(cached) != len(keys):
            raise RuntimeError(f"embedding/pool mismatch {len(cached)} vs {len(keys)}")
        print(f"[emb-cache] HIT {EMB_CACHE.name} ({len(cached):,})", flush=True)
        return cached
    if BACKBONE == "mantis":
        if not CKPT_PATH or not Path(CKPT_PATH).is_file():
            raise FileNotFoundError(
                f"Mantis checkpoint not found for {CKPT_NAME}: {CKPT_PATH}")
        from futures_foundation.finetune.pretext._torch.common import (
            embed_window_chunks,
        )
        encoded = embed_window_chunks(
            _context_chunks(bars_by_stream, keys),
            ckpt=CKPT_PATH,
            device=DEVICE,
            batch=ATLAS_BATCH,
        )
    else:
        checkpoint = LOAD_CKPT_PATH or "autogluon/chronos-2-small"
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists() and not checkpoint_path.is_dir():
            raise RuntimeError(
                "Chronos-2 adapter checkpoint must be a PEFT directory")
        from futures_foundation.finetune.classifiers.chronos2._embed_worker import (
            embed_window_chunks,
        )
        encoded = embed_window_chunks(
            _context_chunks(bars_by_stream, keys),
            checkpoint=checkpoint,
            device=DEVICE or "cpu",
            batch=ATLAS_BATCH,
            pool=POOL,
            context_length=WINDOW,
        )

    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(EMB_CACHE) + ".partial.npy")
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


def _assert_probe_is_causal(name: str) -> None:
    if name in FORBIDDEN_NONCAUSAL_PROBES:
        raise RuntimeError(
            f"Probe {name!r} is forbidden because its target is not bounded "
            "to the configured forward reserve"
        )


def _fit_probe(name: str, family: str, labels: np.ndarray, valid: np.ndarray,
               train: np.ndarray, evaluate: np.ndarray, embeddings: np.ndarray,
               streams: np.ndarray) -> dict | None:
    _assert_probe_is_causal(name)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

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
    fit_embeddings = np.asarray(embeddings[fit_rows], np.float32)
    scaler = StandardScaler().fit(fit_embeddings)
    classifier = LogisticRegression(max_iter=1000, C=0.1).fit(
        scaler.transform(fit_embeddings), binary[fit_rows])
    test_rows = np.where(test)[0]
    probability = classifier.predict_proba(
        scaler.transform(
            np.asarray(embeddings[test_rows], np.float32)))[:, 1]
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
    mv_valid = common & (fields["mv_state"] >= 0)
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
        "ret_structural_direction": (
            "retention",
            fields["trend_dir"] > 0,
            common & (fields["trend_dir"] != 0),
        ),
        "pred_fwd_direction": ("prediction", fields["forward_return"] > 0,
                               common & np.isfinite(fields["forward_return"])),
        "pred_fwd_large_move": ("prediction", np.abs(fields["forward_return"]) > magnitude_cut,
                                common & np.isfinite(fields["forward_return"])),
        "pred_vol_expand": ("prediction", fields["atr_forward"] > 1.2,
                            common & np.isfinite(fields["atr_forward"])),
        "pred_mv_trend_expansion": (
            "prediction", fields["mv_state"] == 0, mv_valid),
        "pred_mv_trend_weakening": (
            "prediction", fields["mv_state"] == 1, mv_valid),
        "pred_mv_noisy_expansion": (
            "prediction", fields["mv_state"] == 2, mv_valid),
        "pred_mv_compression": (
            "prediction", fields["mv_state"] == 3, mv_valid),
    }
    for horizon in PROBE_HORIZONS:
        trend = fields[f"trend_strength_{horizon}"]
        expansion = fields[f"range_expansion_{horizon}"]
        direction = fields[f"future_direction_{horizon}"]
        probes[f"pred_trend_h{horizon}"] = (
            "prediction",
            trend >= MV_MOMENTUM_THRESHOLD,
            common & np.isfinite(trend),
        )
        probes[f"pred_expansion_h{horizon}"] = (
            "prediction",
            expansion >= MV_EXPANSION_THRESHOLD,
            common & np.isfinite(expansion),
        )
        probes[f"pred_direction_h{horizon}"] = (
            "prediction",
            direction > 0,
            common & np.isfinite(direction),
        )
        probes[f"pred_trend_direction_h{horizon}"] = (
            "prediction",
            direction > 0,
            common
            & np.isfinite(direction)
            & np.isfinite(trend)
            & (trend >= MV_MOMENTUM_THRESHOLD),
        )
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
            "schema": ATLAS_SCHEMA, "scope": "9x4_strategy_agnostic",
            "target_schema": TARGET_SCHEMA,
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "backbone": BACKBONE, "control": CONTROL,
            "checkpoint": CKPT_NAME, "checkpoint_path": CKPT_PATH,
            "checkpoint_sha256": os.environ.get("CKPT_SHA256"),
            "stage_report_sha256": STAGE_REPORT_SHA256,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "data_identity_sha256": DATA_IDENTITY_SHA256,
            "base_revision": BASE_REVISION,
            "base_weights_sha256": BASE_WEIGHTS_SHA256,
            "base_config_sha256": BASE_CONFIG_SHA256,
            "embedding_cache": str(EMB_CACHE), "fit": "<2024", "eval": "2025",
            "window": WINDOW, "horizons": list(PROBE_HORIZONS), "pool": POOL,
            "target_reserve_bars": MAX_FORWARD,
            "pool_rows": len(keys), "probes": results,
            "embedding_cache_identity": json.loads(
                Path(str(EMB_CACHE) + ".pool.json").read_text()
            ),
            "embedding_cache_identity_sha256": _file_sha256(
                Path(str(EMB_CACHE) + ".pool.json")
            ),
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
