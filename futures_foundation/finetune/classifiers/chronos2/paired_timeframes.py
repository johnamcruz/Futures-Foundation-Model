"""Causal 3-minute plus developing-15-minute Chronos-2 data.

This module is deliberately torch-free.  It authenticates one ticker's native
3-minute and 15-minute continuous-contract streams, proves that every native
15-minute bar is exactly the aggregation of five 3-minute bars, and exposes a
ten-variate matrix on the 3-minute *close* clock::

    [3m.open, 3m.high, 3m.low, 3m.close, 3m.volume,
     developing15m.open, developing15m.high, developing15m.low,
     developing15m.close, developing15m.volume]

The developing higher-timeframe bar is a prefix aggregation.  At phase one it
contains only the first completed 3-minute bar in its native 15-minute bucket;
at phase five it must equal the authenticated native 15-minute bar exactly.
No future 3-minute value, forward fill, indicator, phase id, or timestamp enters
the model matrix.  Bucket and phase metadata exist only for split/audit use.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.data_provenance import seal_continuous_streams
from futures_foundation.finetune.ssl_data import OHLCV_COLS

from .multivariate import DEFAULT_HOLDOUT_START


SCHEMA = "ffm_chronos2_paired_developing_timeframes_v1"
PRIMARY_TIMEFRAME = "3min"
RELATED_TIMEFRAME = "15min"
PRIMARY_MINUTES = 3
RELATED_MINUTES = 15
PHASES_PER_BUCKET = RELATED_MINUTES // PRIMARY_MINUTES
PRIMARY_CHANNEL_NAMES = tuple(f"3m.{name}" for name in OHLCV_COLS)
DEVELOPING_CHANNEL_NAMES = tuple(
    f"developing15m.{name}" for name in OHLCV_COLS
)
CHANNEL_NAMES = PRIMARY_CHANNEL_NAMES + DEVELOPING_CHANNEL_NAMES
AGGREGATION_SEMANTICS = {
    "clock": "native_15min_bar_open_intervals",
    "decision_clock": "completed_3min_bar_close",
    "source_timestamp_semantics": "bar_open",
    "developing_open": "first_observed_3min_open_in_native_15min_bucket",
    "developing_high": "maximum_observed_3min_high_in_native_15min_bucket",
    "developing_low": "minimum_observed_3min_low_in_native_15min_bucket",
    "developing_close": "latest_observed_3min_close_in_native_15min_bucket",
    "developing_volume": "sum_observed_3min_volume_in_native_15min_bucket",
    "future_values": "forbidden",
    "phase_five": "exact_authenticated_native_15min_ohlcv",
}
SPLIT_POLICY = (
    "pre_holdout_complete_native_15min_buckets_then_chronological_"
    "validation_snapped_to_native_15min_bucket_boundary"
)

_PRIMARY_DELTA = np.timedelta64(PRIMARY_MINUTES, "m")
_RELATED_DELTA = np.timedelta64(RELATED_MINUTES, "m")
_PRIMARY_NS = np.int64(PRIMARY_MINUTES * 60 * 1_000_000_000)
_RELATED_NS = np.int64(RELATED_MINUTES * 60 * 1_000_000_000)


@dataclass(frozen=True)
class Chronos2PairedTimeframeData:
    """Prepared paired matrices plus non-model timing audit metadata."""

    train: np.ndarray
    validation_matrix: np.ndarray
    validation: tuple[np.ndarray, ...]
    channel_names: tuple[str, ...]
    train_close_times: np.ndarray
    validation_close_times: np.ndarray
    train_bucket_open_times: np.ndarray
    validation_bucket_open_times: np.ndarray
    train_phases: np.ndarray
    validation_phases: np.ndarray
    validation_target_ranges: tuple[tuple[str, str], ...]
    report: dict


@dataclass(frozen=True)
class _PairedAssembly:
    values: np.ndarray
    close_times: np.ndarray
    bucket_open_times: np.ndarray
    phases: np.ndarray


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _readonly(values: np.ndarray, *, dtype: object | None = None) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_stream(frame: pd.DataFrame, *, stream: str, path: Path) -> None:
    if frame.empty:
        raise RuntimeError(f"empty OHLCV stream: {path}")
    timestamps = frame["datetime"]
    if timestamps.isna().any():
        raise RuntimeError(f"invalid timestamps in {stream}: {path}")
    if timestamps.duplicated().any():
        raise RuntimeError(f"duplicate timestamps in {stream}: {path}")
    if not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"timestamps are not strictly ordered in {stream}: {path}")
    values = frame[list(OHLCV_COLS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite source OHLCV in {stream}: {path}")
    open_, high, low, close, volume = values.T
    if (
        np.any(high < np.maximum(open_, close))
        or np.any(low > np.minimum(open_, close))
        or np.any(high < low)
    ):
        raise RuntimeError(f"invalid OHLC relationships in {stream}: {path}")
    if np.any(volume < 0.0):
        raise RuntimeError(f"negative volume in {stream}: {path}")
    if np.any(values[:, :4] <= 0.0):
        raise RuntimeError(f"non-positive price in {stream}: {path}")


def _load_stream(path: Path, *, stream: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, usecols=["datetime", *OHLCV_COLS])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"unreadable OHLCV stream {stream}: {path}") from exc
    frame["datetime"] = pd.to_datetime(
        frame["datetime"], utc=True, errors="coerce"
    )
    _validate_stream(frame, stream=stream, path=path)
    return frame


def _assemble_pair(
    primary: pd.DataFrame,
    native_related: pd.DataFrame,
) -> _PairedAssembly:
    """Build prefix-stable developing-15-minute OHLCV on the 3-minute clock."""
    primary_open = pd.DatetimeIndex(primary["datetime"])
    related_open = pd.DatetimeIndex(native_related["datetime"])
    primary_open_ns = primary_open.asi8
    related_open_ns = related_open.asi8
    if (
        primary_open_ns[-1] > np.iinfo(np.int64).max - _PRIMARY_NS
        or related_open_ns[-1] > np.iinfo(np.int64).max - _RELATED_NS
    ):
        raise RuntimeError("paired timestamps cannot form bar closes")

    primary_close_ns = primary_open_ns + _PRIMARY_NS
    # ``side='left'`` is the exact-boundary rule.  At 10:15, the 10:00
    # native bucket has just completed; the empty 10:15 bucket is unavailable.
    bucket_indices = np.searchsorted(
        related_open_ns, primary_close_ns, side="left"
    ) - 1
    if np.any(bucket_indices < 0) or np.any(bucket_indices >= len(related_open_ns)):
        raise RuntimeError(
            "a 3-minute close has no authenticated native 15-minute bucket"
        )

    bucket_open_ns = related_open_ns[bucket_indices]
    offsets = primary_open_ns - bucket_open_ns
    phases_zero = offsets // _PRIMARY_NS
    if (
        np.any(offsets < 0)
        or np.any(offsets % _PRIMARY_NS != 0)
        or np.any(phases_zero < 0)
        or np.any(phases_zero >= PHASES_PER_BUCKET)
        or np.any(primary_close_ns > bucket_open_ns + _RELATED_NS)
    ):
        raise RuntimeError(
            "3-minute timestamps are misaligned with native 15-minute buckets"
        )

    counts = np.bincount(bucket_indices, minlength=len(related_open_ns))
    if (
        len(counts) != len(related_open_ns)
        or np.any(counts != PHASES_PER_BUCKET)
    ):
        raise RuntimeError(
            "every native 15-minute bucket must contain exactly five 3-minute bars"
        )
    expected_buckets = np.repeat(
        np.arange(len(related_open_ns), dtype=np.int64),
        PHASES_PER_BUCKET,
    )
    expected_phases = np.tile(
        np.arange(PHASES_PER_BUCKET, dtype=np.int64),
        len(related_open_ns),
    )
    if (
        not np.array_equal(bucket_indices, expected_buckets)
        or not np.array_equal(phases_zero, expected_phases)
    ):
        raise RuntimeError(
            "3-minute bars do not occupy every native 15-minute phase in order"
        )

    primary_values = primary[list(OHLCV_COLS)].to_numpy(np.float64)
    related_values = native_related[list(OHLCV_COLS)].to_numpy(np.float64)
    blocks = primary_values.reshape(
        len(related_values), PHASES_PER_BUCKET, len(OHLCV_COLS)
    )
    developing_blocks = np.empty_like(blocks)
    developing_blocks[:, :, 0] = blocks[:, :1, 0]
    developing_blocks[:, :, 1] = np.maximum.accumulate(blocks[:, :, 1], axis=1)
    developing_blocks[:, :, 2] = np.minimum.accumulate(blocks[:, :, 2], axis=1)
    developing_blocks[:, :, 3] = blocks[:, :, 3]
    developing_blocks[:, :, 4] = np.cumsum(blocks[:, :, 4], axis=1)

    phase_five = developing_blocks[:, -1, :]
    if not np.array_equal(phase_five, related_values):
        mismatch = np.argwhere(phase_five != related_values)[0]
        bucket, channel = (int(mismatch[0]), int(mismatch[1]))
        raise RuntimeError(
            "fifth developing phase does not exactly match authenticated native "
            f"15-minute OHLCV at bucket={bucket} channel={OHLCV_COLS[channel]}"
        )

    developing = developing_blocks.reshape(len(primary_values), len(OHLCV_COLS))
    paired = np.concatenate((primary_values, developing), axis=1)
    if paired.shape != (len(primary_values), len(CHANNEL_NAMES)):
        raise RuntimeError("paired timeframe channel shape drifted")
    if not np.isfinite(paired).all():
        raise RuntimeError("paired timeframe matrix contains non-finite values")
    return _PairedAssembly(
        values=paired.astype(np.float32),
        close_times=(primary_open + _PRIMARY_DELTA).to_numpy(
            dtype="datetime64[ns]"
        ),
        bucket_open_times=related_open[bucket_indices].to_numpy(
            dtype="datetime64[ns]"
        ),
        phases=(phases_zero + 1).astype(np.int8),
    )


def _validation_windows(
    matrix: np.ndarray,
    close_times: pd.DatetimeIndex,
    *,
    validation_start: int,
    context_length: int,
    prediction_length: int,
    count: int,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[str, str], ...]]:
    first_end = validation_start + prediction_length
    if first_end > len(matrix):
        raise RuntimeError("validation region is shorter than prediction_length")
    candidates = np.linspace(
        first_end, len(matrix), num=count, dtype=np.int64
    )
    windows: list[np.ndarray] = []
    ranges: list[tuple[str, str]] = []
    for end in sorted(set(int(value) for value in candidates)):
        target_start = end - prediction_length
        if target_start < validation_start:
            continue
        start = max(0, end - context_length - prediction_length)
        window = matrix[start:end].T.astype(np.float32, copy=True)
        context = window[:, :-prediction_length]
        target = window[:, -prediction_length:]
        if (
            context.shape[-1] < 1
            or target.shape[-1] != prediction_length
            or not np.isfinite(window).all()
        ):
            continue
        windows.append(window)
        ranges.append(
            (
                close_times[target_start].isoformat(),
                close_times[end - 1].isoformat(),
            )
        )
    if not windows:
        raise RuntimeError("no chronological paired validation window is usable")
    return tuple(windows), tuple(ranges)


def _positive_int(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _holdout_timestamp(value: str) -> pd.Timestamp:
    try:
        holdout = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("holdout_start must be a valid timestamp") from exc
    if pd.isna(holdout):
        raise ValueError("holdout_start must be a valid timestamp")
    return (
        holdout.tz_localize("UTC")
        if holdout.tzinfo is None
        else holdout.tz_convert("UTC")
    )


def _bucket_aligned_validation_start(
    bucket_open_times: np.ndarray,
    *,
    raw_start: int,
) -> int:
    start = int(raw_start)
    while (
        0 < start < len(bucket_open_times)
        and bucket_open_times[start] == bucket_open_times[start - 1]
    ):
        start += 1
    return start


def prepare_paired_timeframes(
    data_dir: str | Path,
    *,
    ticker: str = "NQ",
    repo_root: str | Path | None = None,
    holdout_start: str = DEFAULT_HOLDOUT_START,
    val_frac: float = 0.10,
    context_length: int = 256,
    prediction_length: int = 32,
    validation_windows: int = 8,
    report_path: str | Path | None = None,
) -> Chronos2PairedTimeframeData:
    """Prepare one authenticated 3m/developing-15m ticker group.

    Only whole native 15-minute buckets whose close is strictly before the
    holdout are development data.  The chronological train/validation boundary
    is also snapped forward to the next native bucket, so correlated phases
    never straddle temporal roles.
    """
    if (
        not isinstance(ticker, str)
        or not ticker.strip()
        or "@" in ticker
    ):
        raise ValueError("ticker must be one non-empty ticker root")
    ticker = ticker.strip()
    if (
        isinstance(val_frac, (bool, np.bool_))
        or not isinstance(val_frac, (int, float, np.integer, np.floating))
        or not 0.0 < float(val_frac) < 1.0
    ):
        raise ValueError("val_frac must be between zero and one")
    context_length = _positive_int(context_length, name="context_length")
    prediction_length = _positive_int(
        prediction_length, name="prediction_length"
    )
    validation_windows = _positive_int(
        validation_windows, name="validation_windows"
    )
    holdout = _holdout_timestamp(holdout_start)

    data_dir = Path(data_dir)
    root = Path(repo_root) if repo_root is not None else None
    streams = (
        (ticker, PRIMARY_TIMEFRAME),
        (ticker, RELATED_TIMEFRAME),
    )
    provenance = seal_continuous_streams(data_dir, streams, repo_root=root)
    primary_path = data_dir / f"{ticker}_{PRIMARY_TIMEFRAME}.csv"
    related_path = data_dir / f"{ticker}_{RELATED_TIMEFRAME}.csv"
    primary = _load_stream(
        primary_path, stream=f"{ticker}@{PRIMARY_TIMEFRAME}"
    )
    related = _load_stream(
        related_path, stream=f"{ticker}@{RELATED_TIMEFRAME}"
    )
    assembled = _assemble_pair(primary, related)

    close_times = pd.DatetimeIndex(assembled.close_times).tz_localize("UTC")
    bucket_open_times = pd.DatetimeIndex(
        assembled.bucket_open_times
    ).tz_localize("UTC")
    bucket_close_times = bucket_open_times + _RELATED_DELTA
    # Conservative bucket-atomic holdout: if phase five is at/after the sealed
    # boundary, all earlier partial phases of that bucket are withheld too.
    development_mask = np.asarray(bucket_close_times < holdout)
    development_rows = int(development_mask.sum())
    if (
        development_rows < 1
        or not development_mask[:development_rows].all()
        or development_mask[development_rows:].any()
    ):
        raise RuntimeError(
            "holdout must leave one chronological prefix of complete 15-minute buckets"
        )
    minimum_rows = context_length + 2 * prediction_length
    if development_rows < minimum_rows:
        raise RuntimeError(
            "insufficient pre-holdout paired history for context and validation"
        )

    values = assembled.values[:development_rows]
    development_close = close_times[:development_rows]
    development_bucket = bucket_open_times[:development_rows]
    development_phases = assembled.phases[:development_rows]
    raw_start = int(np.floor(development_rows * (1.0 - float(val_frac))))
    validation_start = _bucket_aligned_validation_start(
        development_bucket.to_numpy(dtype="datetime64[ns]"),
        raw_start=raw_start,
    )
    if (
        validation_start < context_length + prediction_length
        or development_rows - validation_start < prediction_length
        or validation_start >= development_rows
    ):
        raise RuntimeError(
            "bucket-aligned chronological train/validation split is too short"
        )
    if (
        development_bucket[validation_start]
        == development_bucket[validation_start - 1]
        or int(development_phases[validation_start]) != 1
        or int(development_phases[validation_start - 1]) != PHASES_PER_BUCKET
    ):
        raise RuntimeError("chronological validation boundary split a native bucket")

    validation, validation_ranges = _validation_windows(
        values,
        development_close,
        validation_start=validation_start,
        context_length=context_length,
        prediction_length=prediction_length,
        count=validation_windows,
    )
    identity_payload = {
        "schema": SCHEMA,
        "ticker": ticker,
        "primary_timeframe": PRIMARY_TIMEFRAME,
        "related_timeframe": RELATED_TIMEFRAME,
        "channels": list(CHANNEL_NAMES),
        "aggregation": AGGREGATION_SEMANTICS,
        "alignment": "strict_exact_five_3min_bars_per_native_15min_bucket",
        "source_provenance": provenance,
        "holdout_start": holdout.isoformat(),
        "split": {
            "policy": SPLIT_POLICY,
            "val_frac": float(val_frac),
            "context_length": context_length,
            "prediction_length": prediction_length,
            "validation_windows": validation_windows,
            "development_rows": development_rows,
            "validation_start_row": validation_start,
            "validation_start_close": development_close[
                validation_start
            ].isoformat(),
            "validation_start_bucket_open": development_bucket[
                validation_start
            ].isoformat(),
            "validation_target_ranges": [list(item) for item in validation_ranges],
        },
    }
    identity_sha256 = hashlib.sha256(
        _canonical(identity_payload).encode()
    ).hexdigest()
    report = {
        **identity_payload,
        "identity_sha256": identity_sha256,
        "source_rows": {
            PRIMARY_TIMEFRAME: len(primary),
            RELATED_TIMEFRAME: len(related),
        },
        "paired_rows_pre_holdout": development_rows,
        "paired_close_start": development_close[0].isoformat(),
        "paired_close_end": development_close[-1].isoformat(),
        "train_rows": validation_start,
        "train_close_end": development_close[validation_start - 1].isoformat(),
        "validation_rows": development_rows - validation_start,
        "validation_close_start": development_close[validation_start].isoformat(),
        "validation_target_ranges": [list(item) for item in validation_ranges],
        "phase_counts": {
            str(phase): int(np.sum(development_phases == phase))
            for phase in range(1, PHASES_PER_BUCKET + 1)
        },
        "n_variates": len(CHANNEL_NAMES),
        "model_inputs": "paired_ohlcv_channels_only",
        "audit_only_fields": ["bucket_open_time", "phase"],
    }
    if report_path is not None:
        _atomic_json(Path(report_path), report)

    return Chronos2PairedTimeframeData(
        train=_readonly(values[:validation_start].T, dtype=np.float32),
        validation_matrix=_readonly(
            values[validation_start:].T, dtype=np.float32
        ),
        validation=tuple(_readonly(item, dtype=np.float32) for item in validation),
        channel_names=CHANNEL_NAMES,
        train_close_times=_readonly(
            development_close[:validation_start].to_numpy(dtype="datetime64[ns]")
        ),
        validation_close_times=_readonly(
            development_close[validation_start:].to_numpy(dtype="datetime64[ns]")
        ),
        train_bucket_open_times=_readonly(
            development_bucket[:validation_start].to_numpy(
                dtype="datetime64[ns]"
            )
        ),
        validation_bucket_open_times=_readonly(
            development_bucket[validation_start:].to_numpy(
                dtype="datetime64[ns]"
            )
        ),
        train_phases=_readonly(
            development_phases[:validation_start], dtype=np.int8
        ),
        validation_phases=_readonly(
            development_phases[validation_start:], dtype=np.int8
        ),
        validation_target_ranges=validation_ranges,
        report=report,
    )


__all__ = [
    "AGGREGATION_SEMANTICS",
    "CHANNEL_NAMES",
    "Chronos2PairedTimeframeData",
    "DEVELOPING_CHANNEL_NAMES",
    "PHASES_PER_BUCKET",
    "PRIMARY_CHANNEL_NAMES",
    "PRIMARY_TIMEFRAME",
    "RELATED_TIMEFRAME",
    "SCHEMA",
    "SPLIT_POLICY",
    "prepare_paired_timeframes",
]
