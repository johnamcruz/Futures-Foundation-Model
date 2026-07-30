"""Causal multivariate OHLCV assembly for the Chronos-2 classifier package.

This module is intentionally torch/Chronos-free.  It authenticates the same
continuous-contract CSVs used by the Mantis SSL path, aligns streams by bar
*close* time, and returns arrays in Chronos-2's
``[n_variates, history_length]`` convention.

No value is forward-filled.  A missing bar remains NaN so Chronos-2's native
missing-value mask, rather than invented market data, controls attention/loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from futures_foundation.data_provenance import seal_continuous_streams
from futures_foundation.finetune.ssl_data import OHLCV_COLS, TICKERS_9


SCHEMA = "ffm_chronos2_multivariate_v1"
TIMEFRAME = "3min"
TIMEFRAME_MINUTES = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
}
DEFAULT_HOLDOUT_START = "2026-01-01"


@dataclass(frozen=True)
class Chronos2MultivariateData:
    """Prepared Chronos-2 inputs plus their immutable experiment identity."""

    train: np.ndarray
    validation_matrix: np.ndarray
    validation: tuple[np.ndarray, ...]
    channel_names: tuple[str, ...]
    train_close_times: np.ndarray
    validation_target_ranges: tuple[tuple[str, str], ...]
    report: dict


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_stream(frame: pd.DataFrame, *, ticker: str, path: Path) -> None:
    if frame.empty:
        raise RuntimeError(f"empty OHLCV stream: {path}")
    if frame["datetime"].isna().any():
        raise RuntimeError(f"invalid timestamps in {path}")
    if frame["datetime"].duplicated().any():
        raise RuntimeError(f"duplicate timestamps in {path}")
    if not frame["datetime"].is_monotonic_increasing:
        raise RuntimeError(f"timestamps are not strictly ordered in {path}")
    values = frame[OHLCV_COLS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite source OHLCV in {path}")
    o, h, low, c, volume = values.T
    if ((h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low)).any():
        raise RuntimeError(f"invalid OHLC relationships in {path}")
    if (volume < 0).any():
        raise RuntimeError(f"negative volume in {path}")
    if (np.asarray([o, h, low, c]) <= 0).any():
        raise RuntimeError(f"non-positive price in {path}")


def _validation_windows(
        matrix: np.ndarray,
        close_times: pd.DatetimeIndex,
        *,
        validation_start: int,
        context_length: int,
        prediction_length: int,
        count: int,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[str, str], ...]]:
    """Return chronological validation tasks whose targets are wholly in VAL."""
    first_end = validation_start + prediction_length
    last_end = len(matrix)
    if first_end > last_end:
        raise RuntimeError("validation region is shorter than prediction_length")
    candidates = np.linspace(first_end, last_end, num=max(1, count), dtype=np.int64)
    ends = sorted(set(int(value) for value in candidates))
    windows, ranges = [], []
    for end in ends:
        start = max(0, end - context_length - prediction_length)
        target_start = end - prediction_length
        if target_start < validation_start:
            continue
        window = matrix[start:end].T.astype(np.float32, copy=True)
        target = window[:, -prediction_length:]
        context = window[:, :-prediction_length]
        # A fully missing variate cannot be scaled. Require evidence in both the
        # context and target for every OHLCV channel in this validation task.
        if ((np.isfinite(context).sum(axis=1) == 0).any()
                or (np.isfinite(target).sum(axis=1) == 0).any()):
            continue
        windows.append(window)
        ranges.append((
            close_times[target_start].isoformat(),
            close_times[end - 1].isoformat(),
        ))
    if not windows:
        raise RuntimeError(
            "no validation window contains context and target observations for every channel")
    return tuple(windows), tuple(ranges)


def prepare_multivariate(
        data_dir: str | Path,
        *,
        repo_root: str | Path | None = None,
        tickers: Sequence[str] = tuple(TICKERS_9),
        timeframe: str = TIMEFRAME,
        holdout_start: str = DEFAULT_HOLDOUT_START,
        val_frac: float = 0.10,
        context_length: int = 256,
        prediction_length: int = 32,
        validation_windows: int = 8,
        report_path: str | Path | None = None,
) -> Chronos2MultivariateData:
    """Build one aligned ``ticker x OHLCV`` Chronos-2 task for one timeframe.

    CSV timestamps are bar opens.  Alignment and split decisions use
    ``open + timeframe``, the first instant at which the complete bar is
    observable.  The aligned timeline is the union of close times; absent bars
    remain NaN and are never imputed.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    if context_length < 1 or prediction_length < 1:
        raise ValueError("context_length and prediction_length must be positive")
    tickers = tuple(str(ticker) for ticker in tickers)
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be a non-empty unique sequence")
    timeframe = str(timeframe)
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; "
            f"expected one of {tuple(TIMEFRAME_MINUTES)}")
    timeframe_delta = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])

    data_dir = Path(data_dir)
    root = Path(repo_root) if repo_root is not None else None
    provenance = seal_continuous_streams(
        data_dir, ((ticker, timeframe) for ticker in tickers), repo_root=root)

    aligned_frames = []
    source_rows = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}_{timeframe}.csv"
        frame = pd.read_csv(path, usecols=["datetime", *OHLCV_COLS])
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        _validate_stream(frame, ticker=ticker, path=path)
        close_time = pd.DatetimeIndex(frame["datetime"]) + timeframe_delta
        values = frame[OHLCV_COLS].astype(np.float32)
        values.index = close_time
        values.columns = [f"{ticker}.{column}" for column in OHLCV_COLS]
        aligned_frames.append(values)
        source_rows[ticker] = len(values)

    aligned = pd.concat(aligned_frames, axis=1, join="outer").sort_index()
    if aligned.index.has_duplicates or not aligned.index.is_monotonic_increasing:
        raise RuntimeError("aligned close-time index is not unique and ordered")
    channel_names = tuple(aligned.columns)
    expected_channels = tuple(
        f"{ticker}.{column}" for ticker in tickers for column in OHLCV_COLS)
    if channel_names != expected_channels:
        raise RuntimeError("aligned channel order changed unexpectedly")

    holdout = pd.Timestamp(holdout_start)
    holdout = holdout.tz_localize("UTC") if holdout.tzinfo is None else holdout.tz_convert("UTC")
    development = aligned.loc[aligned.index < holdout]
    if len(development) < context_length + 2 * prediction_length:
        raise RuntimeError("insufficient pre-holdout history for the requested contract")
    validation_start = int(np.floor(len(development) * (1.0 - val_frac)))
    if validation_start < context_length + prediction_length:
        raise RuntimeError("training split is too short for context plus target")

    matrix = development.to_numpy(np.float32)
    train = matrix[:validation_start].T.copy()
    if (np.isfinite(train).sum(axis=1) < context_length + prediction_length).any():
        raise RuntimeError("a training channel has insufficient finite history")
    validation, validation_ranges = _validation_windows(
        matrix,
        development.index,
        validation_start=validation_start,
        context_length=context_length,
        prediction_length=prediction_length,
        count=validation_windows,
    )

    identity_payload = {
        "schema": SCHEMA,
        "timeframe": timeframe,
        "tickers": list(tickers),
        "channels": list(channel_names),
        "alignment": "union_of_bar_close_times_no_fill",
        "holdout_start": holdout.isoformat(),
        "val_frac": val_frac,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "source_provenance": provenance,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    finite = np.isfinite(matrix)
    report = {
        **identity_payload,
        "identity_sha256": identity,
        "aligned_rows_pre_holdout": len(development),
        "aligned_close_start": development.index[0].isoformat(),
        "aligned_close_end": development.index[-1].isoformat(),
        "train_rows": train.shape[1],
        "train_close_end": development.index[validation_start - 1].isoformat(),
        "validation_rows": len(development) - validation_start,
        "validation_target_ranges": [list(value) for value in validation_ranges],
        "n_variates": len(channel_names),
        "source_rows": source_rows,
        "finite_fraction": float(finite.mean()),
        "finite_fraction_by_channel": {
            channel: float(finite[:, index].mean())
            for index, channel in enumerate(channel_names)
        },
    }
    if report_path is not None:
        _atomic_json(Path(report_path), report)
    return Chronos2MultivariateData(
        train=train,
        validation_matrix=matrix[validation_start:].T.copy(),
        validation=validation,
        channel_names=channel_names,
        train_close_times=development.index[:validation_start].to_numpy(),
        validation_target_ranges=validation_ranges,
        report=report,
    )


def prepare_3min_multivariate(
        data_dir: str | Path,
        **kwargs,
) -> Chronos2MultivariateData:
    """Backward-compatible three-minute entrypoint."""
    return prepare_multivariate(data_dir, timeframe=TIMEFRAME, **kwargs)


__all__ = [
    "Chronos2MultivariateData",
    "DEFAULT_HOLDOUT_START",
    "SCHEMA",
    "TIMEFRAME",
    "TIMEFRAME_MINUTES",
    "prepare_multivariate",
    "prepare_3min_multivariate",
]
