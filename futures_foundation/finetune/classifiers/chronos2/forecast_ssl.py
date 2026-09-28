"""Chronos-2 forecast-direction SSL on authenticated OHLCV streams.

The frozen contract is ``docs/chronos2_forecast_direction_ssl.md``.  A0 scores
the parent's native quantile forecast, A1 continues the parent's LoRA on the
native pinball loss of the next 64 closes, and A2 adds a BCE on P(up) read from
the same quantile head.  Targets are future bars of the same stream, so the
objective is self-supervised and strategy-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from futures_foundation.data_provenance import seal_continuous_streams
from futures_foundation.finetune.ssl_data import OHLCV_COLS

from .multivariate import TIMEFRAME_MINUTES, _validate_stream
from .ssl_stages import (
    _adapter_state,
    _atomic_json,
    _chronos_base_identity,
    _load_trainable_adapter,
    _restore_adapter,
    _save_final,
    tree_sha256,
)


REPORT_SCHEMA = "ffm_chronos2_forecast_direction_ssl_v1"
EVAL_SCHEMA = "ffm_chronos2_forecast_eval_v1"
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
CONTEXT_LENGTH = 256
FORECAST_LENGTH = 64
HORIZONS = (5, 10, 20, 50)
CLOSE = 3
HOLDOUT_START = "2026-01-01T00:00:00+00:00"
# Decision-close start (inclusive) and target-end close end (exclusive).
PERIODS: dict[str, tuple[str | None, str]] = {
    "train": (None, "2025-07-14T00:00:00+00:00"),
    "select": ("2025-07-15T00:00:00+00:00", "2025-10-01T00:00:00+00:00"),
    "outer": ("2025-10-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
}
ARMS = ("a1", "a2", "a3")
BAR_FEATURES = ("close_location", "body", "upper_wick", "lower_wick",
                "scaled_return", "relative_volume")
BAR_LOOKBACK = 20


def uses_bar_features(arm: str) -> bool:
    """A3 = A2 plus past-only bar-structure inputs; A1/A2 read OHLCV only."""
    return arm == "a3"


@dataclass(frozen=True)
class Stream:
    """One authenticated stream: bar-close times (UTC ns) and raw OHLCV."""

    name: str
    close_ns: np.ndarray
    values: np.ndarray


def _utc_ns(value: str | None) -> int | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return int(stamp.value)


def load_streams(
        data_dir: str | Path,
        *,
        tickers: Sequence[str] = TICKERS,
        timeframes: Sequence[str] = TIMEFRAMES,
        holdout_start: str = HOLDOUT_START,
        repo_root: str | Path | None = None,
) -> tuple[dict[str, Stream], dict]:
    """Authenticate and load streams, dropping every bar that closes in the holdout."""
    data_dir = Path(data_dir)
    pairs = [(str(ticker), str(timeframe)) for ticker in tickers for timeframe in timeframes]
    if not pairs:
        raise ValueError("at least one stream is required")
    unsupported = {timeframe for _, timeframe in pairs} - set(TIMEFRAME_MINUTES)
    if unsupported:
        raise ValueError(f"unsupported timeframes: {sorted(unsupported)}")
    provenance = seal_continuous_streams(
        data_dir, pairs, repo_root=None if repo_root is None else Path(repo_root))
    holdout = _utc_ns(holdout_start)
    streams = {}
    for ticker, timeframe in pairs:
        path = data_dir / f"{ticker}_{timeframe}.csv"
        frame = pd.read_csv(path, usecols=["datetime", *OHLCV_COLS])
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        _validate_stream(frame, ticker=ticker, path=path)
        close = (pd.DatetimeIndex(frame["datetime"])
                 + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])).asi8
        keep = close < holdout
        name = f"{ticker}@{timeframe}"
        streams[name] = Stream(
            name=name,
            close_ns=np.ascontiguousarray(close[keep]),
            values=np.ascontiguousarray(
                frame[OHLCV_COLS].to_numpy(np.float64)[keep]),
        )
    return streams, provenance


def period_bounds(
        close_ns: np.ndarray,
        start: str | None,
        end: str,
        *,
        context_length: int = CONTEXT_LENGTH,
        forecast_length: int = FORECAST_LENGTH,
) -> tuple[int, int]:
    """Half-open anchor range [lo, hi) with a full context and an in-period target.

    Anchor ``t`` is the decision bar.  It needs bars ``t-context_length+1..t`` and
    ``close[t+forecast_length] < end`` so no target crosses the period boundary.
    """
    lo = context_length - 1
    if start is not None:
        lo = max(lo, int(np.searchsorted(close_ns, _utc_ns(start), side="left")))
    end_index = int(np.searchsorted(close_ns, _utc_ns(end), side="left"))
    hi = min(end_index, len(close_ns)) - forecast_length
    return lo, max(lo, hi)


def evenly_spaced(lo: int, hi: int, limit: int) -> np.ndarray:
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(lo, hi - 1, min(limit, hi - lo)).astype(np.int64))


def bar_structure(values: np.ndarray) -> np.ndarray:
    """How each completed bar traded, from that bar and earlier bars only [N, 6].

    Columns follow ``BAR_FEATURES``: close location in the range, body, upper and
    lower wick (all as fractions of the bar range; a zero-range bar is 0.5/0/0/0),
    the bar's log return scaled by the std of the previous 20 returns, and log
    volume relative to the median of the previous 20 bars.  Warmup rows without
    that history are NaN so Chronos masks them instead of seeing invented data.
    """
    values = np.asarray(values, np.float64)
    o, h, low, c, volume = values.T
    span = h - low
    flat = span <= 0
    safe = np.where(flat, 1.0, span)
    location = np.where(flat, 0.5, (c - low) / safe)
    body = np.where(flat, 0.0, (c - o) / safe)
    upper = np.where(flat, 0.0, (h - np.maximum(o, c)) / safe)
    lower = np.where(flat, 0.0, (np.minimum(o, c) - low) / safe)
    returns = pd.Series(np.log(c)).diff()
    sigma = returns.shift(1).rolling(BAR_LOOKBACK, min_periods=BAR_LOOKBACK).std(ddof=0)
    scaled = (returns / sigma.clip(lower=1e-12)).to_numpy()
    median = pd.Series(volume).shift(1).rolling(
        BAR_LOOKBACK, min_periods=BAR_LOOKBACK).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.log(np.maximum(volume, 1e-12) / np.maximum(median, 1e-12))
    relative = np.where(np.isnan(median), np.nan, np.clip(relative, -5.0, 5.0))
    return np.column_stack([location, body, upper, lower,
                            np.clip(scaled, -10.0, 10.0), relative])


def gather(
        values: np.ndarray,
        anchors: np.ndarray,
        *,
        features: np.ndarray | None = None,
        context_length: int = CONTEXT_LENGTH,
        forecast_length: int = FORECAST_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return context [B,5(+K),L], future close [B,H], and decision close [B].

    Optional ``features`` [N,K] (e.g. ``bar_structure``) are appended after the
    OHLCV series as past-only inputs over the same context bars.
    """
    anchors = np.asarray(anchors, np.int64)
    offsets = np.arange(-context_length + 1, forecast_length + 1, dtype=np.int64)
    block = values[anchors[:, None] + offsets[None, :]]
    context = block[:, :context_length].transpose(0, 2, 1)
    if features is not None:
        extra = features[anchors[:, None] + offsets[None, :context_length]]
        context = np.concatenate([context, extra.transpose(0, 2, 1)], axis=1)
    return (np.ascontiguousarray(context), block[:, context_length:, CLOSE],
            block[:, context_length - 1, CLOSE])


def direction_labels(
        future_close: np.ndarray,
        last_close: np.ndarray,
        horizons: Sequence[int] = HORIZONS,
) -> np.ndarray:
    """``close[t+h] > close[t]``; identical to Probe Atlas ``future_direction_h``."""
    return np.stack([
        future_close[:, h - 1] > last_close for h in horizons], axis=1)


# ----------------------------------------------------------------- torch objective

def quantile_cdf(quantiles, levels, x):
    """P(Y <= x) under the piecewise-linear CDF through sorted quantile knots.

    ``quantiles`` is [..., Q], ``x`` is [...].  Outside the outer knots the CDF
    is clamped to the outer levels.  Differentiable in the quantiles.
    """
    import torch

    knots = torch.sort(quantiles, dim=-1).values
    x = x.unsqueeze(-1).to(knots.dtype)
    count = knots.shape[-1]
    below = (knots < x).sum(dim=-1, keepdim=True)
    upper = below.clamp(1, count - 1)
    lower_knot = knots.gather(-1, upper - 1)
    upper_knot = knots.gather(-1, upper)
    levels = levels.to(knots.device, knots.dtype)
    lower_level, upper_level = levels[upper - 1], levels[upper]
    width = (upper_knot - lower_knot).clamp_min(1e-12)
    fraction = ((x - lower_knot) / width).clamp(0.0, 1.0)
    return (lower_level + (upper_level - lower_level) * fraction).squeeze(-1)


def p_up(close_quantiles, last_close, levels, horizons: Sequence[int] = HORIZONS):
    """P(close[t+h] > close[t]) per horizon from quantiles [B,Q,H] -> [B,K]."""
    import torch

    return torch.stack([
        1.0 - quantile_cdf(close_quantiles[:, :, h - 1], levels, last_close)
        for h in horizons
    ], dim=1)


def direction_loss(close_quantiles, last_close, future_close, levels,
                   horizons: Sequence[int] = HORIZONS):
    """Uniform-weight BCE of the model's own P(up) at each horizon, averaged."""
    import torch
    import torch.nn.functional as F

    probability = p_up(close_quantiles, last_close, levels, horizons).clamp(1e-4, 1 - 1e-4)
    target = torch.stack([
        (future_close[:, h - 1] > last_close) for h in horizons], dim=1).to(probability.dtype)
    return F.binary_cross_entropy(probability, target)


def forecast_close(model, context, future_close=None, *, forecast_length=FORECAST_LENGTH,
                   patch_size: int = 16):
    """Run one stream-group per window; return (close pinball or None, close quantiles).

    ``context`` is [B,5,L].  Only the close variate carries a target, so the
    native loss (a mean over all B*5 rows) is rescaled by 5 to a per-target mean.
    """
    import torch

    batch, channels, _ = context.shape
    if forecast_length % patch_size:
        raise ValueError("forecast_length must be a multiple of the output patch size")
    flat = context.reshape(batch * channels, -1)
    groups = torch.arange(batch, device=context.device).repeat_interleave(channels)
    target = None
    if future_close is not None:
        target = torch.full((batch, channels, forecast_length), float("nan"),
                            device=context.device, dtype=context.dtype)
        target[:, CLOSE] = future_close
        target = target.reshape(batch * channels, forecast_length)
    output = model(context=flat, group_ids=groups,
                   num_output_patches=forecast_length // patch_size,
                   future_target=target)
    quantiles = output.quantile_preds.reshape(
        batch, channels, output.quantile_preds.shape[1], -1)[:, CLOSE, :, :forecast_length]
    loss = None if output.loss is None else output.loss * channels
    return loss, quantiles


# ----------------------------------------------------------------- numpy metrics

def auc_with_se(labels: np.ndarray, scores: np.ndarray, n_effective: float) -> tuple[float, float]:
    """ROC AUC and a Hanley-McNeil SE computed on effective (non-overlapping) counts."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, scores))
    scale = min(1.0, float(n_effective) / len(labels))
    n_pos, n_neg = max(positives * scale, 1.0), max(negatives * scale, 1.0)
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
                + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return auc, float(math.sqrt(max(variance, 0.0)))


def pinball(y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per-row mean over levels of 2*|(y-q)(1{y<=q}-tau)|; quantiles [n,Q]."""
    error = y[:, None] - quantiles
    return (2 * np.abs(error * ((error <= 0).astype(float) - levels[None, :]))).mean(axis=1)


def causal_features(stream: Stream, anchors: np.ndarray) -> np.ndarray:
    """Baseline features from bars <= t only: scaled returns, vol, range ratio, hour."""
    anchors = np.asarray(anchors, np.int64)
    if anchors.min(initial=10**9) < 50:
        raise ValueError("causal features need 50 bars of history")
    idx = anchors[:, None] + np.arange(-50, 1)[None, :]
    close = np.log(stream.values[:, CLOSE])[idx]
    high, low = stream.values[:, 1][idx], stream.values[:, 2][idx]
    sigma = np.diff(close[:, -21:], axis=1).std(axis=1) + 1e-12
    returns = [(close[:, -1] - close[:, -1 - k]) / sigma for k in (1, 5, 20, 50)]
    recent = high[:, -20:].max(1) - low[:, -20:].min(1)
    prior = high[:, -40:-20].max(1) - low[:, -40:-20].min(1)
    hours = pd.DatetimeIndex(stream.close_ns[anchors], tz="UTC")
    hour = (hours.hour + hours.minute / 60.0).to_numpy() * (2 * np.pi / 24)
    return np.column_stack([
        *returns, np.log(sigma), np.log((recent + 1e-12) / (prior + 1e-12)),
        np.sin(hour), np.cos(hour)])


def causal_baseline_scores(stream: Stream, train_anchors: np.ndarray, eval_anchors: np.ndarray,
                           horizons: Sequence[int] = HORIZONS) -> np.ndarray:
    """Per-horizon logistic regression fit on train anchors; returns eval P(up) [n,K]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = causal_features(stream, train_anchors)
    x_eval = causal_features(stream, eval_anchors)
    close = stream.values[:, CLOSE]
    scores = []
    for h in horizons:
        y = close[train_anchors + h] > close[train_anchors]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x_train, y)
        scores.append(model.predict_proba(x_eval)[:, 1])
    return np.column_stack(scores)


def score_stream(close_q: np.ndarray, last_close: np.ndarray, future_at_h: np.ndarray,
                 anchors: np.ndarray, levels: np.ndarray, baseline: np.ndarray,
                 p_up_values: np.ndarray, horizons: Sequence[int] = HORIZONS,
                 seed: int = 0) -> dict:
    """Metrics for one stream; close_q [n,Q,K], future_at_h [n,K], p_up_values [n,K]."""
    rng = np.random.default_rng(seed)
    low, high = int(np.argmin(np.abs(levels - 0.1))), int(np.argmin(np.abs(levels - 0.9)))
    median = int(np.argmin(np.abs(levels - 0.5)))
    rows = {}
    span = int(anchors[-1] - anchors[0]) if len(anchors) > 1 else 0
    for k, h in enumerate(horizons):
        y = future_at_h[:, k] > last_close
        n_eff = min(len(y), span // h + 1)
        auc, se = auc_with_se(y, p_up_values[:, k], n_eff)
        shuffled, _ = auc_with_se(rng.permutation(y), p_up_values[:, k], n_eff)
        base_auc, base_se = auc_with_se(y, baseline[:, k], n_eff)
        flat = np.repeat(last_close[:, None], len(levels), axis=1)
        model_loss = pinball(future_at_h[:, k], close_q[:, :, k], levels).sum()
        flat_loss = pinball(future_at_h[:, k], flat, levels).sum()
        rows[f"h{h}"] = {
            "n": int(len(y)),
            "n_effective": int(n_eff),
            "base_rate": float(y.mean()),
            "auc": auc,
            "auc_se": se,
            "median_sign_accuracy": float(
                ((close_q[:, median, k] > last_close) == y).mean()),
            "brier": float(np.mean((p_up_values[:, k] - y) ** 2)),
            "shuffled_auc": shuffled,
            "baseline_auc": base_auc,
            "baseline_auc_se": base_se,
            "scaled_wql": float(model_loss / max(flat_loss, 1e-12)),
            "coverage_80": float(np.mean(
                (close_q[:, low, k] <= future_at_h[:, k])
                & (future_at_h[:, k] <= close_q[:, high, k]))),
        }
    return rows


def summarize(per_stream: Mapping[str, Mapping[str, Mapping]],
              horizons: Sequence[int] = HORIZONS) -> dict:
    summary = {}
    for h in horizons:
        key = f"h{h}"
        rows = [value[key] for value in per_stream.values()]
        auc = np.array([row["auc"] for row in rows], float)
        base = np.array([row["baseline_auc"] for row in rows], float)
        summary[key] = {
            "streams": len(rows),
            "mean_auc": float(np.nanmean(auc)),
            "min_auc": float(np.nanmin(auc)),
            "streams_auc_above_half": int(np.sum(auc > 0.5)),
            "mean_baseline_auc": float(np.nanmean(base)),
            "mean_shuffled_auc": float(np.nanmean([row["shuffled_auc"] for row in rows])),
            "mean_scaled_wql": float(np.mean([row["scaled_wql"] for row in rows])),
            "mean_coverage_80": float(np.mean([row["coverage_80"] for row in rows])),
        }
    return summary


# ----------------------------------------------------------------- model scoring

def _levels(model) -> np.ndarray:
    return np.asarray(model.chronos_config.quantiles, dtype=np.float64)


def predict_period(model, stream: Stream, anchors: np.ndarray, *, device: str,
                   batch_windows: int = 256, horizons: Sequence[int] = HORIZONS,
                   features: np.ndarray | None = None):
    """Close quantiles at each horizon [n,Q,K], P(up) [n,K], decision and future closes."""
    import torch

    levels = torch.tensor(_levels(model), dtype=torch.float32)
    quantiles, probabilities, last, future = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(anchors), batch_windows):
            chunk = anchors[start:start + batch_windows]
            context, future_close, last_close = gather(stream.values, chunk, features=features)
            _, close_q = forecast_close(
                model, torch.from_numpy(context.astype(np.float32)).to(device))
            last_t = torch.from_numpy(last_close.astype(np.float32)).to(device)
            probabilities.append(p_up(close_q, last_t, levels, horizons).cpu().numpy())
            index = torch.tensor([h - 1 for h in horizons], device=close_q.device)
            quantiles.append(close_q.index_select(-1, index).cpu().double().numpy())
            last.append(last_close)
            future.append(future_close[:, [h - 1 for h in horizons]])
    return (np.concatenate(quantiles), np.concatenate(probabilities),
            np.concatenate(last), np.concatenate(future))


def evaluate(model, streams: Mapping[str, Stream], period: str, *, device: str,
             anchors_per_stream: int = 2000, baseline_train_per_stream: int = 3000,
             batch_windows: int = 256, horizons: Sequence[int] = HORIZONS,
             bar_features: bool = False) -> dict:
    """Score one model on one period, per stream × horizon, with controls."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}")
    levels = _levels(model)
    per_stream, started = {}, time.monotonic()
    model.eval()
    for index, (name, stream) in enumerate(sorted(streams.items())):
        lo, hi = period_bounds(stream.close_ns, *PERIODS[period])
        anchors = evenly_spaced(lo, hi, anchors_per_stream)
        train_lo, train_hi = period_bounds(stream.close_ns, *PERIODS["train"])
        train_anchors = evenly_spaced(max(train_lo, 50), train_hi, baseline_train_per_stream)
        if len(anchors) < 20 or len(train_anchors) < 100:
            raise RuntimeError(f"{name}: too few anchors in {period} or train")
        close_q, probability, last_close, future = predict_period(
            model, stream, anchors, device=device, batch_windows=batch_windows,
            horizons=horizons,
            features=bar_structure(stream.values) if bar_features else None)
        baseline = causal_baseline_scores(stream, train_anchors, anchors, horizons)
        per_stream[name] = score_stream(
            close_q, last_close, future, anchors, levels, baseline, probability,
            horizons, seed=index)
        print(f"[forecast-eval:{period}] {name} n={len(anchors)} "
              + " ".join(f"{key}={row['auc']:.4f}" for key, row in per_stream[name].items()),
              flush=True)
    return {
        "schema": EVAL_SCHEMA,
        "period": period,
        "period_bounds": PERIODS[period],
        "horizons": list(horizons),
        "anchors_per_stream": anchors_per_stream,
        "bar_features": bool(bar_features),
        "summary": summarize(per_stream, horizons),
        "per_stream": per_stream,
        "elapsed_seconds": time.monotonic() - started,
    }


# ----------------------------------------------------------------- gates

def forecast_gate(candidate: Mapping, reference: Mapping) -> dict:
    """A1 vs A0: scaled WQL lower on >=75% of streams, none worse by >5% relative."""
    result = {}
    for key in candidate["summary"]:
        ratios = {
            name: row[key]["scaled_wql"] / reference["per_stream"][name][key]["scaled_wql"]
            for name, row in candidate["per_stream"].items()}
        better = sum(value < 1.0 for value in ratios.values())
        result[key] = {
            "streams_better": better,
            "worst_ratio": max(ratios.values()),
            "pass": better >= math.ceil(0.75 * len(ratios)) and max(ratios.values()) <= 1.05,
        }
    return {"pass": all(row["pass"] for row in result.values()), "horizons": result}


def direction_gate(candidate: Mapping, reference: Mapping,
                   horizons: Sequence[str] = tuple(f"h{h}" for h in HORIZONS)) -> dict:
    """A2 vs A1 at every horizon: mean ΔAUC >= 0.005 with lower 95% bound > 0,
    >=2/3 streams up, beats causal baseline, and WQL no worse than reference + 2%."""
    result = {}
    for key in horizons:
        names = sorted(candidate["per_stream"])
        delta = np.array([
            candidate["per_stream"][name][key]["auc"]
            - reference["per_stream"][name][key]["auc"] for name in names])
        mean = float(delta.mean())
        lower = mean - 1.96 * float(delta.std(ddof=1)) / math.sqrt(len(delta))
        summary = candidate["summary"][key]
        wql = summary["mean_scaled_wql"] / reference["summary"][key]["mean_scaled_wql"]
        checks = {
            "mean_delta_auc": mean >= 0.005,
            "lower_bound": lower > 0.0,
            "streams_up": int((delta > 0).sum()) >= math.ceil(2 * len(delta) / 3),
            "beats_causal_baseline": summary["mean_auc"] > summary["mean_baseline_auc"],
            "wql_within_2pct": wql <= 1.02,
        }
        result[key] = {"mean_delta_auc": mean, "lower_95": lower,
                       "streams_up": int((delta > 0).sum()), "wql_ratio": wql,
                       "checks": checks, "pass": all(checks.values())}
    return {"pass": all(row["pass"] for row in result.values()), "horizons": result}


def _mean_over_horizons(report: Mapping, field: str) -> float:
    return float(np.mean([row[field] for row in report["summary"].values()]))


def select_direction_weight(a1_select: Mapping, candidates: Mapping[float, Mapping], *,
                            wql_budget: float = 1.02) -> dict:
    """Choose λ on the select period only: best mean AUC over all horizons among
    candidates whose mean scaled WQL stays within ``wql_budget`` x A1."""
    reports = [a1_select, *candidates.values()]
    if any(report.get("period") != "select" for report in reports):
        raise ValueError("λ selection may only read select-period reports")
    limit = _mean_over_horizons(a1_select, "mean_scaled_wql") * wql_budget
    scores = {float(weight): {"mean_auc": _mean_over_horizons(report, "mean_auc"),
                              "mean_scaled_wql": _mean_over_horizons(report, "mean_scaled_wql")}
              for weight, report in candidates.items()}
    eligible = sorted(weight for weight, row in scores.items()
                      if row["mean_scaled_wql"] <= limit)
    chosen = max(eligible, key=lambda weight: scores[weight]["mean_auc"]) if eligible else None
    return {"direction_weight": chosen, "eligible": eligible, "wql_limit": limit,
            "a1_mean_auc": _mean_over_horizons(a1_select, "mean_auc"), "scores": scores}


def _is_forward_direction_probe(name: str) -> bool:
    return name.startswith("pred_") and "direction" in name


def retention_gate(candidate_atlas: Mapping, parent_atlas: Mapping, *,
                   pooled_tolerance: float = 0.01, stream_tolerance: float = 0.03) -> dict:
    """Probe Atlas retention versus the parent for every non-target probe.

    Forward-direction probes are the objective and are excluded; expansion,
    volatility, trend and retention probes (including the in-window
    ``ret_structural_direction``) may not drop by more than the tolerances.
    """
    result = {}
    for name, parent in parent_atlas["probes"].items():
        if _is_forward_direction_probe(name):
            continue
        child = candidate_atlas["probes"][name]
        pooled_drop = parent["auc"] - child["auc"]
        stream_drop = max(
            parent["per_stream_auc"][stream] - child["per_stream_auc"][stream]
            for stream in parent["per_stream_auc"])
        result[name] = {
            "parent_auc": parent["auc"], "candidate_auc": child["auc"],
            "pooled_drop": pooled_drop, "worst_stream_drop": stream_drop,
            "pass": pooled_drop <= pooled_tolerance and stream_drop <= stream_tolerance,
        }
    return {"pass": all(row["pass"] for row in result.values()), "probes": result}


# ----------------------------------------------------------------- training

def build_stage_report(*, arm: str, direction_weight: float, seed: int, parent_path: str,
                       parent_sha256: str, checkpoint: Path, base_identity: Mapping,
                       provenance: Mapping, timeframes: Sequence[str], streams: Sequence[str],
                       code: Mapping | None, training: Mapping, history: list,
                       parent_select: Mapping, best_select: Mapping, best_epoch: int,
                       step_seconds: Sequence[float], elapsed_seconds: float) -> dict:
    """Completed-stage report in the shape Probe Atlas authenticates."""
    return {
        "schema": REPORT_SCHEMA,
        "stage": "forecast_direction",
        "status": "complete",
        "contract": "docs/chronos2_forecast_direction_ssl.md",
        "parent": {"path": parent_path, "sha256": parent_sha256},
        "checkpoint": {"path": str(checkpoint), "sha256": tree_sha256(checkpoint)},
        "data_identity_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        "streams": list(streams),
        "periods": PERIODS,
        "holdout_start": HOLDOUT_START,
        "code": code,
        "config": {
            "arm": arm, "direction_weight": direction_weight, "seed": seed,
            "timeframes": list(timeframes), "context_length": CONTEXT_LENGTH,
            "forecast_length": FORECAST_LENGTH, "base_model": dict(base_identity),
            **training,
        },
        "parent_select": parent_select,
        "best_select": best_select,
        "best_epoch": best_epoch,
        "improved_over_parent": best_epoch >= 0,
        "history": history,
        "mean_step_seconds": float(np.mean(step_seconds)) if len(step_seconds) else None,
        "elapsed_seconds": elapsed_seconds,
    }


def _git_identity(root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=False).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def train_forecast_direction(
        streams: Mapping[str, Stream],
        *,
        parent: str | Path,
        base_snapshot: str | Path,
        out_dir: str | Path,
        provenance: Mapping,
        arm: str,
        direction_weight: float = 0.0,
        device: str = "mps",
        seed: int = 0,
        epochs: int = 20,
        steps_per_epoch: int = 100,
        batch_windows: int = 32,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        patience: int = 3,
        select_anchors_per_stream: int = 100,
        eval_batch_windows: int = 256,
        repo_root: str | Path | None = None,
        horizons: Sequence[int] = HORIZONS,
) -> dict:
    """Continue the parent's LoRA on native pinball (A1) or pinball + λ·direction BCE (A2)."""
    import torch

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if (arm == "a1") != (direction_weight == 0.0) or direction_weight < 0.0:
        raise ValueError("A1 requires direction_weight=0; A2/A3 require direction_weight>0")
    out_dir, parent = Path(out_dir), Path(parent)
    checkpoint = out_dir / "checkpoint"
    if checkpoint.exists():
        raise RuntimeError(f"completed checkpoint already exists: {checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_identity = _chronos_base_identity(parent, Path(base_snapshot))
    parent_sha = tree_sha256(parent)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, base = _load_trainable_adapter(
        parent, device, base_revision=base_identity["revision"],
        base_snapshot=base_snapshot)
    levels = torch.tensor(_levels(base), dtype=torch.float32, device=device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    names = sorted(streams)
    features = ({name: bar_structure(streams[name].values) for name in names}
                if uses_bar_features(arm) else {name: None for name in names})
    train_bounds = {name: period_bounds(streams[name].close_ns, *PERIODS["train"])
                    for name in names}
    if any(hi - lo < 1000 for lo, hi in train_bounds.values()):
        raise RuntimeError("a stream has fewer than 1000 train anchors")
    select_anchors = {
        name: evenly_spaced(*period_bounds(streams[name].close_ns, *PERIODS["select"]),
                            select_anchors_per_stream) for name in names}

    def batch_loss(context, future_close, last_close):
        native, close_q = forecast_close(base, context, future_close)
        direction = direction_loss(close_q, last_close, future_close, levels, horizons)
        return native, direction

    def tensors(values):
        return tuple(torch.from_numpy(np.asarray(value, np.float32)).to(device)
                     for value in values)

    def selection_metric() -> dict:
        model.eval()
        native_by_stream, direction_by_stream = [], []
        with torch.no_grad():
            for name in names:
                anchors = select_anchors[name]
                native_sum, direction_sum = 0.0, 0.0
                for start in range(0, len(anchors), eval_batch_windows):
                    chunk = anchors[start:start + eval_batch_windows]
                    native, direction = batch_loss(*tensors(gather(
                        streams[name].values, chunk, features=features[name])))
                    native_sum += float(native) * len(chunk)
                    direction_sum += float(direction) * len(chunk)
                native_by_stream.append(native_sum / len(anchors))
                direction_by_stream.append(direction_sum / len(anchors))
        model.train()
        native, direction = float(np.mean(native_by_stream)), float(np.mean(direction_by_stream))
        return {"native": native, "direction_bce": direction,
                "objective": native + direction_weight * direction}

    started = time.monotonic()
    parent_metric = selection_metric()
    best_metric, best_adapter, best_epoch = parent_metric, _adapter_state(model), -1
    history = [{"epoch": -1, "select": parent_metric, "improved": False}]
    print(f"[forecast-ssl:{arm}] parent select={json.dumps(parent_metric)}", flush=True)
    bad, step_seconds = 0, []
    for epoch in range(epochs):
        model.train()
        totals = {"native": 0.0, "direction_bce": 0.0}
        for _ in range(steps_per_epoch):
            tick = time.monotonic()
            picks = rng.integers(len(names), size=batch_windows)
            contexts, futures, lasts = [], [], []
            for pick in np.unique(picks):
                name = names[int(pick)]
                lo, hi = train_bounds[name]
                anchors = rng.integers(lo, hi, size=int((picks == pick).sum()))
                context, future_close, last_close = gather(
                    streams[name].values, anchors, features=features[name])
                contexts.append(context)
                futures.append(future_close)
                lasts.append(last_close)
            context, future_close, last_close = tensors((
                np.concatenate(contexts), np.concatenate(futures), np.concatenate(lasts)))
            native, direction = batch_loss(context, future_close, last_close)
            loss = native + direction_weight * direction
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite forecast training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["native"] += float(native.detach())
            totals["direction_bce"] += float(direction.detach())
            step_seconds.append(time.monotonic() - tick)
        metric = selection_metric()
        improved = metric["objective"] < best_metric["objective"] - 1e-6
        if improved:
            best_metric, best_adapter, best_epoch, bad = metric, _adapter_state(model), epoch, 0
        else:
            bad += 1
        row = {"epoch": epoch,
               "train": {key: value / steps_per_epoch for key, value in totals.items()},
               "select": metric, "improved": improved}
        history.append(row)
        print(f"[forecast-ssl:{arm}] ep={epoch} train_native={row['train']['native']:.5f} "
              f"train_dir={row['train']['direction_bce']:.5f} "
              f"select={metric['objective']:.5f} native={metric['native']:.5f} "
              f"dir={metric['direction_bce']:.5f}{' *' if improved else ''} "
              f"step={np.mean(step_seconds[-steps_per_epoch:]):.3f}s", flush=True)
        if bad >= patience:
            break
    _restore_adapter(model, best_adapter)
    _save_final(model, checkpoint)
    report = build_stage_report(
        arm=arm, direction_weight=direction_weight, seed=seed, parent_path=str(parent),
        parent_sha256=parent_sha, checkpoint=checkpoint, base_identity=base_identity,
        provenance=provenance,
        timeframes=tuple(dict.fromkeys(name.split("@", 1)[1] for name in names)),
        streams=names,
        code=_git_identity(Path(repo_root)) if repo_root is not None else None,
        training={
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "batch_windows": batch_windows, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "patience": patience,
            "select_anchors_per_stream": select_anchors_per_stream,
            "horizons": list(horizons), "device": device,
            "bar_features": uses_bar_features(arm),
            "bar_feature_names": list(BAR_FEATURES) if uses_bar_features(arm) else [],
        },
        history=history, parent_select=parent_metric, best_select=best_metric,
        best_epoch=best_epoch, step_seconds=step_seconds,
        elapsed_seconds=time.monotonic() - started)
    _atomic_json(out_dir / "report.json", report)
    return report


__all__ = [
    "CONTEXT_LENGTH", "FORECAST_LENGTH", "HORIZONS", "PERIODS", "Stream",
    "BAR_FEATURES", "bar_structure", "build_stage_report", "uses_bar_features", "causal_baseline_scores", "causal_features", "direction_gate", "direction_labels",
    "direction_loss", "evaluate", "evenly_spaced", "forecast_close", "forecast_gate",
    "retention_gate", "select_direction_weight",
    "gather", "load_streams", "p_up", "period_bounds", "quantile_cdf",
    "train_forecast_direction",
]
