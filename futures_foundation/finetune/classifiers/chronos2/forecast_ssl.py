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
ARMS = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8")
# Mirrored-future contrastive direction: the context must pick its real future
# path over the same path sign-flipped.  The paths have identical magnitude, so
# only direction can separate them.
MIRROR_LENGTH = 20
MIRROR_DIM = 128
MIRROR_TEMPERATURE = 0.1
# CLM lessons: hard negatives after an in-batch warm-up (curriculum), a learned
# temperature capped at 100, a symmetric loss, and false-negative masking.
MIRROR_WARMUP_EPOCHS = 3
MIRROR_LOGIT_SCALE_INIT = float(np.log(1 / 0.07))
MIRROR_LOGIT_SCALE_CAP = 100.0
# Osler (2003, 2005): stops cluster beyond obvious levels; a close through one
# triggers them and the overshoot tends to revert.  Event bars are detected from
# bars <= t only; prior-day levels come from the completed previous RTH session.
LIQUIDITY_EVENTS = ("break_20_high", "break_20_low", "break_prior_day_high",
                    "break_prior_day_low", "crt_bull_breakout", "crt_bear_breakout")
LIQUIDITY_HORIZONS = (5, 10)
# Candle Range Theory: what each candle did to the previous candle's range.
CRT_CLASSES = ("inside", "bull_breakout", "bear_sweep", "bear_breakout", "bull_sweep",
               "outside")
CRT_STEPS = 5
CRT_WEIGHT = 1.0
DIRECTION_HEAD_POLICY = "training_only_teacher_discarded"
BAR_FEATURES = ("close_location", "body", "upper_wick", "lower_wick",
                "scaled_return", "relative_volume")
BAR_LOOKBACK = 20
EXPANSION_QUANTILE = 0.8
# Evaluation memory bound: 256 five-series windows per pass.  Groups with extra
# input series (A3/A6: 11 series) get proportionally fewer windows per pass.
MAX_SERIES_PER_PASS = 256 * 5
# Symmetric ±k·σ barriers per horizon, k ≈ 0.95·√H: the scaling shared by the
# downstream expansion labels (3.0σ in 10 bars on 3m, 5.35σ in 30 bars on 1m),
# expressed as a generic family rather than any one private label point.
FIRST_PASSAGE_BARRIERS = ((5, 2.1), (10, 3.0), (20, 4.2), (50, 6.7))
FIRST_PASSAGE_LOOKBACK = 128


def uses_bar_features(arm: str) -> bool:
    """A3/A6 add past-only bar-structure inputs; the other arms read OHLCV only."""
    return arm in ("a3", "a6")


def uses_direction_head(arm: str) -> bool:
    """A4 teaches the encoder through a direction head on its forecast tokens.

    The head is a training-only teacher: it is discarded after training and
    never ships.  Direction is evaluated from the model itself (native quantile
    P(up) and Probe Atlas REG probes), so the stage stays SSL-only.
    """
    return arm in ("a4", "a5", "a6", "a7", "a8")


def uses_liquidity_teacher(arm: str) -> bool:
    """A7 = A1 plus a training-only teacher that learns direction over the next
    5 and 10 bars only on liquidity-break bars (ties masked)."""
    return arm == "a7"


def uses_mirror_contrastive(arm: str) -> bool:
    """A8 (plan R3'): A1 plus a training-only mirrored-future contrastive teacher."""
    return arm == "a8"


def uses_crt_teacher(arm: str) -> bool:
    """A6 = A3 plus a training-only teacher that learns, candle by candle, what
    each of the next ``CRT_STEPS`` candles does to its predecessor's range."""
    return arm == "a6"


def uses_first_passage(arm: str) -> bool:
    """A5: the teacher learns P(hit) and P(up | hit) of symmetric ±k·σ barriers,
    i.e. which side a coming expansion breaks first.  Still SSL and discarded."""
    return arm == "a5"


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
                   horizons: Sequence[int] = HORIZONS, *, mask_ties: bool = False):
    """Uniform-weight BCE of the model's own P(up) at each horizon, averaged.

    ``mask_ties`` drops cells whose close is unchanged at the horizon, so the
    loss cannot be lowered by predicting ties (an unchanged close is not down).
    """
    import torch
    import torch.nn.functional as F

    probability = p_up(close_quantiles, last_close, levels, horizons).clamp(1e-4, 1 - 1e-4)
    ends = torch.stack([future_close[:, h - 1] for h in horizons], dim=1)
    target = (ends > last_close[:, None]).to(probability.dtype)
    if not mask_ties:
        return F.binary_cross_entropy(probability, target)
    moved = ends != last_close[:, None]
    if not moved.any():
        return probability.sum() * 0.0
    return F.binary_cross_entropy(probability[moved], target[moved])


def forecast_close(model, context, future_close=None, *, forecast_length=FORECAST_LENGTH,
                   patch_size: int = 16, return_hidden: bool = False):
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
    captured = {}
    handle = (model.output_patch_embedding.register_forward_pre_hook(
        lambda _module, args: captured.__setitem__("tokens", args[0]))
        if return_hidden else None)
    try:
        output = model(context=flat, group_ids=groups,
                       num_output_patches=forecast_length // patch_size,
                       future_target=target)
    finally:
        if handle is not None:
            handle.remove()
    quantiles = output.quantile_preds.reshape(
        batch, channels, output.quantile_preds.shape[1], -1)[:, CLOSE, :, :forecast_length]
    loss = None if output.loss is None else output.loss * channels
    if not return_hidden:
        return loss, quantiles
    tokens = captured["tokens"]
    hidden = tokens.reshape(batch, channels, tokens.shape[1], tokens.shape[2])[:, CLOSE]
    return loss, quantiles, hidden


def make_direction_head(d_model: int, n_patches: int, horizons: Sequence[int] = HORIZONS,
                        outputs_per_horizon: int = 1):
    """Training-only teacher: close-row forecast tokens [B,P,D] -> logits per horizon."""
    import torch.nn as nn

    width = d_model * n_patches
    return nn.Sequential(nn.Flatten(), nn.LayerNorm(width), nn.Linear(width, 128),
                         nn.GELU(), nn.Linear(128, outputs_per_horizon * len(horizons)))


def first_passage_targets(future_close, last_close, sigma, barriers=FIRST_PASSAGE_BARRIERS):
    """Torch twin of ``first_passage`` for a batch: (hit [B,K], up [B,K])."""
    import torch

    # log of the ratio, not a difference of logs: float32-safe (MPS has no float64)
    log_move = torch.log(future_close / last_close[:, None])
    hits, ups = [], []
    for horizon, k in barriers:
        progress = log_move[:, :horizon] / sigma[:, None]
        steps = torch.arange(1, horizon + 1, device=progress.device)
        never = torch.full_like(progress, horizon + 1, dtype=torch.long)
        first_up = torch.where(progress >= k, steps, never).min(1).values
        first_down = torch.where(progress <= -k, steps, never).min(1).values
        hits.append(torch.minimum(first_up, first_down) <= horizon)
        ups.append(first_up < first_down)
    return torch.stack(hits, 1), torch.stack(ups, 1)


def liquidity_teacher_loss(logits, up, valid):
    """BCE on P(up at h) over cells that are liquidity-event bars whose close moved."""
    import torch.nn.functional as F

    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], up[valid].to(logits.dtype))


def future_path(future_close, last_close, length: int | None = None):
    """Cumulative log move of the next bars, scaled to unit L1 norm (size-free) [B, L]."""
    import torch

    path = torch.log(future_close[:, :length] / last_close[:, None])
    return path / path.abs().sum(1, keepdim=True).clamp_min(1e-12)


def mirror_path(path):
    """Same path, opposite direction: identical magnitude and shape of moves."""
    return -path


def make_mirror_teacher(d_model: int, n_patches: int):
    """Training-only heads: forecast tokens -> context code; future path -> path
    code; plus a learned InfoNCE logit scale (CLM: init 1/0.07, capped at 100)."""
    import torch
    import torch.nn as nn

    class _LogitScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(MIRROR_LOGIT_SCALE_INIT))

    return nn.ModuleDict({
        "context": make_direction_head(d_model, n_patches, range(1),
                                       outputs_per_horizon=MIRROR_DIM),
        "future": nn.Sequential(nn.Linear(MIRROR_LENGTH, 128), nn.GELU(),
                                nn.Linear(128, MIRROR_DIM)),
        "scale": _LogitScale(),
    })


def overlap_false_negatives(streams: np.ndarray, anchors: np.ndarray,
                            window: int = MIRROR_LENGTH) -> np.ndarray:
    """[B,B] pairs from the same stream whose futures overlap: not true negatives."""
    streams, anchors = np.asarray(streams), np.asarray(anchors, np.int64)
    same = streams[:, None] == streams[None, :]
    near = np.abs(anchors[:, None] - anchors[None, :]) < window
    mask = same & near
    np.fill_diagonal(mask, False)
    return mask


def mirror_contrastive_loss(context_embedding, real_embedding, mirror_embedding, *,
                            temperature: float = MIRROR_TEMPERATURE, valid=None,
                            logit_scale=None, use_mirror: bool = True,
                            symmetric: bool = False, false_negatives=None):
    """InfoNCE: each context must score its real future above its own mirror
    (hard negative, optional for a curriculum) and other rows' real futures
    (in-batch negatives, minus masked false negatives).  ``symmetric`` adds the
    future->context direction.  ``logit_scale`` (log) overrides ``temperature``."""
    import torch
    import torch.nn.functional as F

    context = F.normalize(context_embedding, dim=-1)
    real = F.normalize(real_embedding, dim=-1)
    mirror = F.normalize(mirror_embedding, dim=-1)
    scale = (torch.exp(logit_scale).clamp(max=MIRROR_LOGIT_SCALE_CAP)
             if logit_scale is not None else 1.0 / temperature)
    in_batch = scale * (context @ real.T)                                # [B, B]
    if false_negatives is not None:
        in_batch = in_batch.masked_fill(false_negatives, float("-inf"))
    logits = in_batch
    if use_mirror:
        own_mirror = scale * (context * mirror).sum(-1, keepdim=True)    # [B, 1]
        logits = torch.cat([in_batch, own_mirror], dim=1)
    target = torch.arange(len(context), device=context.device)
    losses = F.cross_entropy(logits, target, reduction="none")
    if symmetric:
        losses = 0.5 * (losses + F.cross_entropy(in_batch.T, target, reduction="none"))
    if valid is not None:
        if not valid.any():
            return losses.sum() * 0.0
        return losses[valid].mean()
    return losses.mean()


def crt_teacher_loss(logits, targets):
    """Cross-entropy of the teacher's per-candle CRT class logits [B, steps*C]."""
    import torch.nn.functional as F

    count = len(CRT_CLASSES)
    return F.cross_entropy(logits.reshape(-1, count), targets.reshape(-1).long())


def first_passage_teacher_loss(logits, hit, up):
    """BCE on P(hit) plus BCE on P(up | hit) over rows where a barrier was hit."""
    import torch.nn.functional as F

    count = hit.shape[1]
    hit_logits, side_logits = logits[:, :count], logits[:, count:]
    loss = F.binary_cross_entropy_with_logits(hit_logits, hit.to(logits.dtype))
    if hit.any():
        loss = loss + F.binary_cross_entropy_with_logits(
            side_logits[hit], up[hit].to(logits.dtype))
    return loss


def head_direction_loss(logits, last_close, future_close, horizons: Sequence[int] = HORIZONS):
    """BCE-with-logits of the teacher head against ``close[t+h] > close[t]``."""
    import torch
    import torch.nn.functional as F

    target = torch.stack([
        (future_close[:, h - 1] > last_close) for h in horizons], dim=1).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target)


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


def true_range_scale(values: np.ndarray, lookback: int = FIRST_PASSAGE_LOOKBACK) -> np.ndarray:
    """Causal σ_t: trailing median (including bar t) of log true range; NaN in warmup."""
    values = np.asarray(values, np.float64)
    high, low, close = np.log(values[:, 1]), np.log(values[:, 2]), np.log(values[:, 3])
    previous = np.concatenate([[close[0]], close[:-1]])
    true_range = np.maximum.reduce([
        high - low, np.abs(high - previous), np.abs(low - previous)])
    true_range[0] = high[0] - low[0]
    return pd.Series(true_range).rolling(lookback, min_periods=lookback).median().to_numpy()


def first_passage(values: np.ndarray, anchors: np.ndarray, sigma: np.ndarray, *,
                  horizon: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which ±k·σ_t barrier a close in t+1..t+H reaches first.

    Returns (hit, up, first_offset): ``up`` is meaningful only where ``hit``;
    ``first_offset`` is 1..H, or 0 when neither barrier is reached.  Progress
    is (log close[t+j] - log close[t]) / σ_t with σ held fixed at the anchor.
    """
    anchors = np.asarray(anchors, np.int64)
    log_close = np.log(np.asarray(values, np.float64)[:, CLOSE])
    offsets = np.arange(1, horizon + 1)
    progress = ((log_close[anchors[:, None] + offsets[None, :]] - log_close[anchors][:, None])
                / sigma[anchors][:, None])
    never = horizon + 1
    first_up = np.where((progress >= k).any(1), (progress >= k).argmax(1) + 1, never)
    first_down = np.where((progress <= -k).any(1), (progress <= -k).argmax(1) + 1, never)
    hit = np.minimum(first_up, first_down) <= horizon
    return hit, first_up < first_down, np.where(hit, np.minimum(first_up, first_down), 0)


def candle_range_classes(values: np.ndarray) -> np.ndarray:
    """CRT class of every candle against the previous candle (index into CRT_CLASSES).

    Took only the prior high: bull breakout if it closed above it, else bear
    sweep.  Took only the prior low: bear breakout if it closed below it, else
    bull sweep.  Took both: outside.  Took neither (or first candle): inside.
    """
    values = np.asarray(values, np.float64)
    high, low, close = values[:, 1], values[:, 2], values[:, 3]
    classes = np.zeros(len(values), dtype=np.int64)
    if len(values) < 2:
        return classes
    prior_high, prior_low = high[:-1], low[:-1]
    took_high, took_low = high[1:] > prior_high, low[1:] < prior_low
    current = np.zeros(len(values) - 1, dtype=np.int64)
    only_high, only_low = took_high & ~took_low, took_low & ~took_high
    current[only_high & (close[1:] > prior_high)] = CRT_CLASSES.index("bull_breakout")
    current[only_high & (close[1:] <= prior_high)] = CRT_CLASSES.index("bear_sweep")
    current[only_low & (close[1:] < prior_low)] = CRT_CLASSES.index("bear_breakout")
    current[only_low & (close[1:] >= prior_low)] = CRT_CLASSES.index("bull_sweep")
    current[took_high & took_low] = CRT_CLASSES.index("outside")
    classes[1:] = current
    return classes


def liquidity_break_events(stream: Stream) -> np.ndarray:
    """Bool [N, len(LIQUIDITY_EVENTS)]: closes through a liquidity level at bar t.

    20-bar levels use bars t-20..t-1.  Prior-day levels are the previous
    trading day's RTH high/low (bars closing 09:31-16:00 ET; the trading day
    rolls at 18:00 ET), so they are complete before the current session opens.
    A prior-day break is the first close beyond the level (previous close was
    not beyond it).
    """
    values = np.asarray(stream.values, np.float64)
    high, low, close = values[:, 1], values[:, 2], values[:, 3]
    previous_close = np.concatenate([[close[0]], close[:-1]])
    high_20 = pd.Series(high).rolling(20, min_periods=20).max().shift(1).to_numpy()
    low_20 = pd.Series(low).rolling(20, min_periods=20).min().shift(1).to_numpy()
    et = pd.DatetimeIndex(stream.close_ns, tz="UTC").tz_convert("America/New_York")
    minutes = (et.hour * 60 + et.minute).to_numpy()
    regular = (minutes > 570) & (minutes <= 960)
    trading_day = (et.tz_localize(None) - np.timedelta64(18, "h")).normalize()
    frame = pd.DataFrame({"day": trading_day, "high": high, "low": low})
    session = frame[regular].groupby("day").agg(high=("high", "max"), low=("low", "min"))
    days = pd.Index(sorted(frame["day"].unique()))
    session = session.reindex(days).shift(1)            # previous trading day's RTH
    prior_high = pd.Series(trading_day).map(session["high"]).to_numpy(float)
    prior_low = pd.Series(trading_day).map(session["low"]).to_numpy(float)
    classes = candle_range_classes(values)
    with np.errstate(invalid="ignore"):
        events = np.column_stack([
            close > high_20,
            close < low_20,
            (close > prior_high) & (previous_close <= prior_high),
            (close < prior_low) & (previous_close >= prior_low),
            classes == CRT_CLASSES.index("bull_breakout"),
            classes == CRT_CLASSES.index("bear_breakout"),
        ])
    return events & np.isfinite(values).all(1)[:, None]


def crt_targets(classes: np.ndarray, anchors: np.ndarray, steps: int = CRT_STEPS) -> np.ndarray:
    """CRT classes of candles t+1..t+steps for each anchor t -> [B, steps]."""
    anchors = np.asarray(anchors, np.int64)
    return classes[anchors[:, None] + 1 + np.arange(steps)[None, :]]


def recent_volatility(stream: Stream, anchors: np.ndarray, lookback: int = BAR_LOOKBACK) -> np.ndarray:
    """Price-unit std of the last ``lookback`` one-bar moves ending at each anchor."""
    anchors = np.asarray(anchors, np.int64)
    idx = anchors[:, None] + np.arange(-lookback, 1)[None, :]
    close = np.log(stream.values[:, CLOSE])[idx]
    sigma = np.diff(close, axis=1).std(axis=1)
    return np.maximum(sigma * stream.values[anchors, CLOSE], 1e-12)


def causal_baseline_scores(stream: Stream, train_anchors: np.ndarray, eval_anchors: np.ndarray,
                           horizons: Sequence[int] = HORIZONS) -> np.ndarray:
    """Per-horizon logistic regression fit on train anchors whose close moved
    (an unchanged close is not direction); returns eval P(up) [n,K]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = causal_features(stream, train_anchors)
    x_eval = causal_features(stream, eval_anchors)
    close = stream.values[:, CLOSE]
    scores = []
    for h in horizons:
        move = close[train_anchors + h] - close[train_anchors]
        moved = move != 0
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x_train[moved], move[moved] > 0)
        scores.append(model.predict_proba(x_eval)[:, 1])
    return np.column_stack(scores)


def score_stream(close_q: np.ndarray, last_close: np.ndarray, future_at_h: np.ndarray,
                 anchors: np.ndarray, levels: np.ndarray, baseline: np.ndarray,
                 p_up_values: np.ndarray, horizons: Sequence[int] = HORIZONS,
                 seed: int = 0, range_scale: np.ndarray | None = None,
                 first_passage_labels: tuple[np.ndarray, np.ndarray] | None = None,
                 event_mask: np.ndarray | None = None) -> dict:
    """Metrics for one stream; close_q [n,Q,K], future_at_h [n,K], p_up_values [n,K].

    Direction metrics use only rows whose close moved: an unchanged close is not
    "down", and counting ties as negatives rewards predicting ties (common on
    coarse-tick contracts such as ZN/ZB 1min).  The tie-inclusive AUC is kept as
    ``auc_with_ties`` for continuity; forecast metrics use every row.
    """
    rng = np.random.default_rng(seed)
    scale = np.asarray(last_close if range_scale is None else range_scale, float)
    low, high = int(np.argmin(np.abs(levels - 0.1))), int(np.argmin(np.abs(levels - 0.9)))
    median = int(np.argmin(np.abs(levels - 0.5)))
    rows = {}
    span = int(anchors[-1] - anchors[0]) if len(anchors) > 1 else 0
    anchors = np.asarray(anchors)
    for k, h in enumerate(horizons):
        all_up = future_at_h[:, k] > last_close
        moved = future_at_h[:, k] != last_close
        y, p_k, base_k = all_up[moved], p_up_values[moved, k], baseline[moved, k]
        kept = anchors[moved]
        n_eff = min(len(y), (int(kept[-1] - kept[0]) // h + 1) if len(kept) > 1 else len(y))
        auc, se = auc_with_se(y, p_k, n_eff)
        shuffled, _ = auc_with_se(rng.permutation(y), p_k, n_eff)
        base_auc, base_se = auc_with_se(y, base_k, n_eff)
        auc_ties, _ = auc_with_se(all_up, p_up_values[:, k], min(len(all_up), span // h + 1))
        flat = np.repeat(last_close[:, None], len(levels), axis=1)
        model_loss = pinball(future_at_h[:, k], close_q[:, :, k], levels).sum()
        flat_loss = pinball(future_at_h[:, k], flat, levels).sum()
        rows[f"h{h}"] = {
            "n": int(len(y)),
            "n_effective": int(n_eff),
            "tie_rate": float(1.0 - moved.mean()),
            "base_rate": float(y.mean()) if len(y) else float("nan"),
            "auc": auc,
            "auc_se": se,
            "auc_with_ties": auc_ties,
            "stack": stack_gain(y, p_k, base_k),
            "median_sign_accuracy": float(
                ((close_q[moved, median, k] > last_close[moved]) == y).mean()),
            "brier": float(np.mean((p_k - y) ** 2)),
            "shuffled_auc": shuffled,
            "baseline_auc": base_auc,
            "baseline_auc_se": base_se,
            "scaled_wql": float(model_loss / max(flat_loss, 1e-12)),
            "coverage_80": float(np.mean(
                (close_q[:, low, k] <= future_at_h[:, k])
                & (future_at_h[:, k] <= close_q[:, high, k]))),
            "expansion_slice": _expansion_slice(
                ((close_q[:, high, k] - close_q[:, low, k]) / scale)[moved], y, p_k,
                base_k, kept, h),
        }
        if event_mask is not None:
            chosen = np.asarray(event_mask, bool)[moved]
            picked = kept[chosen]
            e_eff = min(int(chosen.sum()), (int(picked[-1] - picked[0]) // h + 1)
                        if len(picked) > 1 else int(chosen.sum()))
            e_auc, e_se = auc_with_se(y[chosen], p_k[chosen], e_eff)
            e_base, _ = auc_with_se(y[chosen], base_k[chosen], e_eff)
            rows[f"h{h}"]["event_slice"] = {
                "n": int(chosen.sum()), "n_effective": int(e_eff),
                "base_rate": float(y[chosen].mean()) if chosen.any() else float("nan"),
                "auc": e_auc, "auc_se": e_se, "baseline_auc": e_base}
        if first_passage_labels is not None:
            hit, up = first_passage_labels[0][:, k], first_passage_labels[1][:, k]
            picked = np.asarray(anchors)[hit]
            n_eff = min(int(hit.sum()), (int(picked[-1] - picked[0]) // h + 1)
                        if len(picked) > 1 else int(hit.sum()))
            side_auc, side_se = auc_with_se(up[hit], p_up_values[hit, k], n_eff)
            side_base, _ = auc_with_se(up[hit], baseline[hit, k], n_eff)
            rows[f"h{h}"]["first_passage_side"] = {
                "n": int(hit.sum()), "n_effective": int(n_eff),
                "hit_rate": float(hit.mean()),
                "up_rate": float(up[hit].mean()) if hit.any() else float("nan"),
                "auc": side_auc, "auc_se": side_se, "baseline_auc": side_base}
    return rows


def _expansion_slice(width: np.ndarray, y: np.ndarray, p_up_values: np.ndarray,
                     baseline: np.ndarray, anchors: np.ndarray, h: int) -> dict:
    """Direction on rows whose own predicted 10-90 range, relative to recent
    volatility, is in the stream's top quintile: where a consumer expects a big
    move and most needs the side.  Rows are chosen from the forecast and past
    bars only, never from the realized move."""
    chosen = width >= np.quantile(width, EXPANSION_QUANTILE)
    picked = np.asarray(anchors)[chosen]
    span = int(picked[-1] - picked[0]) if len(picked) > 1 else 0
    n_eff = min(int(chosen.sum()), span // h + 1)
    auc, se = auc_with_se(y[chosen], p_up_values[chosen], n_eff)
    base_auc, _ = auc_with_se(y[chosen], baseline[chosen], n_eff)
    return {"n": int(chosen.sum()), "n_effective": int(n_eff),
            "threshold_quantile": EXPANSION_QUANTILE, "base_rate": float(y[chosen].mean()),
            "auc": auc, "auc_se": se, "baseline_auc": base_auc}


def stream_prediction_rows(last_close: np.ndarray, future_at_h: np.ndarray,
                           p_up_values: np.ndarray, baseline: np.ndarray, anchors: np.ndarray,
                           horizons: Sequence[int] = HORIZONS) -> dict:
    """Per-row arrays for later analysis (stack test): no metric, just the rows."""
    return {"anchors": np.asarray(anchors, np.int64),
            "p_up": np.asarray(p_up_values, np.float32),
            "baseline": np.asarray(baseline, np.float32),
            "moved": future_at_h != last_close[:, None],
            "up": future_at_h > last_close[:, None],
            "horizons": np.asarray(horizons, np.int64)}


def stack_gain(y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray) -> dict:
    """Does the model add direction information beyond the causal baseline?

    Two contiguous time halves: a logistic stack of [logit p_model, logit p_base]
    is fit on one half and scored on the other (cross-fitted), and
    gain = AUC(stack) - AUC(baseline) on the same rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, bool)
    if y.all() or not y.any() or len(y) < 40:
        return {"gain": float("nan"), "stack_auc": float("nan"),
                "baseline_auc": float("nan"), "model_auc": float("nan"), "n": int(len(y))}

    def logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    x = np.column_stack([logit(p_model), logit(p_base)])
    half = len(y) // 2
    stacked = np.empty(len(y))
    for fit, score in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        if y[fit].all() or not y[fit].any():
            stacked[score] = x[score, 1]
            continue
        model = LogisticRegression(max_iter=1000).fit(x[fit], y[fit])
        stacked[score] = model.decision_function(x[score])
    base_auc = float(roc_auc_score(y, p_base))
    stack_auc = float(roc_auc_score(y, stacked))
    return {"gain": stack_auc - base_auc, "stack_auc": stack_auc, "baseline_auc": base_auc,
            "model_auc": float(roc_auc_score(y, p_model)), "n": int(len(y))}


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
        if all("stack" in row for row in rows):
            gains = np.array([row["stack"]["gain"] for row in rows], float)
            summary[key]["mean_stack_gain"] = float(np.nanmean(gains))
            summary[key]["streams_stack_gain_positive"] = int(np.sum(gains > 0))
        if all("expansion_slice" in row for row in rows):
            summary[key]["mean_expansion_auc"] = float(np.nanmean(
                [row["expansion_slice"]["auc"] for row in rows]))
            summary[key]["mean_expansion_baseline_auc"] = float(np.nanmean(
                [row["expansion_slice"]["baseline_auc"] for row in rows]))
        if all("event_slice" in row for row in rows):
            summary[key]["mean_event_auc"] = float(np.nanmean(
                [row["event_slice"]["auc"] for row in rows]))
            summary[key]["mean_event_baseline_auc"] = float(np.nanmean(
                [row["event_slice"]["baseline_auc"] for row in rows]))
        if all("first_passage_side" in row for row in rows):
            summary[key]["mean_first_passage_side_auc"] = float(np.nanmean(
                [row["first_passage_side"]["auc"] for row in rows]))
            summary[key]["mean_first_passage_side_baseline_auc"] = float(np.nanmean(
                [row["first_passage_side"]["baseline_auc"] for row in rows]))
    return summary


# ----------------------------------------------------------------- model scoring

def _levels(model) -> np.ndarray:
    return np.asarray(model.chronos_config.quantiles, dtype=np.float64)


def predict_period(model, stream: Stream, anchors: np.ndarray, *, device: str,
                   batch_windows: int = 256, horizons: Sequence[int] = HORIZONS,
                   features: np.ndarray | None = None,
                   context_length: int = CONTEXT_LENGTH):
    """Close quantiles at each horizon [n,Q,K], P(up) [n,K], decision and future closes."""
    import torch

    levels = torch.tensor(_levels(model), dtype=torch.float32)
    channels = 5 + (0 if features is None else features.shape[1])
    batch_windows = max(1, min(batch_windows, MAX_SERIES_PER_PASS // channels))
    quantiles, probabilities, last, future = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(anchors), batch_windows):
            chunk = anchors[start:start + batch_windows]
            context, future_close, last_close = gather(
                stream.values, chunk, features=features, context_length=context_length)
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
             bar_features: bool = False, context_length: int = CONTEXT_LENGTH) -> dict:
    """Score one model on one period, per stream × horizon, with controls."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}")
    levels = _levels(model)
    per_stream, prediction_rows, started = {}, {}, time.monotonic()
    model.eval()
    for index, (name, stream) in enumerate(sorted(streams.items())):
        lo, hi = period_bounds(stream.close_ns, *PERIODS[period], context_length=context_length)
        anchors = evenly_spaced(lo, hi, anchors_per_stream)
        train_lo, train_hi = period_bounds(stream.close_ns, *PERIODS["train"],
                                           context_length=context_length)
        train_anchors = evenly_spaced(max(train_lo, 50), train_hi, baseline_train_per_stream)
        if len(anchors) < 20 or len(train_anchors) < 100:
            raise RuntimeError(f"{name}: too few anchors in {period} or train")
        close_q, probability, last_close, future = predict_period(
            model, stream, anchors, device=device, batch_windows=batch_windows,
            horizons=horizons,
            features=bar_structure(stream.values) if bar_features else None,
            context_length=context_length)
        baseline = causal_baseline_scores(stream, train_anchors, anchors, horizons)
        sigma = true_range_scale(stream.values)
        passage = [first_passage(stream.values, anchors, sigma, horizon=h, k=k)
                   for h, k in FIRST_PASSAGE_BARRIERS if h in horizons]
        labels = ((np.column_stack([item[0] for item in passage]),
                   np.column_stack([item[1] for item in passage]))
                  if len(passage) == len(horizons) else None)
        per_stream[name] = score_stream(
            close_q, last_close, future, anchors, levels, baseline, probability,
            horizons, seed=index, range_scale=recent_volatility(stream, anchors),
            first_passage_labels=labels,
            event_mask=liquidity_break_events(stream)[anchors].any(1))
        prediction_rows[name] = stream_prediction_rows(
            last_close, future, probability, baseline, anchors, horizons)
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
        "context_length": int(context_length),
        "summary": summarize(per_stream, horizons),
        "per_stream": per_stream,
        "elapsed_seconds": time.monotonic() - started,
        "_rows": prediction_rows,
    }


# ----------------------------------------------------------------- REG probe (consumer view)

def reg_embeddings(model, context, ohlcv_series: int = 5):
    """Consumer embedding: the REG token of each OHLCV series, concatenated [B, 5*D].

    Extra input series (A3/A6) join the Chronos group but are not part of the
    consumer vector; a checkpoint that needs them changes the downstream input
    contract and must be handed off explicitly.
    """
    import torch

    batch, channels, length = context.shape
    flat = context.reshape(batch * channels, length)
    groups = torch.arange(batch, device=context.device).repeat_interleave(channels)
    outputs, _, _, context_patches = model.encode(
        context=flat, group_ids=groups, num_output_patches=1)
    reg = outputs[0][:, context_patches, :]
    return reg.reshape(batch, channels, -1)[:, :ohlcv_series].reshape(batch, -1)


def probe_side(x_train, y_train, x_eval, y_eval, streams, *, seed: int = 0,
               regularization: float = 0.1) -> dict:
    """Pooled linear probe fit on train rows, scored per stream on eval rows,
    with a random-side control (eval labels permuted within each stream)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=regularization, max_iter=2000))
    probe.fit(x_train, y_train)
    score = probe.predict_proba(x_eval)[:, 1]
    per_stream, shuffled = {}, []
    for name in sorted(set(streams)):
        mask = streams == name
        labels = y_eval[mask]
        if labels.all() or not labels.any():
            continue
        per_stream[str(name)] = float(roc_auc_score(labels, score[mask]))
        shuffled.append(float(roc_auc_score(rng.permutation(labels), score[mask])))
    values = np.array(list(per_stream.values()))
    return {"mean_auc": float(values.mean()), "streams_above_half": int((values > 0.5).sum()),
            "streams": len(values), "mean_shuffled_auc": float(np.mean(shuffled)),
            "n_train": int(len(y_train)), "n_eval": int(len(y_eval)),
            "per_stream_auc": per_stream}


def reg_probe_report(model, streams: Mapping[str, Stream], *, device: str,
                     bar_features: bool = False, train_per_stream: int = 1500,
                     eval_per_stream: int = 1500, period: str = "select",
                     horizons: Sequence[int] = LIQUIDITY_HORIZONS,
                     side_barrier: tuple[int, float] = (10, 3.0),
                     context_length: int = CONTEXT_LENGTH) -> dict:
    """What the consumer embedding knows about side, against causal features.

    Targets on ``period`` rows: all-bar direction (ties excluded), direction on
    liquidity-break bars, and ±kσ breakout side on hit rows.  Probes are fit on
    train-period rows pooled across streams.
    """
    import torch

    def collect(split, limit):
        rows = {"reg": [], "causal": [], "stream": [], "move": [], "event": [],
                "hit": [], "up_first": []}
        for name, stream in sorted(streams.items()):
            lo, hi = period_bounds(stream.close_ns, *PERIODS[split],
                                   context_length=context_length)
            anchors = evenly_spaced(max(lo, 260), hi, limit)
            features = bar_structure(stream.values) if bar_features else None
            channels = 5 + (0 if features is None else features.shape[1])
            step = max(1, MAX_SERIES_PER_PASS // channels)
            with torch.no_grad():
                for start in range(0, len(anchors), step):
                    chunk = anchors[start:start + step]
                    context, _, _ = gather(stream.values, chunk, features=features,
                                           context_length=context_length)
                    rows["reg"].append(reg_embeddings(
                        model, torch.from_numpy(context.astype(np.float32)).to(device)
                    ).float().cpu().numpy())
            close = stream.values[:, CLOSE]
            rows["causal"].append(causal_features(stream, anchors))
            rows["stream"].append(np.full(len(anchors), name))
            rows["move"].append(np.column_stack([close[anchors + h] - close[anchors]
                                                 for h in horizons]))
            rows["event"].append(liquidity_break_events(stream)[anchors].any(1))
            sigma = true_range_scale(stream.values)
            hit, up, _ = first_passage(stream.values, anchors, sigma,
                                       horizon=side_barrier[0], k=side_barrier[1])
            rows["hit"].append(hit)
            rows["up_first"].append(up)
        return {key: np.concatenate(value) for key, value in rows.items()}

    model.eval()
    train, evaluation = collect("train", train_per_stream), collect(period, eval_per_stream)
    report = {"schema": "ffm_chronos2_reg_probe_v1", "period": period,
              "horizons": list(horizons), "side_barrier": list(side_barrier), "targets": {}}
    for k, h in enumerate(horizons):
        for target, train_rows, eval_rows in (
                ("all_bars", train["move"][:, k] != 0, evaluation["move"][:, k] != 0),
                ("liquidity_break", (train["move"][:, k] != 0) & train["event"],
                 (evaluation["move"][:, k] != 0) & evaluation["event"])):
            report["targets"][f"{target}_h{h}"] = {
                source: probe_side(train[source][train_rows], train["move"][train_rows, k] > 0,
                                   evaluation[source][eval_rows],
                                   evaluation["move"][eval_rows, k] > 0,
                                   evaluation["stream"][eval_rows])
                for source in ("reg", "causal")}
    report["targets"][f"breakout_side_h{side_barrier[0]}"] = {
        source: probe_side(train[source][train["hit"]], train["up_first"][train["hit"]],
                           evaluation[source][evaluation["hit"]],
                           evaluation["up_first"][evaluation["hit"]],
                           evaluation["stream"][evaluation["hit"]])
        for source in ("reg", "causal")}
    return report


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


def rank_arms(evals: Mapping[str, Mapping], probes: Mapping[str, Mapping], *,
              reference: str = "a1", primary: Sequence[str] = ("h5", "h10"),
              min_delta: float = 0.005, wql_budget: float = 1.02) -> list[dict]:
    """Rank arms by the pre-registered, generic select-period checks (see contract).

    FFM is a strategy-agnostic market-context model, so only generic direction
    counts:
    1. direction_vs_reference: mean per-stream ΔAUC ≥ ``min_delta`` and more
       than 2/3 of streams up, at every ``primary`` horizon.
    2. beats_causal_baseline: mean AUC ≥ mean causal-baseline AUC at ``primary``.
    3. embedding_probe: the REG-embedding probe beats both the random-side
       control and the causal-feature probe on all-bar direction at h5 and h10.
    4. forecast_not_degraded: mean scaled WQL ≤ reference × ``wql_budget``.
    Liquidity-break and breakout-side slices are reported as diagnostics only.
    """
    ref = evals[reference]
    rows = []
    for arm, report in evals.items():
        if arm == reference:
            continue
        summary = report["summary"]
        deltas, ups = {}, {}
        for key in primary:
            names = sorted(report["per_stream"])
            delta = np.array([report["per_stream"][n][key]["auc"]
                              - ref["per_stream"][n][key]["auc"] for n in names])
            deltas[key], ups[key] = float(delta.mean()), int((delta > 0).sum())
        streams = len(report["per_stream"])
        slices = {
            "event_h5": (summary["h5"].get("mean_event_auc"),
                         summary["h5"].get("mean_event_baseline_auc")),
            "side_h10": (summary["h10"].get("mean_first_passage_side_auc"),
                         summary["h10"].get("mean_first_passage_side_baseline_auc")),
        }
        probe = probes.get(arm, {}).get("targets", {})
        probe_ok = bool(probe) and all(
            target in probe
            and probe[target]["reg"]["mean_auc"] > probe[target]["reg"]["mean_shuffled_auc"]
            and probe[target]["reg"]["mean_auc"] > probe[target]["causal"]["mean_auc"]
            for target in ("all_bars_h5", "all_bars_h10"))
        wql = _mean_over_horizons(report, "mean_scaled_wql") / _mean_over_horizons(
            ref, "mean_scaled_wql")
        checks = {
            "direction_vs_reference": all(
                deltas[key] >= min_delta and ups[key] > 2 * streams / 3 for key in primary),
            "beats_causal_baseline": all(
                summary[key]["mean_auc"] >= summary[key]["mean_baseline_auc"] for key in primary),
            "embedding_probe": probe_ok,
            "forecast_not_degraded": wql <= wql_budget,
        }
        diagnostics = {"slices_above_baseline": all(
            value is not None and base is not None and value > base
            for value, base in slices.values())}
        rows.append({
            "arm": arm, "passes": int(sum(checks.values())), "checks": checks,
            "diagnostics": diagnostics,
            "mean_auc": {key: summary[key]["mean_auc"] for key in primary},
            "delta_vs_reference": deltas, "streams_up": ups, "streams": streams,
            "slices": {key: value for key, (value, _) in slices.items()},
            "big_move_h5": summary["h5"].get("mean_expansion_auc"),
            "probe": {target: probe[target]["reg"]["mean_auc"] for target in probe},
            "wql_ratio": wql,
        })
    rows.sort(key=lambda row: (-row["passes"],
                               -np.mean(list(row["delta_vs_reference"].values()))))
    return rows


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
                       step_seconds: Sequence[float], elapsed_seconds: float,
                       context_length: int = CONTEXT_LENGTH) -> dict:
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
            "direction_head": DIRECTION_HEAD_POLICY if uses_direction_head(arm) else "none",
            "direction_target": ("first_passage_side" if uses_first_passage(arm)
                                 else "close_above_decision_close"),
            "crt_teacher": ({"steps": CRT_STEPS, "classes": list(CRT_CLASSES),
                             "weight": CRT_WEIGHT} if uses_crt_teacher(arm) else None),
            "mirror_contrastive": ({"path_length": MIRROR_LENGTH, "dim": MIRROR_DIM,
                                    "logit_scale_init": MIRROR_LOGIT_SCALE_INIT,
                                    "logit_scale_cap": MIRROR_LOGIT_SCALE_CAP,
                                    "symmetric": True,
                                    "hard_negative_warmup_epochs": MIRROR_WARMUP_EPOCHS,
                                    "false_negatives": "same stream within path length",
                                    "negatives": "own mirrored future + in-batch futures"}
                                   if uses_mirror_contrastive(arm) else None),
            "liquidity_teacher": ({"events": list(LIQUIDITY_EVENTS),
                                   "horizons": list(LIQUIDITY_HORIZONS), "ties": "masked"}
                                  if uses_liquidity_teacher(arm) else None),
            "first_passage_barriers": [list(item) for item in FIRST_PASSAGE_BARRIERS],
            "timeframes": list(timeframes), "context_length": int(context_length),
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
        mask_ties: bool = False,
        context_length: int = CONTEXT_LENGTH,
) -> dict:
    """Continue the parent's LoRA on native pinball (A1) or pinball + λ·direction BCE (A2)."""
    import torch

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if context_length <= 0 or context_length % 16:
        raise ValueError("context_length must be a positive multiple of the 16-bar patch")
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
    teacher = None
    if uses_direction_head(arm):
        if uses_mirror_contrastive(arm):
            teacher = make_mirror_teacher(int(base.model_dim), FORECAST_LENGTH // 16).to(device)
        elif uses_liquidity_teacher(arm):
            teacher = make_direction_head(
                int(base.model_dim), FORECAST_LENGTH // 16, LIQUIDITY_HORIZONS).to(device)
        elif uses_crt_teacher(arm):
            teacher = make_direction_head(
                int(base.model_dim), FORECAST_LENGTH // 16, range(CRT_STEPS),
                outputs_per_horizon=len(CRT_CLASSES)).to(device)
        else:
            teacher = make_direction_head(
                int(base.model_dim), FORECAST_LENGTH // 16, horizons,
                outputs_per_horizon=2 if uses_first_passage(arm) else 1).to(device)
        parameters += list(teacher.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    names = sorted(streams)
    features = ({name: bar_structure(streams[name].values) for name in names}
                if uses_bar_features(arm) else {name: None for name in names})
    sigmas = {name: true_range_scale(streams[name].values) for name in names}
    crts = ({name: candle_range_classes(streams[name].values) for name in names}
            if uses_crt_teacher(arm) else None)
    liquidity = ({name: liquidity_break_events(streams[name]).any(1) for name in names}
                 if uses_liquidity_teacher(arm) else None)
    train_bounds = {name: period_bounds(streams[name].close_ns, *PERIODS["train"],
                                        context_length=context_length)
                    for name in names}
    if any(hi - lo < 1000 for lo, hi in train_bounds.values()):
        raise RuntimeError("a stream has fewer than 1000 train anchors")
    select_anchors = {
        name: evenly_spaced(*period_bounds(streams[name].close_ns, *PERIODS["select"],
                                           context_length=context_length),
                            select_anchors_per_stream) for name in names}

    def batch_loss(context, future_close, last_close, sigma, crt=None, event=None):
        """-> (native, direction, extra); objective = native + λ·direction + extra."""
        zero = torch.zeros((), device=context.device)
        if uses_mirror_contrastive(arm):
            native, _, hidden = forecast_close(base, context, future_close, return_hidden=True)
            path = future_path(future_close, last_close, MIRROR_LENGTH)
            moved = (future_close[:, :MIRROR_LENGTH] != last_close[:, None]).any(1)
            return native, mirror_contrastive_loss(
                teacher["context"](hidden), teacher["future"](path),
                teacher["future"](mirror_path(path)), valid=moved,
                logit_scale=teacher["scale"].value, symmetric=True,
                use_mirror=mirror_state["use_mirror"], false_negatives=event), zero
        if uses_liquidity_teacher(arm):
            native, _, hidden = forecast_close(base, context, future_close, return_hidden=True)
            ends = torch.stack([future_close[:, h - 1] for h in LIQUIDITY_HORIZONS], 1)
            valid = event[:, None] & (ends != last_close[:, None])
            return (native, liquidity_teacher_loss(
                teacher(hidden), ends > last_close[:, None], valid), zero)
        if uses_crt_teacher(arm):
            native, close_q, hidden = forecast_close(
                base, context, future_close, return_hidden=True)
            direction = direction_loss(close_q, last_close, future_close, levels, horizons,
                                       mask_ties=mask_ties)
            return native, direction, CRT_WEIGHT * crt_teacher_loss(teacher(hidden), crt)
        if teacher is not None:
            native, _, hidden = forecast_close(base, context, future_close, return_hidden=True)
            if uses_first_passage(arm):
                hit, up = first_passage_targets(future_close, last_close, sigma)
                return native, first_passage_teacher_loss(teacher(hidden), hit, up), zero
            return (native, head_direction_loss(teacher(hidden), last_close, future_close,
                                                horizons), zero)
        native, close_q = forecast_close(base, context, future_close)
        direction = direction_loss(close_q, last_close, future_close, levels, horizons,
                                   mask_ties=mask_ties)
        return native, direction, zero

    mirror_state = {"use_mirror": True}      # selection always scores the full objective

    def overlap_tensor(streams_of_rows, anchors_of_rows):
        if not uses_mirror_contrastive(arm):
            return None
        return torch.from_numpy(overlap_false_negatives(
            streams_of_rows, anchors_of_rows)).to(device)

    def crt_tensor(name, anchors):
        if crts is None:
            return None
        return torch.from_numpy(crt_targets(crts[name], anchors)).to(device)

    def event_tensor(name, anchors):
        if uses_mirror_contrastive(arm):
            return overlap_tensor(np.zeros(len(anchors)), anchors)
        if liquidity is None:
            return None
        return torch.from_numpy(liquidity[name][anchors]).to(device)

    def tensors(values):
        return tuple(torch.from_numpy(np.asarray(value, np.float32)).to(device)
                     for value in values)

    def selection_metric() -> dict:
        model.eval()
        if teacher is not None:
            teacher.eval()
        native_by_stream, direction_by_stream, extra_by_stream = [], [], []
        with torch.no_grad():
            for name in names:
                anchors = select_anchors[name]
                native_sum, direction_sum, extra_sum = 0.0, 0.0, 0.0
                for start in range(0, len(anchors), eval_batch_windows):
                    chunk = anchors[start:start + eval_batch_windows]
                    native, direction, extra = batch_loss(*tensors((*gather(
                        streams[name].values, chunk, features=features[name],
                        context_length=context_length),
                        sigmas[name][chunk])), crt_tensor(name, chunk),
                        event_tensor(name, chunk))
                    native_sum += float(native) * len(chunk)
                    direction_sum += float(direction) * len(chunk)
                    extra_sum += float(extra) * len(chunk)
                native_by_stream.append(native_sum / len(anchors))
                direction_by_stream.append(direction_sum / len(anchors))
                extra_by_stream.append(extra_sum / len(anchors))
        model.train()
        if teacher is not None:
            teacher.train()
        native, direction = float(np.mean(native_by_stream)), float(np.mean(direction_by_stream))
        extra = float(np.mean(extra_by_stream))
        metric = {"native": native, "direction_bce": direction,
                  "objective": native + direction_weight * direction + extra}
        if uses_crt_teacher(arm):
            metric["crt_ce"] = extra / CRT_WEIGHT
        return metric

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
            contexts, futures, lasts, scales, candle_targets, events = [], [], [], [], [], []
            row_streams, row_anchors = [], []
            for pick in np.unique(picks):
                name = names[int(pick)]
                lo, hi = train_bounds[name]
                anchors = rng.integers(lo, hi, size=int((picks == pick).sum()))
                context, future_close, last_close = gather(
                    streams[name].values, anchors, features=features[name],
                    context_length=context_length)
                contexts.append(context)
                futures.append(future_close)
                lasts.append(last_close)
                scales.append(sigmas[name][anchors])
                if crts is not None:
                    candle_targets.append(crt_targets(crts[name], anchors))
                if liquidity is not None:
                    events.append(liquidity[name][anchors])
                row_streams.append(np.full(len(anchors), int(pick)))
                row_anchors.append(anchors)
            context, future_close, last_close, sigma = tensors((
                np.concatenate(contexts), np.concatenate(futures), np.concatenate(lasts),
                np.concatenate(scales)))
            crt = (torch.from_numpy(np.concatenate(candle_targets)).to(device)
                   if candle_targets else None)
            event = (torch.from_numpy(np.concatenate(events)).to(device) if events else None)
            if uses_mirror_contrastive(arm):
                mirror_state["use_mirror"] = epoch >= MIRROR_WARMUP_EPOCHS
                event = overlap_tensor(np.concatenate(row_streams), np.concatenate(row_anchors))
            native, direction, extra = batch_loss(
                context, future_close, last_close, sigma, crt, event)
            loss = native + direction_weight * direction + extra
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite forecast training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["native"] += float(native.detach())
            totals["direction_bce"] += float(direction.detach())
            step_seconds.append(time.monotonic() - tick)
        mirror_state["use_mirror"] = True
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
            "mask_ties": bool(mask_ties),
            "bar_feature_names": list(BAR_FEATURES) if uses_bar_features(arm) else [],
        },
        history=history, parent_select=parent_metric, best_select=best_metric,
        best_epoch=best_epoch, step_seconds=step_seconds,
        elapsed_seconds=time.monotonic() - started, context_length=context_length)
    _atomic_json(out_dir / "report.json", report)
    return report


__all__ = [
    "CONTEXT_LENGTH", "FORECAST_LENGTH", "HORIZONS", "PERIODS", "Stream",
    "BAR_FEATURES", "DIRECTION_HEAD_POLICY", "FIRST_PASSAGE_BARRIERS", "MAX_SERIES_PER_PASS", "first_passage",
    "CRT_CLASSES", "LIQUIDITY_EVENTS", "future_path", "make_mirror_teacher",
    "overlap_false_negatives", "mirror_contrastive_loss", "mirror_path",
    "uses_mirror_contrastive", "stack_gain", "stream_prediction_rows", "probe_side", "reg_embeddings", "reg_probe_report", "candle_range_classes", "liquidity_break_events",
    "liquidity_teacher_loss", "uses_liquidity_teacher", "crt_targets", "crt_teacher_loss", "uses_crt_teacher",
    "first_passage_targets", "first_passage_teacher_loss", "uses_first_passage",
    "true_range_scale", "bar_structure", "head_direction_loss",
    "make_direction_head", "recent_volatility", "uses_direction_head", "build_stage_report", "uses_bar_features", "causal_baseline_scores", "causal_features", "direction_gate", "direction_labels",
    "direction_loss", "evaluate", "evenly_spaced", "forecast_close", "forecast_gate",
    "rank_arms", "retention_gate", "select_direction_weight",
    "gather", "load_streams", "p_up", "period_bounds", "quantile_cdf",
    "train_forecast_direction",
]
