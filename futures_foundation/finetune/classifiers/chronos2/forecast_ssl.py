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
ARMS = ("a1", "a2", "a3", "a8", "a9")
# A9: direction BCE placed directly on the consumer embedding (the OHLCV REG
# tokens, concatenated) so the gradient shapes what downstream models read.
REG_HORIZONS = (5, 10, 20)
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
DIRECTION_HEAD_POLICY = "training_only_teacher_discarded"
BAR_FEATURES = ("close_location", "body", "upper_wick", "lower_wick",
                "scaled_return", "relative_volume")
BAR_LOOKBACK = 20
# Evaluation memory bound: 256 five-series windows per pass.  Groups with extra
# input series (A3: 11 series) get proportionally fewer windows per pass.
MAX_SERIES_PER_PASS = 256 * 5


def uses_bar_features(arm: str) -> bool:
    """A3 adds past-only bar-structure inputs; the other arms read OHLCV only."""
    return arm == "a3"


def uses_direction_head(arm: str) -> bool:
    """A8/A9 teach the encoder through a training-only head (discarded after training).

    The head is a training-only teacher: it is discarded after training and
    never ships.  Direction is evaluated from the model itself (native quantile
    P(up) and Probe Atlas REG probes), so the stage stays SSL-only.
    """
    return arm in ("a8", "a9")


def uses_reg_teacher(arm: str) -> bool:
    """A9: A1 plus a training-only direction head on the consumer REG embedding
    (BCE on close[t+h] > close[t], ties masked, h in REG_HORIZONS)."""
    return arm == "a9"


def uses_mirror_contrastive(arm: str) -> bool:
    """A8 (plan R3'): A1 plus a training-only mirrored-future contrastive teacher."""
    return arm == "a8"


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
                   patch_size: int = 16, return_hidden: bool = False,
                   return_reg: bool = False, ohlcv_series: int = 5):
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
    handles = []
    if return_hidden:
        handles.append(model.output_patch_embedding.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("tokens", args[0])))
    if return_reg:
        handles.append(model.encoder.register_forward_hook(
            lambda _module, _args, out: captured.__setitem__("sequence", out.last_hidden_state)))
    try:
        output = model(context=flat, group_ids=groups,
                       num_output_patches=forecast_length // patch_size,
                       future_target=target)
    finally:
        for handle in handles:
            handle.remove()
    quantiles = output.quantile_preds.reshape(
        batch, channels, output.quantile_preds.shape[1], -1)[:, CLOSE, :, :forecast_length]
    loss = None if output.loss is None else output.loss * channels
    if return_reg:
        # sequence = [context patches | REG | forecast patches]; REG is the consumer token
        sequence = captured["sequence"]
        reg = sequence[:, sequence.shape[1] - 1 - forecast_length // patch_size]
        reg = reg.reshape(batch, channels, -1)[:, :ohlcv_series].reshape(batch, -1)
        return loss, quantiles, reg
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


def masked_direction_bce(logits, up, valid, *, focal_gamma: float = 0.0):
    """BCE on P(up at h) over the ``valid`` cells (e.g. rows whose close moved).

    ``focal_gamma`` > 0 turns it into a focal loss: each cell is scaled by
    (1 - p_true)^gamma, so confident correct calls fade and wrong-way calls
    dominate the gradient.
    """
    import torch
    import torch.nn.functional as F

    if not valid.any():
        return logits.sum() * 0.0
    selected, target = logits[valid], up[valid].to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(selected, target, reduction="none")
    if focal_gamma > 0:
        probability = torch.sigmoid(selected)
        p_true = torch.where(target > 0.5, probability, 1.0 - probability)
        losses = (1.0 - p_true).pow(focal_gamma) * losses
    return losses.mean()


def future_path(future_close, last_close, length: int | None = None):
    """Cumulative log move of the next bars, scaled to unit L1 norm (size-free) [B, L]."""
    import torch

    path = torch.log(future_close[:, :length] / last_close[:, None])
    return path / path.abs().sum(1, keepdim=True).clamp_min(1e-12)


def mirror_path(path):
    """Same path, opposite direction: identical magnitude and shape of moves."""
    return -path


def make_reg_teacher(d_model: int, horizons: Sequence[int] = REG_HORIZONS,
                     ohlcv_series: int = 5):
    """Training-only head: consumer REG embedding [B, 5*D] -> one logit per horizon."""
    import torch.nn as nn

    width = ohlcv_series * d_model
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(),
                         nn.Linear(128, len(horizons)))


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
                 seed: int = 0) -> dict:
    """Metrics for one stream; close_q [n,Q,K], future_at_h [n,K], p_up_values [n,K].

    Direction metrics use only rows whose close moved: an unchanged close is not
    "down", and counting ties as negatives rewards predicting ties (common on
    coarse-tick contracts such as ZN/ZB 1min).  The tie-inclusive AUC is kept as
    ``auc_with_ties`` for continuity; forecast metrics use every row.
    """
    rng = np.random.default_rng(seed)
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
        }
    return rows


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
        per_stream[name] = score_stream(
            close_q, last_close, future, anchors, levels, baseline, probability,
            horizons, seed=index)
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
                     horizons: Sequence[int] = REG_HORIZONS,
                     context_length: int = CONTEXT_LENGTH) -> dict:
    """What the consumer embedding knows about side, against causal features.

    Target on ``period`` rows: all-bar direction at each horizon (ties excluded).  Probes are fit on
    train-period rows pooled across streams.
    """
    import torch

    def collect(split, limit):
        rows = {"reg": [], "causal": [], "stream": [], "move": []}
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
        return {key: np.concatenate(value) for key, value in rows.items()}

    model.eval()
    train, evaluation = collect("train", train_per_stream), collect(period, eval_per_stream)
    report = {"schema": "ffm_chronos2_reg_probe_v1", "period": period,
              "horizons": list(horizons), "targets": {}}
    for k, h in enumerate(horizons):
        for target, train_rows, eval_rows in (
                ("all_bars", train["move"][:, k] != 0, evaluation["move"][:, k] != 0),):
            report["targets"][f"{target}_h{h}"] = {
                source: probe_side(train[source][train_rows], train["move"][train_rows, k] > 0,
                                   evaluation[source][eval_rows],
                                   evaluation["move"][eval_rows, k] > 0,
                                   evaluation["stream"][eval_rows])
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
        rows.append({
            "arm": arm, "passes": int(sum(checks.values())), "checks": checks,
            "mean_auc": {key: summary[key]["mean_auc"] for key in primary},
            "delta_vs_reference": deltas, "streams_up": ups, "streams": streams,
            "probe": {target: probe[target]["reg"]["mean_auc"] for target in probe},
            "wql_ratio": wql,
        })
    rows.sort(key=lambda row: (-row["passes"],
                               -np.mean(list(row["delta_vs_reference"].values()))))
    return rows


def objective_score(probe: Mapping, evaluation: Mapping, reference: Mapping, *,
                    horizons: Sequence[int] = REG_HORIZONS, wql_budget: float = 1.02,
                    penalty_weight: float = 1.0) -> dict:
    """Sweep objective on the select period: mean REG-embedding direction AUC over
    ``horizons`` (all bars, ties excluded), minus a penalty only when the mean
    scaled WQL exceeds ``wql_budget`` x the reference (A1)."""
    for report in (evaluation, reference):
        if report.get("period") != "select":
            raise ValueError("the sweep objective may only read select-period reports")
    aucs = [probe["targets"][f"all_bars_h{h}"]["reg"]["mean_auc"] for h in horizons]
    wql_ratio = (_mean_over_horizons(evaluation, "mean_scaled_wql")
                 / _mean_over_horizons(reference, "mean_scaled_wql"))
    penalty = penalty_weight * max(0.0, wql_ratio - wql_budget)
    direction = float(np.mean(aucs))
    return {"score": direction - penalty, "direction_auc": direction,
            "per_horizon_auc": dict(zip((f"h{h}" for h in horizons), aucs)),
            "wql_ratio": wql_ratio, "penalty": penalty}


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
            "direction_target": "close_above_decision_close",
            "reg_teacher": ({"ties": "masked",
                             "pooling": "OHLCV REG tokens concatenated (consumer embedding)"}
                            if uses_reg_teacher(arm) else None),
            "mirror_contrastive": ({"path_length": MIRROR_LENGTH, "dim": MIRROR_DIM,
                                    "logit_scale_init": MIRROR_LOGIT_SCALE_INIT,
                                    "logit_scale_cap": MIRROR_LOGIT_SCALE_CAP,
                                    "symmetric": True,
                                    "hard_negative_warmup_epochs": MIRROR_WARMUP_EPOCHS,
                                    "false_negatives": "same stream within path length",
                                    "negatives": "own mirrored future + in-batch futures"}
                                   if uses_mirror_contrastive(arm) else None),
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
        focal_gamma: float = 0.0,
        reg_horizons: Sequence[int] = REG_HORIZONS,
        epoch_callback=None,
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
        if uses_reg_teacher(arm):
            if max(reg_horizons) > FORECAST_LENGTH or min(reg_horizons) < 1:
                raise ValueError("reg_horizons must lie within the forecast length")
            teacher = make_reg_teacher(int(base.model_dim), tuple(reg_horizons)).to(device)
        elif uses_mirror_contrastive(arm):
            teacher = make_mirror_teacher(int(base.model_dim), FORECAST_LENGTH // 16).to(device)
        parameters += list(teacher.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    names = sorted(streams)
    features = ({name: bar_structure(streams[name].values) for name in names}
                if uses_bar_features(arm) else {name: None for name in names})
    train_bounds = {name: period_bounds(streams[name].close_ns, *PERIODS["train"],
                                        context_length=context_length)
                    for name in names}
    if any(hi - lo < 1000 for lo, hi in train_bounds.values()):
        raise RuntimeError("a stream has fewer than 1000 train anchors")
    select_anchors = {
        name: evenly_spaced(*period_bounds(streams[name].close_ns, *PERIODS["select"],
                                           context_length=context_length),
                            select_anchors_per_stream) for name in names}

    def batch_loss(context, future_close, last_close, event=None):
        """-> (native, direction, extra); objective = native + λ·direction + extra."""
        zero = torch.zeros((), device=context.device)
        if uses_reg_teacher(arm):
            native, _, reg = forecast_close(base, context, future_close, return_reg=True)
            ends = torch.stack([future_close[:, h - 1] for h in reg_horizons], 1)
            valid = ends != last_close[:, None]
            return (native, masked_direction_bce(
                teacher(reg), ends > last_close[:, None], valid,
                focal_gamma=focal_gamma), zero)
        if uses_mirror_contrastive(arm):
            native, _, hidden = forecast_close(base, context, future_close, return_hidden=True)
            path = future_path(future_close, last_close, MIRROR_LENGTH)
            moved = (future_close[:, :MIRROR_LENGTH] != last_close[:, None]).any(1)
            return native, mirror_contrastive_loss(
                teacher["context"](hidden), teacher["future"](path),
                teacher["future"](mirror_path(path)), valid=moved,
                logit_scale=teacher["scale"].value, symmetric=True,
                use_mirror=mirror_state["use_mirror"], false_negatives=event), zero
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

    def event_tensor(name, anchors):
        if uses_mirror_contrastive(arm):
            return overlap_tensor(np.zeros(len(anchors)), anchors)
        return None

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
                    native, direction, extra = batch_loss(*tensors(gather(
                        streams[name].values, chunk, features=features[name],
                        context_length=context_length)), event_tensor(name, chunk))
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
            contexts, futures, lasts, events = [], [], [], []
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
                row_streams.append(np.full(len(anchors), int(pick)))
                row_anchors.append(anchors)
            context, future_close, last_close = tensors((
                np.concatenate(contexts), np.concatenate(futures), np.concatenate(lasts)))
            event = (torch.from_numpy(np.concatenate(events)).to(device) if events else None)
            if uses_mirror_contrastive(arm):
                mirror_state["use_mirror"] = epoch >= MIRROR_WARMUP_EPOCHS
                event = overlap_tensor(np.concatenate(row_streams), np.concatenate(row_anchors))
            native, direction, extra = batch_loss(context, future_close, last_close, event)
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
        if epoch_callback is not None:
            epoch_callback(epoch, metric)            # may raise to stop (e.g. Optuna prune)
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
            "reg_horizons": list(reg_horizons) if uses_reg_teacher(arm) else None,
            "focal_gamma": float(focal_gamma) if uses_reg_teacher(arm) else None,
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
    "BAR_FEATURES", "DIRECTION_HEAD_POLICY", "MAX_SERIES_PER_PASS",
    "REG_HORIZONS", "future_path", "make_mirror_teacher",
    "make_reg_teacher", "uses_reg_teacher",
    "overlap_false_negatives", "mirror_contrastive_loss", "mirror_path",
    "uses_mirror_contrastive", "stack_gain", "stream_prediction_rows", "probe_side", "reg_embeddings", "reg_probe_report",
    "masked_direction_bce",
    "bar_structure",
    "make_direction_head", "uses_direction_head", "build_stage_report", "uses_bar_features", "causal_baseline_scores", "causal_features", "direction_gate", "direction_labels",
    "direction_loss", "evaluate", "evenly_spaced", "forecast_close", "forecast_gate",
    "objective_score", "rank_arms", "retention_gate", "select_direction_weight",
    "gather", "load_streams", "p_up", "period_bounds", "quantile_cdf",
    "train_forecast_direction",
]
