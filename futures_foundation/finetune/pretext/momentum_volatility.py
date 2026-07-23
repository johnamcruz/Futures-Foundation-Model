"""Causal, strategy-agnostic momentum-volatility transition objective.

Inputs end on the decision bar and contain raw OHLCV only.  Targets describe signed future
path efficiency, future completed-candle range expansion, and the full 2x2 momentum/volatility
transition matrix. The objective never consumes ATR, entries, stops, costs, R multiples, or
strategy labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import ForecastTask


MOMENTUM_VOLATILITY_SCHEMA = "causal_momentum_volatility_v2"
MV_TREND_EXPANSION = 0
MV_TREND_WEAKENING = 1
MV_NOISY_EXPANSION = 2
MV_COMPRESSION = 3
COUPLING_CLASSES = (
    "trend_expansion",
    "trend_weakening",
    "noisy_expansion",
    "compression",
)


@dataclass(frozen=True)
class MomentumVolatilityTargets:
    """One decision bar's multi-horizon, scale-normalized SSL targets."""

    momentum: np.ndarray
    volatility: np.ndarray
    coupling: np.ndarray
    valid: np.ndarray
    causal_scale: float
    past_momentum: float


def transition_class(
    momentum_strength: float,
    future_range_ratio: float,
    *,
    momentum_threshold: float = 0.5,
    expansion_threshold: float = 1.1,
) -> int:
    """Map future path persistence and range behavior to the complete 2x2 state matrix."""
    values = (momentum_strength, future_range_ratio)
    if not all(np.isfinite(value) for value in values):
        return -1
    if momentum_threshold <= 0 or expansion_threshold <= 0:
        raise ValueError("momentum and expansion thresholds must be positive")
    directional = abs(float(momentum_strength)) >= float(momentum_threshold)
    expanding = float(future_range_ratio) >= float(expansion_threshold)
    if directional and expanding:
        return MV_TREND_EXPANSION
    if directional:
        return MV_TREND_WEAKENING
    if expanding:
        return MV_NOISY_EXPANSION
    return MV_COMPRESSION


def momentum_volatility_targets(
    high,
    low,
    close,
    decision: int,
    *,
    horizons=(5, 10, 20, 25),
    scale_lookback: int = 64,
    momentum_lookback: int = 20,
    momentum_threshold: float = 0.5,
    expansion_threshold: float = 1.1,
) -> MomentumVolatilityTargets:
    """Build future-only targets normalized by completed bars available at ``decision``.

    Momentum is signed path efficiency: net future displacement divided by total absolute
    close-to-close travel. Unlike endpoint return, it distinguishes a persistent move from a
    round trip. Volatility is the log ratio between the median future candle range and the causal
    completed-candle range. Rows without complete finite history/future fail closed.
    """
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    if high.shape != low.shape or high.shape != close.shape or high.ndim != 1:
        raise ValueError("high, low, and close must be aligned one-dimensional arrays")
    horizons = tuple(int(value) for value in horizons)
    if (not horizons or any(value <= 0 for value in horizons)
            or any(right <= left for left, right in zip(horizons, horizons[1:]))):
        raise ValueError("horizons must be positive and strictly increasing")
    decision = int(decision)
    scale_lookback, momentum_lookback = int(scale_lookback), int(momentum_lookback)
    if scale_lookback < 2 or momentum_lookback < 1:
        raise ValueError("scale_lookback >= 2 and momentum_lookback >= 1 are required")

    count = len(horizons)
    momentum = np.full(count, np.nan, np.float32)
    volatility = np.full(count, np.nan, np.float32)
    coupling = np.full(count, -1, np.int8)
    valid = np.zeros(count, bool)
    history_start = decision - scale_lookback + 1
    momentum_start = decision - momentum_lookback
    if (history_start < 0 or momentum_start < 0 or decision >= len(close)
            or decision < 0):
        return MomentumVolatilityTargets(
            momentum, volatility, coupling, valid, float("nan"), float("nan"))
    past_ranges = high[history_start:decision + 1] - low[history_start:decision + 1]
    causal_scale = float(np.median(past_ranges))
    if (not np.isfinite(past_ranges).all() or not np.isfinite(causal_scale)
            or causal_scale <= np.finfo(np.float32).eps
            or not np.isfinite(close[[momentum_start, decision]]).all()):
        return MomentumVolatilityTargets(
            momentum, volatility, coupling, valid, float("nan"), float("nan"))
    past_momentum = float(
        (close[decision] - close[momentum_start]) / causal_scale)

    for index, horizon in enumerate(horizons):
        end = decision + horizon
        if end >= len(close):
            continue
        future_ranges = high[decision + 1:end + 1] - low[decision + 1:end + 1]
        if (not np.isfinite(future_ranges).all()
                or not np.isfinite(close[end])
                or (future_ranges <= 0).any()):
            continue
        path = np.concatenate(([close[decision]], close[decision + 1:end + 1]))
        steps = np.diff(path)
        path_length = float(np.abs(steps).sum())
        future_momentum = float(
            (close[end] - close[decision]) / max(path_length, np.finfo(np.float32).eps))
        range_ratio = float(np.median(future_ranges) / causal_scale)
        if not np.isfinite(range_ratio) or range_ratio <= 0:
            continue
        momentum[index] = future_momentum
        volatility[index] = float(np.log(range_ratio))
        coupling[index] = transition_class(
            abs(future_momentum),
            range_ratio,
            momentum_threshold=momentum_threshold,
            expansion_threshold=expansion_threshold,
        )
        valid[index] = coupling[index] >= 0
    return MomentumVolatilityTargets(
        momentum, volatility, coupling, valid, causal_scale, past_momentum)


class MomentumVolatilityTask(ForecastTask):
    name, trainer = "momentum_volatility", "train_ssl_momentum_volatility"

    @staticmethod
    def _history_metric(history_row, name):
        value = history_row.get(name)
        return None if value is None else float(value)

    @property
    def control_contract(self):
        return "momentum_volatility_transition_v2"

    def control_evidence(self, history_row, probe_res):
        return {
            metric: self._history_metric(history_row, metric)
            for metric in (
                "mv_momentum_corr",
                "mv_volatility_corr",
                "mv_transition_auc",
                "mv_transition_worst_auc",
            )
        }

    def compare_control_evidence(self, real, controls):
        metrics = (
            "mv_momentum_corr",
            "mv_volatility_corr",
            "mv_transition_auc",
            "mv_transition_worst_auc",
        )
        margins = {
            name: {
                metric: (
                    None
                    if real.get(metric) is None or row.get(metric) is None
                    else float(real[metric]) - float(row[metric])
                )
                for metric in metrics
            }
            for name, row in controls.items()
        }
        positive = (
            real.get("mv_momentum_corr") is not None
            and float(real["mv_momentum_corr"]) > 0
            and real.get("mv_volatility_corr") is not None
            and float(real["mv_volatility_corr"]) > 0
            and real.get("mv_transition_auc") is not None
            and float(real["mv_transition_auc"]) > 0.5
            and real.get("mv_transition_worst_auc") is not None
            and float(real["mv_transition_worst_auc"]) > 0.5
        )
        passed = bool(
            positive
            and controls
            and all(
                margin is not None and margin > 0
                for row in margins.values()
                for margin in row.values()
            )
        )
        return passed, margins, (margins.get("shuffle") or {}).get("mv_transition_auc")

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict["momentum_volatility_schema"] = MOMENTUM_VOLATILITY_SCHEMA
        verdict["pretext_note"] = (
            "Raw-OHLCV causal momentum/volatility transition encoding; the exported encoder must "
            "beat its immediate parent after every disposable task head is removed, while "
            "retaining parent capabilities and passing downstream temporal validation."
        )
        return verdict


__all__ = [
    "COUPLING_CLASSES",
    "MV_COMPRESSION",
    "MV_NOISY_EXPANSION",
    "MV_TREND_EXPANSION",
    "MV_TREND_WEAKENING",
    "MOMENTUM_VOLATILITY_SCHEMA",
    "MomentumVolatilityTargets",
    "MomentumVolatilityTask",
    "momentum_volatility_targets",
    "transition_class",
]
