"""Causal, strategy-agnostic momentum-volatility coupling objective.

Inputs end on the decision bar and contain raw OHLCV only.  Targets describe signed future
displacement, future completed-candle range expansion, and whether that expansion continues,
reverses, or launches relative to the causal past direction.  The objective never consumes ATR,
entries, stops, costs, R multiples, or strategy labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import ForecastTask


MOMENTUM_VOLATILITY_SCHEMA = "causal_momentum_volatility_v1"
COUPLING_CHOP = 0
COUPLING_CONTINUATION = 1
COUPLING_REVERSAL = 2
COUPLING_LAUNCH = 3
COUPLING_CLASSES = (
    "directionless_or_contracting",
    "expanding_continuation",
    "expanding_reversal",
    "expanding_directional_launch",
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


def coupling_class(
    past_momentum: float,
    future_momentum: float,
    future_range_ratio: float,
    *,
    momentum_threshold: float = 0.5,
    expansion_threshold: float = 1.1,
) -> int:
    """Map causal past direction and future behavior to an auditable generic state."""
    values = (past_momentum, future_momentum, future_range_ratio)
    if not all(np.isfinite(value) for value in values):
        return -1
    if momentum_threshold <= 0 or expansion_threshold <= 0:
        raise ValueError("momentum and expansion thresholds must be positive")
    if (abs(float(future_momentum)) < float(momentum_threshold)
            or float(future_range_ratio) < float(expansion_threshold)):
        return COUPLING_CHOP
    if abs(float(past_momentum)) < float(momentum_threshold):
        return COUPLING_LAUNCH
    return (
        COUPLING_CONTINUATION
        if np.sign(future_momentum) == np.sign(past_momentum)
        else COUPLING_REVERSAL
    )


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

    Momentum is endpoint displacement divided by the median causal candle range.  Volatility is
    the log ratio between the median future candle range and that same causal scale.  Rows without
    complete finite history/future fail closed rather than being silently clipped into training.
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
        future_momentum = float((close[end] - close[decision]) / causal_scale)
        range_ratio = float(np.median(future_ranges) / causal_scale)
        if not np.isfinite(range_ratio) or range_ratio <= 0:
            continue
        momentum[index] = future_momentum
        volatility[index] = float(np.log(range_ratio))
        coupling[index] = coupling_class(
            past_momentum,
            future_momentum,
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
        return "momentum_volatility_coupling_v1"

    def control_evidence(self, history_row, probe_res):
        return {
            metric: self._history_metric(history_row, metric)
            for metric in (
                "mv_momentum_corr",
                "mv_volatility_corr",
                "mv_coupling_auc",
            )
        }

    def compare_control_evidence(self, real, controls):
        metrics = (
            "mv_momentum_corr",
            "mv_volatility_corr",
            "mv_coupling_auc",
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
            and real.get("mv_coupling_auc") is not None
            and float(real["mv_coupling_auc"]) > 0.5
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
        return passed, margins, (margins.get("shuffle") or {}).get("mv_coupling_auc")

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict["momentum_volatility_schema"] = MOMENTUM_VOLATILITY_SCHEMA
        verdict["pretext_note"] = (
            "Raw-OHLCV causal momentum/volatility coupling; require positive momentum and "
            "volatility correlations, coupling AUC above chance, corrupted-input control wins, "
            "parent capability retention, and downstream temporal validation."
        )
        return verdict


__all__ = [
    "COUPLING_CHOP",
    "COUPLING_CLASSES",
    "COUPLING_CONTINUATION",
    "COUPLING_LAUNCH",
    "COUPLING_REVERSAL",
    "MOMENTUM_VOLATILITY_SCHEMA",
    "MomentumVolatilityTargets",
    "MomentumVolatilityTask",
    "coupling_class",
    "momentum_volatility_targets",
]
