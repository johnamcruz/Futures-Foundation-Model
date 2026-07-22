"""Causal, self-supervised HH/HL/LH/LL targets for Structural NextLeg.

The input decision is a *confirmed* fractal pivot.  Current structure uses only that pivot and
earlier confirmed pivots.  Future structure, duration, and excursion are training targets only.
Every target records the confirmation index of its furthest future pivot so the trainer can prove
that the complete label lies inside one stream and one temporal split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ...primitives.detection import detect_fractal_pivots
from .nextleg import NextLegTask


UPTREND = 0                 # HH + HL
DOWNTREND = 1               # LH + LL
EXPANDING = 2               # HH + LL
CONTRACTING = 3             # LH + HL
STRUCTURE_NAMES = ("uptrend_hh_hl", "downtrend_lh_ll", "expanding_hh_ll",
                   "contracting_lh_hl")
MASKED_BREAK = -1
NO_BREAK = 0
BULLISH_BOS = 1
BEARISH_BOS = 2
BULLISH_CHOCH = 3
BEARISH_CHOCH = 4
BREAK_NAMES = ("no_break", "bullish_bos", "bearish_bos", "bullish_choch",
               "bearish_choch")


@dataclass(frozen=True)
class StructuralTargets:
    confirms: np.ndarray
    future_ends: np.ndarray
    durations: np.ndarray
    current_state: np.ndarray
    next_state: np.ndarray
    excursions: np.ndarray
    break_event: np.ndarray
    break_delay: np.ndarray

    def __post_init__(self):
        n = len(self.confirms)
        if any(len(value) != n for value in (
                self.future_ends, self.durations, self.current_state,
                self.next_state, self.excursions, self.break_event, self.break_delay)):
            raise ValueError("structural target arrays must have equal length")
        if self.durations.shape != (n, 2) or self.excursions.shape != (n, 2):
            raise ValueError("duration and excursion targets must be [N,2]")

    @property
    def values(self) -> np.ndarray:
        return np.column_stack((self.durations, self.current_state, self.next_state,
                                self.excursions, self.break_event,
                                np.log1p(self.break_delay))).astype(np.float32, copy=False)


def _empty() -> StructuralTargets:
    return StructuralTargets(
        np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 2), np.float32),
        np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 2), np.float32),
        np.empty(0, np.int64), np.empty(0, np.int64))


def _state(high_relation: np.ndarray, low_relation: np.ndarray) -> np.ndarray:
    """Map same-side swing comparisons (+1 higher, -1 lower) into four structures."""
    out = np.full(len(high_relation), -1, np.int64)
    out[(high_relation > 0) & (low_relation > 0)] = UPTREND
    out[(high_relation < 0) & (low_relation < 0)] = DOWNTREND
    out[(high_relation > 0) & (low_relation < 0)] = EXPANDING
    out[(high_relation < 0) & (low_relation > 0)] = CONTRACTING
    return out


def alternating_fractals(high, low, k: int) -> list[tuple[int, int, int]]:
    """Pure raw-candle fractals with deterministic keep-first direction alternation."""
    pivots = detect_fractal_pivots(np.asarray(high, float), np.asarray(low, float), k=int(k))
    out, last_direction = [], 0
    for pivot in pivots:
        direction = int(pivot["direction"])
        if direction == last_direction:
            continue
        out.append((int(pivot["origin"]), int(pivot["confirm"]), direction))
        last_direction = direction
    return out


def first_bos_choch_event(close_from_decision, state: int, *, last_high: float,
                          last_low: float, horizon: int) -> tuple[int, int]:
    """First close-confirmed BOS/CHOCH strictly after one directional decision.

    Element zero is the completed decision candle and is never inspected as a future event.
    ``NO_BREAK`` means no event inside the complete declared horizon; non-directional structure
    is ``MASKED_BREAK`` because assigning BOS/CHOCH there is subjective.
    """
    close = np.asarray(close_from_decision, np.float64)
    horizon = int(horizon)
    if horizon < 1 or len(close) < horizon + 1:
        raise ValueError("a complete positive BOS/CHOCH horizon is required")
    if state not in (UPTREND, DOWNTREND):
        return MASKED_BREAK, 0
    future = close[1:horizon + 1]
    if state == UPTREND:
        bos, choch = future > float(last_high), future < float(last_low)
        bos_code, choch_code = BULLISH_BOS, BEARISH_CHOCH
    else:
        bos, choch = future < float(last_low), future > float(last_high)
        bos_code, choch_code = BEARISH_BOS, BULLISH_CHOCH
    bos_at = int(np.flatnonzero(bos)[0] + 1) if bos.any() else horizon + 1
    choch_at = int(np.flatnonzero(choch)[0] + 1) if choch.any() else horizon + 1
    if min(bos_at, choch_at) > horizon:
        return NO_BREAK, horizon
    return (bos_code, bos_at) if bos_at < choch_at else (choch_code, choch_at)


def _bos_choch_events(close, confirms, states, last_high, last_low, horizon: int,
                      chunk_size: int = 50_000):
    """Vectorized/chunked form of :func:`first_bos_choch_event` for millions of pivots."""
    close = np.asarray(close, np.float64)
    confirms, states = np.asarray(confirms, np.int64), np.asarray(states, np.int64)
    last_high, last_low = np.asarray(last_high, np.float64), np.asarray(last_low, np.float64)
    n, horizon = len(confirms), int(horizon)
    event = np.full(n, MASKED_BREAK, np.int64)
    delay = np.zeros(n, np.int64)
    complete = ((states == UPTREND) | (states == DOWNTREND)) & (confirms + horizon < len(close))
    eligible = np.flatnonzero(complete)
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    for start in range(0, len(eligible), int(chunk_size)):
        rows = eligible[start:start + int(chunk_size)]
        future = close[confirms[rows, None] + offsets[None, :]]
        up = states[rows] == UPTREND
        bos = np.where(up[:, None], future > last_high[rows, None],
                       future < last_low[rows, None])
        choch = np.where(up[:, None], future < last_low[rows, None],
                         future > last_high[rows, None])
        bos_any, choch_any = bos.any(1), choch.any(1)
        bos_at = np.where(bos_any, bos.argmax(1) + 1, horizon + 1)
        choch_at = np.where(choch_any, choch.argmax(1) + 1, horizon + 1)
        is_bos = bos_at < choch_at
        happened = np.minimum(bos_at, choch_at) <= horizon
        codes = np.where(
            is_bos, np.where(up, BULLISH_BOS, BEARISH_BOS),
            np.where(up, BEARISH_CHOCH, BULLISH_CHOCH))
        event[rows] = np.where(happened, codes, NO_BREAK)
        delay[rows] = np.where(happened, np.minimum(bos_at, choch_at), horizon)
    return event, delay, complete


def structural_targets_from_pivots(ohlcv, pivots, *, k: int, leg_cap: int,
                                   event_horizon: int = 128) -> StructuralTargets:
    """Vectorized resolved targets from one stream and an alternating pivot sequence.

    Excursion is ``log1p(basis points)`` from the decision close to the first future extreme and
    between the first/second future extremes.  It is dimensionless and invariant to multiplying
    every OHLC price by a constant.
    """
    bars = np.asarray(ohlcv, np.float32)
    if bars.ndim != 2 or bars.shape[1] < 4:
        raise ValueError("ohlcv must be [N,>=4]")
    pivots = list(pivots)
    if len(pivots) < 6:
        return _empty()
    origins = np.asarray([row[0] for row in pivots], np.int64)
    confirms = np.asarray([row[1] for row in pivots], np.int64)
    directions = np.asarray([row[2] for row in pivots], np.int8)
    if ((origins < 0) | (confirms < origins) | (confirms >= len(bars))).any():
        raise ValueError("pivot origin/confirmation lies outside its stream")
    if (directions[1:] == directions[:-1]).any() or not np.isin(directions, (-1, 1)).all():
        raise ValueError("pivots must strictly alternate +/-1 directions")

    high, low, close = bars[:, 1], bars[:, 2], bars[:, 3]
    prices = np.where(directions == -1, high[origins], low[origins]).astype(np.float64)
    relation = np.zeros(len(pivots), np.int8)
    delta = prices[2:] - prices[:-2]
    relation[2:] = np.sign(delta).astype(np.int8)  # high: HH/LH; low: HL/LL

    i = np.arange(2, len(pivots) - 2, dtype=np.int64)
    current_high = np.where(directions[i] == -1, relation[i], relation[i - 1])
    current_low = np.where(directions[i] == 1, relation[i], relation[i - 1])
    next_high = np.where(directions[i + 1] == -1, relation[i + 1], relation[i])
    next_low = np.where(directions[i + 1] == 1, relation[i + 1], relation[i])
    current_state, next_state = _state(current_high, current_low), _state(next_high, next_low)
    last_high = np.where(directions[i] == -1, prices[i], prices[i - 1])
    last_low = np.where(directions[i] == 1, prices[i], prices[i - 1])

    t1 = origins[i + 1] - confirms[i]
    t2 = origins[i + 2] - origins[i + 1]
    future_ends = confirms[i + 2].copy()  # target identity needs confirmation, not just extreme
    break_event, break_delay, complete_break = _bos_choch_events(
        close, confirms[i], current_state, last_high, last_low, int(event_horizon))
    future_ends[complete_break] = np.maximum(
        future_ends[complete_break], confirms[i][complete_break] + int(event_horizon))
    ref1 = close[confirms[i]].astype(np.float64)
    ref2 = prices[i + 1]
    eps = np.finfo(np.float32).eps
    mag1 = np.log1p(10_000.0 * np.abs(prices[i + 1] - ref1) / np.maximum(np.abs(ref1), eps))
    mag2 = np.log1p(10_000.0 * np.abs(prices[i + 2] - ref2) / np.maximum(np.abs(ref2), eps))
    valid = ((t1 > 0) & (t2 > 0) & (t1 <= int(leg_cap)) & (t2 <= int(leg_cap))
             & (current_state >= 0) & (next_state >= 0)
             & (future_ends > confirms[i])
             & np.isfinite(mag1) & np.isfinite(mag2))
    if not valid.any():
        return _empty()
    return StructuralTargets(
        confirms[i][valid], future_ends[valid],
        np.column_stack((np.log1p(t1[valid]), np.log1p(t2[valid]))).astype(np.float32),
        current_state[valid], next_state[valid],
        np.column_stack((mag1[valid], mag2[valid])).astype(np.float32),
        break_event[valid], break_delay[valid])


def structural_targets(ohlcv, k: int, leg_cap: int, event_horizon: int = 128) -> StructuralTargets:
    bars = np.asarray(ohlcv, np.float32)
    pivots = alternating_fractals(bars[:, 1], bars[:, 2], int(k))
    return structural_targets_from_pivots(
        bars, pivots, k=int(k), leg_cap=int(leg_cap), event_horizon=int(event_horizon))


def structural_targets_by_segments(
        big, segments: Iterable[tuple[int, int]], *, k: int, leg_cap: int,
        event_horizon: int = 128,
        builder: Callable = structural_targets) -> StructuralTargets:
    """Build targets independently inside each stream; cross-stream pivots are impossible."""
    bars = np.asarray(big, np.float32)
    rows = []
    for base, size in segments:
        base, size = int(base), int(size)
        if base < 0 or size < 0 or base + size > len(bars):
            raise ValueError("stream segment lies outside assembled bars")
        target = builder(
            bars[base:base + size], int(k), int(leg_cap),
            event_horizon=int(event_horizon))
        if len(target.confirms):
            if target.future_ends.max() >= size:
                raise RuntimeError("structural target crosses its source stream")
            rows.append(StructuralTargets(
                target.confirms + base, target.future_ends + base, target.durations,
                target.current_state, target.next_state, target.excursions,
                target.break_event, target.break_delay))
    if not rows:
        return _empty()
    return StructuralTargets(
        np.concatenate([row.confirms for row in rows]),
        np.concatenate([row.future_ends for row in rows]),
        np.concatenate([row.durations for row in rows]),
        np.concatenate([row.current_state for row in rows]),
        np.concatenate([row.next_state for row in rows]),
        np.concatenate([row.excursions for row in rows]),
        np.concatenate([row.break_event for row in rows]),
        np.concatenate([row.break_delay for row in rows]))


def validate_structural_reserve(confirms, future_ends, *, max_ctx: int,
                                target_reserve: int) -> None:
    """Prove the furthest future confirmation fits the assembler's split-safe window."""
    confirms = np.asarray(confirms, np.int64)
    future_ends = np.asarray(future_ends, np.int64)
    if len(confirms) != len(future_ends):
        raise ValueError("confirm and future-end arrays must align")
    touched = future_ends - confirms
    permitted = int(target_reserve) - int(max_ctx)
    if len(touched) and (touched < 1).any():
        raise AssertionError("TEMPORAL LEAK: structural target is not strictly future")
    if len(touched) and int(touched.max()) > permitted:
        raise AssertionError(
            f"TEMPORAL LEAK: structural target reads {int(touched.max())} bars ahead but only "
            f"{permitted} split-safe future bars were reserved")


def structural_span_bounds(context_length: int, *, confirmation_lag: int,
                           span_width: int) -> tuple[int, int]:
    """Return the causal formation span around the current pivot origin.

    Structural examples end on the pivot confirmation candle.  A ``k``-fractal's origin is
    therefore exactly ``k`` completed bars behind the right edge.  The reconstructed span may
    include formation bars on either side of that origin, but never a bar after the decision
    close.  Keeping this calculation torch-free makes the boundary independently testable.
    """
    length, lag, width = int(context_length), int(confirmation_lag), int(span_width)
    if length < 1 or lag < 1:
        raise ValueError("context_length and confirmation_lag must be positive")
    if width < 1 or width % 2 != 1:
        raise ValueError("structural span width must be a positive odd integer")
    if width > 2 * lag + 1:
        raise ValueError("structural span cannot extend beyond the confirmed pivot formation")
    center = length - 1 - lag
    half = width // 2
    bounds = center - half, center + half + 1
    if bounds[0] < 0 or bounds[1] > length:
        raise ValueError("structural span falls outside causal context")
    return bounds


class StructuralNextLegTask(NextLegTask):
    name, trainer = "nextleg_structural", "train_ssl_nextleg_structural"
    requires_stream_layout = True

    @property
    def control_contract(self):
        return "structural_nextleg_temporal_skill_v1"

    def reserve(self, cfg):
        future = max(2 * int(cfg.get("leg_cap", 256)) + int(cfg.get("leg_k", 2)),
                     int(cfg.get("structure_event_horizon", 128)))
        return max(int(x) for x in cfg["context_lengths"]) + future

    def control_evidence(self, history_row, probe_res):
        evidence = super().control_evidence(history_row, probe_res)
        evidence.update({
            "current_structure_bal_acc": self._metric(
                history_row, "current_structure_bal_acc"),
            "next_structure_bal_acc": self._metric(history_row, "next_structure_bal_acc"),
            "break_bal_acc": self._metric(history_row, "break_bal_acc"),
            "excursion_corr": self._metric(history_row, "excursion_corr"),
            "span_skill": self._metric(history_row, "span_skill"),
        })
        return evidence

    def compare_control_evidence(self, real, controls):
        metrics = ("forecast_skill", "leg_corr1", "leg_corr2",
                   "current_structure_bal_acc", "next_structure_bal_acc",
                   "break_bal_acc", "excursion_corr", "span_skill")
        margins = {name: {
            metric: (None if real.get(metric) is None or row.get(metric) is None else
                     float(real[metric]) - float(row[metric]))
            for metric in metrics} for name, row in controls.items()}
        # Regression skill/correlation must beat zero; balanced accuracy must beat the exact
        # uniform-chance baseline.  This prevents an imbalanced but non-learning event head from
        # passing merely because its raw accuracy is positive.
        thresholds = {
            "forecast_skill": 0.0, "leg_corr1": 0.0, "leg_corr2": 0.0,
            "current_structure_bal_acc": 1.0 / len(STRUCTURE_NAMES),
            "next_structure_bal_acc": 1.0 / len(STRUCTURE_NAMES),
            "break_bal_acc": 1.0 / len(BREAK_NAMES),
            "excursion_corr": 0.0, "span_skill": 0.0,
        }
        positive = all(real.get(metric) is not None
                       and float(real[metric]) > thresholds[metric]
                       for metric in metrics)
        passed = bool(positive and controls and all(
            margin is not None and margin > 0
            for per_control in margins.values() for margin in per_control.values()))
        return passed, margins, (margins.get("shuffle") or {}).get("next_structure_bal_acc")

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict["pretext_note"] = (
            "Structural NextLeg SSL: confirmed-pivot HH/HL/LH/LL state plus future structure, "
            "duration, and scale-free excursion; require exact stream boundaries, full future-"
            "confirmation reserve, corrupted-input controls, Probe Atlas retention, and "
            "downstream anchored Pivot walk-forward before promotion")
        return verdict


__all__ = [
    "UPTREND", "DOWNTREND", "EXPANDING", "CONTRACTING", "STRUCTURE_NAMES",
    "MASKED_BREAK", "NO_BREAK", "BULLISH_BOS", "BEARISH_BOS", "BULLISH_CHOCH",
    "BEARISH_CHOCH", "BREAK_NAMES", "first_bos_choch_event",
    "StructuralTargets", "alternating_fractals", "structural_targets_from_pivots",
    "structural_targets", "structural_targets_by_segments", "validate_structural_reserve",
    "structural_span_bounds",
    "StructuralNextLegTask",
]
