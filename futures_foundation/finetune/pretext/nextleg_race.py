"""Stage-2.8 v2: causal, scale-normalized NEXT-LEG path forecasting.

The original race target divided adverse movement by the future leg's eventual maximum.  That
answered "how rough was a leg which eventually existed?" but discarded the launch question: a tiny
leg and a large trend could receive the same target, and unreached progress levels had no label.

V2 uses only information available at the confirmed-pivot decision to establish scale, then asks:

* does the newborn leg reach 1/2/3/4 causal range units before its next extreme;
* how much adverse movement occurs before each level (or before censoring); and
* how many bars elapse before each level (or before censoring)?

The future path is a self-supervised target only.  Inputs end at confirmation.  There is no ATR,
entry, stop, target, R multiple, cost, or strategy label.  A downstream strategy decides how the
generic reach/adverse/time distribution maps to its own economics.
"""
from __future__ import annotations

import numpy as np

from .nextleg import NextLegTask


RACE_LEVELS = (1.0, 2.0, 3.0, 4.0)
RACE_SCHEMA = "causal_range_competing_path_v2"


def causal_bar_scale(high, low, confirm: int, lookback: int = 64) -> float:
    """Median completed-candle ``high-low`` range ending at ``confirm``.

    This is deliberately not ATR: it reads no previous-close gap and applies no Wilder smoothing.
    It is raw bar geometry, strictly causal, robust to isolated range spikes, positive, and
    invariant to a constant multiplication of all prices. ``NaN`` means the decision lacks
    sufficient finite history and must not become a training anchor.
    """
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    confirm, lookback = int(confirm), int(lookback)
    if lookback < 2:
        raise ValueError("lookback must be at least two bars")
    start = confirm - lookback + 1
    if start < 0 or confirm >= len(high):
        return float("nan")
    candle_range = high[start:confirm + 1] - low[start:confirm + 1]
    scale = float(np.median(candle_range))
    return scale if np.isfinite(scale) and scale > np.finfo(np.float32).eps else float("nan")


def scaled_path_race(high, low, close, confirm: int, leg_end: int, direction: int, *,
                     levels=RACE_LEVELS, lookback: int = 64, cap: float = 8.0) -> np.ndarray:
    """Return reach/adverse/delay targets for one future newborn leg.

    Output is ``[3, levels]``:

    ``reach``
        One when favourable excursion first reaches ``level * causal_range_scale``.
    ``adverse``
        Maximum adverse excursion, in causal range units, through the first-reaching bar.  If the
        level is not reached, this is the full adverse excursion through ``leg_end`` (right-censor).
    ``delay``
        ``log1p`` bars to first reach, or ``log1p`` full leg duration when censored.

    Same-bar high/low ambiguity is conservative: adverse excursion on the reaching bar is included.
    Reach is non-increasing with level; adverse and delay are non-decreasing.
    """
    levels = tuple(float(value) for value in levels)
    if not levels or any(value <= 0 for value in levels):
        raise ValueError("levels must be non-empty and positive")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly increasing")
    if cap <= 0:
        raise ValueError("cap must be positive")
    confirm, leg_end, direction = int(confirm), int(leg_end), int(direction)
    if leg_end <= confirm:
        return np.full((3, len(levels)), np.nan, np.float32)
    scale = causal_bar_scale(high, low, confirm, lookback)
    if not np.isfinite(scale):
        return np.full((3, len(levels)), np.nan, np.float32)

    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    reference = float(close[confirm])
    segment_high = high[confirm + 1:leg_end + 1]
    segment_low = low[confirm + 1:leg_end + 1]
    if direction == 1:
        favourable, adverse = segment_high - reference, reference - segment_low
    elif direction == -1:
        favourable, adverse = reference - segment_low, segment_high - reference
    else:
        raise ValueError(f"direction must be +/-1, got {direction}")
    if not len(favourable) or not np.isfinite(favourable).all() or not np.isfinite(adverse).all():
        return np.full((3, len(levels)), np.nan, np.float32)

    adverse = np.maximum(adverse, 0.0)
    reach, risk, delay = [], [], []
    final = len(favourable) - 1
    for level in levels:
        hit = np.flatnonzero(favourable >= level * scale)
        reached = bool(len(hit))
        stop = int(hit[0]) if reached else final
        reach.append(float(reached))
        risk.append(min(float(np.max(adverse[:stop + 1]) / scale), float(cap)))
        delay.append(float(np.log1p(stop + 1)))
    return np.asarray((reach, risk, delay), np.float32)


class NextLegRaceTask(NextLegTask):
    name, trainer = "nextleg_race", "train_ssl_nextleg_race"
    requires_stream_layout = True

    @property
    def control_contract(self):
        return "nextleg_causal_range_race_v2"

    def control_evidence(self, history_row, probe_res):
        evidence = super().control_evidence(history_row, probe_res)
        for metric in ("race_reach_auc", "race_adverse_corr", "race_delay_corr"):
            evidence[metric] = self._metric(history_row, metric)
        return evidence

    def reserve(self, cfg):
        # Resolving the second future pivot reads through its confirmation bars, not merely its
        # origin.  The original race inherited NextLeg's 2*leg_cap reserve and omitted this final
        # ``leg_k`` confirmation tail, allowing boundary labels to inspect the next split.
        return (max(int(value) for value in cfg["context_lengths"])
                + 2 * int(cfg.get("leg_cap", 256)) + int(cfg.get("leg_k", 2)))

    def compare_control_evidence(self, real, controls):
        metrics = ("forecast_skill", "leg_corr1", "leg_corr2",
                   "race_reach_auc", "race_adverse_corr", "race_delay_corr")
        margins = {
            name: {
                metric: (None if real.get(metric) is None or row.get(metric) is None else
                         float(real[metric]) - float(row[metric]))
                for metric in metrics
            }
            for name, row in controls.items()
        }
        thresholds = {
            "forecast_skill": 0.0, "leg_corr1": 0.0, "leg_corr2": 0.0,
            "race_reach_auc": 0.5, "race_adverse_corr": 0.0, "race_delay_corr": 0.0,
        }
        positive = all(real.get(metric) is not None
                       and float(real[metric]) > threshold
                       for metric, threshold in thresholds.items())
        passed = bool(positive and controls and all(
            margin is not None and margin > 0
            for per_control in margins.values() for margin in per_control.values()))
        return passed, margins, (margins.get("shuffle") or {}).get("race_reach_auc")

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict["race_schema"] = RACE_SCHEMA
        verdict["pretext_note"] = (
            "NextLeg causal-range path race v2: require reach AUC > chance, positive adverse/delay "
            "correlation, retained candle/leg skill, corrupted-input control wins, Probe Atlas "
            "retention, then downstream temporal WF.")
        return verdict


__all__ = [
    "RACE_LEVELS", "RACE_SCHEMA", "NextLegRaceTask", "causal_bar_scale",
    "scaled_path_race",
]
