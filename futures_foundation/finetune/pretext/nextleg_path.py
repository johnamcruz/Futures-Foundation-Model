"""Stage-2.7 pretext: NEXT-LEG + PATH — stage 2.6 plus the leg's ROUGHNESS.

2.6 (nextleg, the graduated backbone) teaches HOW FAR and HOW LONG a leg runs, both in bars.
It never teaches HOW THE LEG GETS THERE. 2.7 adds exactly one target:

  r1 = the deepest pullback WITHIN the newborn leg / that leg's OWN extent

A ratio of two price distances from the same leg -> unitless, scale-free, instrument- and
TF-agnostic, read straight off candle highs/lows. NO ATR, NO cost, NO entry/stop/R: FFM learns
candles, the strategy layer owns risk. Pure self-supervision — the target is the market's own
structure applied to future price, same as t1/t2.

MEASURED HOLE this was built for (ES+NQ@3min, 130,994 production pivots): predicting whether a
leg's path takes out a stop reads 0.5184 from pure stop geometry, 0.5572 from the nextleg
embedding, and 0.5572 from both together — so real path structure exists (+0.039 over geometry),
geometry explains none of what the encoder has, and the encoder plainly has only part of it.
2.6 was never asked for the rest. This asks.

Why it should survive where turn-electra died: it stays a FORECASTER of the market's own pattern
unit (generic objective, aimed NEAR the downstream event, never AT it — the banked design law),
keeps the candle-seq2seq ANCHOR, and warm-starts from 2.6 with early blocks frozen. Expressing
the same idea in R-units would smuggle the strategy's stop into the pretext and reproduce exactly
the shape that lost.

GATES (all must pass before 2.7 replaces 2.6 anywhere):
  1. retrace_corr materially > 0                    — it learned the path target at all
  2. skill / leg_corr1 / leg_corr2 not degraded     — no drift damage to what already works
  3. probe_atlas pred_stopped_out > 0.5473          — the hole actually closed
     (geometry-only floor 0.5184) AND pred_vol_expand not below 0.8420 (drift kill-switch)
  4. error_mining missed_winners > 0.5174           — the recall blind spot
  5. trend-lifecycle scorecard >= 2.6               — RE-BASELINE 2.6 first: the banked
     0.7689/0.7635 bars were measured under the OLD shuffled probe split (fixed 1e4bf45)
     and are NOT comparable to a contiguous-split number
  6. downstream WR@3R / meanR >= 2.6 at the deploy operating points

This module is TORCH-FREE (the pure path math + the task); the trainer lives in
_torch/nextleg_path.py. Same split as spans.py/electra.py — keeps the math testable without
importing torch (libomp isolation).
"""
import numpy as np

from .nextleg import NextLegTask


def leg_retrace(h, l, o_i, o_n, d, cap=2.0):
    """The newborn leg's PATH ROUGHNESS — pure candle structure, nothing else.

    The leg runs from its origin extreme (o_i) to its end extreme (o_n) in direction d. Walk it
    and take the deepest giveback from the running favourable extreme, divided by the leg's OWN
    extent -> unitless in [0, cap]. 0.0 = a clean one-way leg; 0.5 = at some point it handed back
    half of its best progress.

    Scale-free BY CONSTRUCTION: a ratio of two price distances from the SAME leg, so x10 the
    prices and r1 is unchanged. That is what makes it instrument- and timeframe-agnostic without
    ever touching ATR — the same property that lets t1/t2 live in bars.

    -> float in [0, cap], or NaN when the leg has no extent (target UNRESOLVED -> the caller
    drops the anchor; never fabricate 0.0, which would read as 'a clean leg')."""
    h = np.asarray(h, float); l = np.asarray(l, float)
    seg_h, seg_l = h[o_i:o_n + 1], l[o_i:o_n + 1]
    if len(seg_h) < 2:
        return np.nan
    if d == 1:                                             # pivot LOW -> leg travels UP
        extent = h[o_n] - l[o_i]
        if not (extent > 0):
            return np.nan
        dd = float((np.maximum.accumulate(seg_h) - seg_l).max())     # peak-to-trough giveback
    else:                                                  # pivot HIGH -> leg travels DOWN
        extent = h[o_i] - l[o_n]
        if not (extent > 0):
            return np.nan
        dd = float((seg_h - np.minimum.accumulate(seg_l)).max())
    return float(min(dd / extent, cap))


class NextLegPathTask(NextLegTask):
    name, trainer = 'nextleg_path', 'train_ssl_nextleg_path'

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict['pretext_note'] = ('next-leg + PATH forecaster (bars + unitless retrace): judge '
                                   'on (a) retrace_corr > 0, (b) no regression in skill/leg_corr '
                                   'vs 2.6, (c) probe_atlas pred_stopped_out beating 0.5473 with '
                                   'pred_vol_expand intact, (d) the trend-lifecycle scorecard '
                                   're-baselined under the contiguous probe split, THEN the WR '
                                   'pipeline; leg/retrace skill are learning diagnostics only, '
                                   'never a ship gate')
        return verdict
