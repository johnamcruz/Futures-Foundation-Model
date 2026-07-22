"""Pretext-task REGISTRY — pluggable SSL pretraining objectives (stages).

Add a new pretrain experiment by dropping a module here (subclass PretextTask) and registering
its instance in PRETEXTS below; the orchestrator (ssl.py) stays untouched. Keeping each task in
its own file keeps ssl.py a clean orchestrator.

  mask          (stage 1)   — BERT-style masked modeling
  forecast      (stage 2)   — multi-horizon / variable-context candle seq2seq
  forecast_dist (stage 2.5) — DISTRIBUTIONAL forecast refine ON stage-2 (Chronos-style
                              quantile/bin objectives; own modules, stage-2 untouched)
  contrastive   (stage 3)   — TEMPORAL-NEIGHBORHOOD contrastive: regime geometry from
                              multi-scale time proximity + augmentations, sigma-weighted
                              (replaced the outcome-keyed v1-v3, dropped 2026-07-02)
  nextleg       (stage 2.6) — NEXT-LEG forecasting: from a context ending at a confirmed
                              fractal pivot, predict the newborn leg's / counter-leg's length in
                              BARS. The GRADUATED backbone (mantis_ssl_nextleg.pt) — do not edit.
  nextleg_path  (stage 2.7) — nextleg + the leg's PATH ROUGHNESS (deepest pullback within the
                              leg / that leg's own extent — unitless, pure candles). Own modules;
                              2.6 untouched, so the A/B stays honest.
  nextleg_race  (stage 2.8) — nextleg + a future-only ORDERED adverse/progress curve. New modules;
                              production nextleg and its checkpoint remain untouched.
  nextleg_structural        — SpanBERT-style confirmed-pivot reconstruction plus causal
                              HH/HL/LH/LL, BOS/CHOCH, duration, and excursion prediction. It is
                              an opt-in encoder refinement; production nextleg stays untouched.
  electra       (stage 4)   — TURN-ELECTRA (replaced-TURN detection): span-mask the regions around
                              DETECTED SWINGS (the event a pivot entry trades), a weak generator
                              fills each masked turn with a plausible alternative development (a
                              SYNTHETIC FAKE TURN), the encoder labels every bar real/replaced —
                              so it must learn how GENUINE turns develop vs plausible imposters
                              (fakeout-vs-real), pure SSL, zero labels, fully generic; warm from
                              the promoted base. (Prior slot occupants in git history: replaced-
                              candle RTD, break-hold.)
"""
from .base import PretextTask
from .mask import MaskTask
from .forecast import ForecastTask
from .forecast_dist import ForecastDistTask
from .contrastive import ContrastiveTask
from .electra import TurnElectraTask
from .nextleg import NextLegTask
from .nextleg_path import NextLegPathTask
from .nextleg_race import NextLegRaceTask
from .related_nextleg import RelatedNextLegTask
from .nextleg_structural import StructuralNextLegTask

PRETEXTS = {t.name: t for t in (MaskTask(), ForecastTask(), ForecastDistTask(),
                                ContrastiveTask(), TurnElectraTask(), NextLegTask(),
                                NextLegPathTask(), NextLegRaceTask(), RelatedNextLegTask(),
                                StructuralNextLegTask())}


def get_pretext(name):
    """Resolve the pretext task by name (None -> 'mask'). Unknown name -> KeyError (fail fast)."""
    return PRETEXTS[name or 'mask']


__all__ = ['PretextTask', 'MaskTask', 'ForecastTask', 'ForecastDistTask', 'ContrastiveTask',
           'TurnElectraTask', 'NextLegTask', 'NextLegPathTask', 'NextLegRaceTask', 'PRETEXTS',
           'RelatedNextLegTask', 'StructuralNextLegTask', 'get_pretext']
