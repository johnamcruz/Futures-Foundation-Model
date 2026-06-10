"""Chronos pipeline — a generic, strategy-pluggable fine-tune framework.

Lives inside `futures_foundation` (promoted from the former `pipelines/chronos`). The Chronos-Bolt backbone is fine-tuned with a
small classification head on labels supplied by a pluggable strategy; results
are judged only under a leak-free, shuffle/random/cost honest ruler. Strategy
specifics live OUTSIDE this public package (private colabs/); this package
stays strategy-agnostic.

Files:
  (backbone)  — the ONLY Chronos seam is `futures_foundation.foundation`;
                modules here import it as `backbone` (Bolt load, pristine
                reset, causal masked-mean encoder pooling).
  finetune.py — generic seeded/deterministic backbone+head supervised
                fine-tune (train/predict); no strategy or eval logic.
  strategy.py — StrategyLabeler protocol (calendar/build/evaluate);
                contract: causal contexts, leak-free purge, cost inside
                evaluate(). The single pluggable seam.
  evaluate.py — honest ruler: leak-free walk-forward x seeds x
                REAL/SHUFFLE/RANDOM; believe a result only if REAL clearly
                beats SHUFFLE and RANDOM.
  data.py     — 6-ticker bars -> long format; leak-free walk-forward splits
                (born-tested: tests/test_chronos_data.py).
  embed.py    — frozen Chronos mean-pooled embeddings (generic utility).

Non-negotiable: strictly causal (bars <= t), splits never leak forward, no
performance number believed until it clears shuffle + random + cost.
"""
import sys as _sys

# LEGACY-PICKLE COMPAT: production joblib bundles trained before the
# consolidation (e.g. kalman_nw_chronos.joblib, supertrend_chronos.joblib)
# reference classes under the old 'pipelines.chronos.*' module paths.
# Aliasing this package under that name lets Python's import machinery
# resolve those submodules through OUR __path__, so old bundles keep
# unpickling without modification. New bundles pickle under the new paths.
_sys.modules.setdefault('pipelines.chronos', _sys.modules[__name__])
