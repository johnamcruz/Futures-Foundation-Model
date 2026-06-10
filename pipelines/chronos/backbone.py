"""Back-compat shim — the Chronos-Bolt seam was PROMOTED to
`futures_foundation.foundation` (it is the repo's foundation surface now).

All existing imports (`from pipelines.chronos import backbone`;
`backbone.embed(...)`) keep working unchanged: this module re-exports the
canonical implementations. New code should import
`futures_foundation.foundation` directly.

The process contract is unchanged: `embed()` runs torch in an isolated
subprocess; the parent stays torch-free (macOS torch+xgboost libomp
segfault — see foundation.py docstring).
"""
from futures_foundation.foundation import (   # noqa: F401
    MODEL, D_MODEL, CTX,
    active_source, stamp_active_source, _find_unused_finetunes,
    embed, embed_bars,
    pipeline, _model, d_model, fresh_model, pool,
)
