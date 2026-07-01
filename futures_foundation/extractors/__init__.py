"""Swappable feature-extractor interface.

The pipeline depends ONLY on this package's interface (FeatureExtractor +
evaluate_with_extractor). Each concrete backbone lives in its own subpackage and
implements FeatureExtractor; swap by name via get_extractor().

DEPRECATED (2026-07-01): the Chronos extractor is the LEGACY / backward-compat path — it
remains fully functional and is the LIVE incumbent baseline (SuperTrend Chronos), but NEW
code should use the Mantis foundation in `futures_foundation.finetune` (SSL pretext stages +
mantis_frozen), not this extractor. Kept working until Mantis Pivot Trend replaces it in
production; do NOT delete while the live bot depends on it.
"""
import warnings

from .base import FeatureExtractor, evaluate_with_extractor, _windows
from .chronos import ChronosExtractor

_REGISTRY = {'chronos': ChronosExtractor}
_DEPRECATED = {'chronos'}


def get_extractor(name: str = 'chronos', **kw) -> FeatureExtractor:
    """Resolve an extractor by name. 'chronos' is DEPRECATED (legacy/backward-compat, still the
    live incumbent) — new code should use the Mantis foundation in futures_foundation.finetune."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown extractor {name!r}; have {list(_REGISTRY)}")
    if name in _DEPRECATED:
        warnings.warn(
            f"extractor {name!r} is DEPRECATED (legacy/backward-compat; still the LIVE incumbent "
            f"baseline). New code should use the Mantis foundation in futures_foundation.finetune.",
            DeprecationWarning, stacklevel=2)
    return _REGISTRY[name](**kw)


__all__ = ['FeatureExtractor', 'ChronosExtractor', 'get_extractor',
           'evaluate_with_extractor']
