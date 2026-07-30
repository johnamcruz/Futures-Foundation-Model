"""Chronos-2 classifier plugin package.

The public adapter is torch-free and self-registers lazily through the generic
classifier seam.  Heavy Chronos/torch imports live only in ``_embed_worker``.
"""
from .frozen import Chronos2FrozenClassifier  # noqa: F401
from .multivariate import (  # noqa: F401
    Chronos2MultivariateData,
    DEFAULT_HOLDOUT_START,
    SCHEMA,
    TIMEFRAME,
    prepare_3min_multivariate,
)


# A remote base model is a valid parent identity.  Production/research callers
# should pass the exact adapted checkpoint via ``backbone_ckpt``.
BASE_CKPT = "autogluon/chronos-2-small"

__all__ = [
    "BASE_CKPT",
    "Chronos2FrozenClassifier",
    "Chronos2MultivariateData",
    "DEFAULT_HOLDOUT_START",
    "SCHEMA",
    "TIMEFRAME",
    "prepare_3min_multivariate",
]
