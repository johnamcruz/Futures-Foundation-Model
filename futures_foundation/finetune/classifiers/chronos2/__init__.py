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
from .paired_embeddings import (  # noqa: F401
    Chronos2PairedEmbeddings,
    SCHEMA as PAIRED_EMBEDDING_SCHEMA,
    embed_paired_window_chunks,
    paired_encoder_identity,
    validate_paired_windows,
)
from .paired_timeframes import (  # noqa: F401
    CHANNEL_NAMES as PAIRED_CHANNEL_NAMES,
    Chronos2PairedTimeframeData,
    SCHEMA as PAIRED_DATA_SCHEMA,
    prepare_paired_timeframes,
)


# A remote base model is a valid parent identity.  Production/research callers
# should pass the exact adapted checkpoint via ``backbone_ckpt``.
BASE_CKPT = "autogluon/chronos-2-small"

__all__ = [
    "BASE_CKPT",
    "Chronos2FrozenClassifier",
    "Chronos2MultivariateData",
    "Chronos2PairedEmbeddings",
    "Chronos2PairedTimeframeData",
    "DEFAULT_HOLDOUT_START",
    "PAIRED_CHANNEL_NAMES",
    "PAIRED_DATA_SCHEMA",
    "PAIRED_EMBEDDING_SCHEMA",
    "SCHEMA",
    "TIMEFRAME",
    "embed_paired_window_chunks",
    "paired_encoder_identity",
    "prepare_3min_multivariate",
    "prepare_paired_timeframes",
    "validate_paired_windows",
]
