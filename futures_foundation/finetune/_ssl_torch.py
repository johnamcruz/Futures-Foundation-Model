"""Backward-compat SHIM — the torch SSL layer now lives under `pretext/_torch/`.

The per-pretext trainers (mask / forecast / contrastive) each got their own module under a shared
BaseTrainer, and the window/embedding/ONNX helpers moved to `pretext/_torch/common.py`. This module
re-exports everything so existing references keep working (ssl_probe.embed_encoder, classifiers'
embed_windows / export_encoder_onnx, tests' train_ssl_*, ContrastiveTrendNet, _future_key, ...).

New code should import from `futures_foundation.finetune.pretext._torch` directly. Torch loads only
when THIS module (or that subpackage) is imported — which is always lazy — so the orchestrator and
pretext task registry stay torch-free.
"""
from .pretext._torch.common import (                      # noqa: F401  window/embedding/ONNX + base
    _nullctx, _enc, _standardize, _time_shuffle, _apply_control, _gather_batch,
    embed_encoder, embed_windows, _EncoderONNX, export_encoder_onnx, BaseTrainer)
from .pretext._torch.mask import MaskNetwork, train_ssl_mask                             # noqa: F401
from .pretext._torch.forecast import MultiHorizonForecastNet, train_ssl_forecast         # noqa: F401
from .pretext._torch.forecast_dist import (                # noqa: F401  stage-2.5 distributional refine
    train_ssl_forecast_dist, get_dist_objective, CandleQuantile, CandleBins)
from .pretext._torch.contrastive import (                 # noqa: F401  stage-3 temporal regime geometry
    ContrastiveTrendNet, train_ssl_contrastive, regime_gate, _random_crop_resize)
from .pretext._torch.nextleg import NextLegNet, train_ssl_nextleg              # noqa: F401
from .pretext._torch.nextleg_path import (                # noqa: F401  stage-2.7 nextleg + PATH
    NextLegPathNet, leg_retrace, train_ssl_nextleg_path)
from .pretext._torch.nextleg_race import (                # noqa: F401  causal-range path race
    NextLegRaceNet, train_ssl_nextleg_race)
from .pretext._torch.momentum_volatility import (         # noqa: F401  momentum/range coupling
    MomentumVolatilityNet, train_ssl_momentum_volatility)
from .pretext._torch.momentum_volatility_inference import (  # noqa: F401  compact MV readout
    export_mv_readout, load_mv_forecaster, mv_features)
from .pretext._torch.related_nextleg import (             # noqa: F401  grouped related-series refine
    RelatedNextLegNet, train_ssl_related_nextleg)
from .pretext._torch.nextleg_structural import (          # noqa: F401  HH/HL/LH/LL + BOS/CHOCH SSL
    StructuralNextLegNet, train_ssl_nextleg_structural)
from .pretext._torch.electra import (                     # noqa: F401  stage-4 turn-electra (replaced-TURN)
    ElectraNetwork, train_ssl_electra, clamp_valid_ohlc_t)
