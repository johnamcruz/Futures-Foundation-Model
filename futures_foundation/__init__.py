"""
Futures Foundation Model (FFM)

A futures-market foundation layer built on **pretrained Chronos-Bolt**
(`amazon/chronos-bolt-tiny`, frozen, subprocess-isolated) plus torch-free
feature/label/primitive libraries and a strategy labeling/eval framework.

The from-scratch FFM transformer (model/dataset/pretrain/torch fine-tune
trainer) was retired in favor of Chronos-Bolt (see git tag
`ffm-transformer-final` for the last version with the full torch backbone
stack). Walk-forward training now lives in `pipelines/chronos`
(Bolt embeddings + XGBoost). The foundation surface is
`futures_foundation.foundation`:

    from futures_foundation import foundation

    foundation.stamp_active_source(context='my run')
    E = foundation.embed_bars(close, indices)      # [N, 256], strictly causal

IMPORT CONTRACT: importing this package must stay **torch-free** — parents
that consume embeddings run XGBoost, and torch+xgboost segfault in one
process on macOS (libomp collision). Every submodule here (including
`finetune`, now labeling/reporting/economics only) imports zero
torch/transformers.
"""

# Torch-free, always available.
from . import foundation
from .foundation import embed_bars, stamp_active_source, D_MODEL
from .features import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from .labels import (
    generate_all_labels, generate_regime_labels, generate_volatility_labels,
    generate_structure_labels, generate_range_labels, print_label_distribution,
    REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
    LABEL_CONFIDENCE_SENTINEL,
)
from . import primitives
from .prepare import prepare_data

# finetune is torch-free (labeling / reporting / realized-R economics) —
# imported eagerly.
from .finetune import (
    StrategyLabeler, TrainingConfig, FoldHealthMonitor,
    run_labeling, print_eval_summary, print_fold_progression,
    summarize_fold_precision,
)

__version__ = "2.0.0"

__all__ = [
    # foundation (Chronos-Bolt)
    "foundation", "embed_bars", "stamp_active_source", "D_MODEL",
    # torch-free libraries
    "derive_features", "get_model_feature_columns", "INSTRUMENT_MAP",
    "generate_all_labels", "generate_regime_labels", "generate_volatility_labels",
    "generate_structure_labels", "generate_range_labels", "print_label_distribution",
    "REGIME_LABELS", "VOLATILITY_LABELS", "STRUCTURE_LABELS", "RANGE_LABELS",
    "LABEL_CONFIDENCE_SENTINEL", "primitives", "prepare_data",
    # finetune (labeling / reporting / economics — torch-free)
    "StrategyLabeler", "TrainingConfig", "FoldHealthMonitor",
    "run_labeling", "print_eval_summary", "print_fold_progression",
    "summarize_fold_precision",
]
