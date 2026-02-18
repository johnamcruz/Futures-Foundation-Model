"""
Futures Foundation Model (FFM)

A pretrained transformer backbone for futures market structure and
regime classification. Built on HuggingFace architecture.

Usage:
    from futures_foundation import FFMConfig, FFMBackbone, FFMForPretraining

    config = FFMConfig(hidden_size=256, num_hidden_layers=6)
    model = FFMForPretraining(config)

Strategy fine-tuning (e.g. ORB):
    from strategies.orb import HybridORBModel, HybridORBDataset
"""


from .config import FFMConfig
from .model import (
    FFMBackbone, FFMForPretraining, FFMForClassification,
    FFMForRegression, FFMForStrategyWithRisk,
)
from .features import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from .labels import (
    generate_all_labels, generate_regime_labels, generate_volatility_labels,
    generate_structure_labels, generate_range_labels, print_label_distribution,
    REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
)
from .dataset import (
    FFMDataset, FFMMultiInstrumentDataset,
    temporal_train_val_split, create_dataloaders,
)

__version__ = "0.2.0"
__all__ = [
    "FFMConfig", "FFMBackbone", "FFMForPretraining", "FFMForClassification",
    "FFMForRegression", "FFMForStrategyWithRisk",
    "derive_features", "get_model_feature_columns", "INSTRUMENT_MAP",
    "generate_all_labels", "generate_regime_labels", "generate_volatility_labels",
    "generate_structure_labels", "generate_range_labels", "print_label_distribution",
    "REGIME_LABELS", "VOLATILITY_LABELS", "STRUCTURE_LABELS", "RANGE_LABELS",
    "FFMDataset", "FFMMultiInstrumentDataset", "temporal_train_val_split", "create_dataloaders",
]