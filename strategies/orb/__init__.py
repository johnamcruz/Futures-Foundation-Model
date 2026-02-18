"""
ORB (Opening Range Breakout) Strategy
======================================

Hybrid fine-tuning: FFM backbone (market context) + ORB-specific features
(breakout pattern) → signal/risk predictions.

Usage:
    from strategies.orb import (
        HybridORBModel, HybridORBDataset,
        label_instrument, align_orb_to_ffm,
        create_orb_features, ORB_FEATURE_COLS,
    )
"""

from .model import HybridORBModel
from .dataset import HybridORBDataset
from .features import create_orb_features, ORB_FEATURE_COLS, NUM_ORB_FEATURES, SESSION_DEFS
from .labeler import (
    label_instrument, label_session, align_orb_to_ffm,
    detect_session_bars, compute_orb_range,
)

__all__ = [
    "HybridORBModel", "HybridORBDataset",
    "create_orb_features", "ORB_FEATURE_COLS", "NUM_ORB_FEATURES", "SESSION_DEFS",
    "label_instrument", "label_session", "align_orb_to_ffm",
    "detect_session_bars", "compute_orb_range",
]
