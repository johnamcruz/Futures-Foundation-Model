"""
Strategy labeling, reporting, and realized-R economics.

The torch walk-forward trainer was retired with the from-scratch FFM
backbone (see git tag `ffm-transformer-final`). Walk-forward training now
lives in `futures_foundation.chronos` (frozen Chronos-Bolt embeddings + XGBoost).
This package keeps the torch-free layers that any strategy pipeline reuses.

IMPORT CONTRACT: fully torch-free — safe to import eagerly from the
`futures_foundation` parent (torch+xgboost segfault in one macOS process).

Usage — implement one labeler class, get the label mechanics for free:

    from futures_foundation.finetune import StrategyLabeler, run_labeling

    class MyStrategyLabeler(StrategyLabeler):
        name = 'my_strategy'
        feature_cols = ['zone_height', 'entry_depth', ...]

        def detect_events(self, df_raw, ffm_df, ticker):
            # one row per signal bar:
            #   bar_idx, direction(+1/-1), sl_distance(>0), tp_rr(>=1)
            return events_df

        def compute_features(self, df_raw, ffm_df, ticker):
            # feature_cols matrix aligned to ffm_df.index
            return strategy_features_df

        # run() is FINAL — the base applies the session-calibrated TP>=SL
        # triple barrier (entry = next-bar open) and emits signal_label /
        # max_rr / sl_distance / direction for free.

    labeler = MyStrategyLabeler()
    run_labeling(labeler, tickers, raw_dir, ffm_dir, cache_dir)
"""

from .base import StrategyLabeler
from .config import TrainingConfig
from .health import FoldHealthMonitor
from .trainer import (
    print_eval_summary,
    print_fold_progression,
    run_labeling,
    summarize_fold_precision,
)

__all__ = [
    'StrategyLabeler',
    'TrainingConfig',
    'FoldHealthMonitor',
    'run_labeling',
    'print_eval_summary',
    'print_fold_progression',
    'summarize_fold_precision',
]
