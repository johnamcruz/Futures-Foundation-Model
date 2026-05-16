"""
Fine-tuning framework for strategy-specific models on top of the FFM backbone.

Usage — implement one class, get everything else for free:

    from futures_foundation.finetune import StrategyLabeler, TrainingConfig, run_walk_forward

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

    fold_results = run_walk_forward(
        folds=FOLDS, tickers=tickers, ffm_dir=ffm_dir,
        strategy_dir=cache_dir, output_dir=output_dir,
        backbone_path=backbone_path, ffm_config=config,
        training_cfg=TrainingConfig(), num_strategy_features=len(labeler.feature_cols),
        strategy_feature_cols=labeler.feature_cols,
    )

    print_eval_summary(fold_results, baseline_wr=BASELINE_WR)
"""

from .base import StrategyLabeler
from .config import TrainingConfig
from .health import FoldHealthMonitor
from .dataset import HybridStrategyDataset
from .losses import FocalLoss
from .model import HybridStrategyModel
from .trainer import (
    export_onnx,
    extract_backbone,
    print_eval_summary,
    print_fold_progression,
    print_rr_calibration,
    run_finetune,
    run_labeling,
    run_risk_head_calibration,
    run_shuffle_audit,
    run_walk_forward,
    summarize_fold_precision,
    validate_setup,
)

__all__ = [
    'StrategyLabeler',
    'TrainingConfig',
    'FoldHealthMonitor',
    'HybridStrategyDataset',
    'FocalLoss',
    'HybridStrategyModel',
    'run_finetune',
    'run_labeling',
    'run_walk_forward',
    'run_risk_head_calibration',
    'run_shuffle_audit',
    'print_rr_calibration',
    'export_onnx',
    'extract_backbone',
    'print_eval_summary',
    'print_fold_progression',
    'summarize_fold_precision',
    'validate_setup',
]
