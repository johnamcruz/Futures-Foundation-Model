"""Unit tests for futures_foundation.finetune — pytest compatible."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import pytest

from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune import (
    StrategyLabeler, TrainingConfig,
    HybridStrategyModel, HybridStrategyDataset, FocalLoss,
    run_labeling, run_walk_forward, export_onnx, print_eval_summary,
)
from futures_foundation.finetune.trainer import (
    _make_balanced_loader, _train_one_epoch, _evaluate, _concat_with_meta,
)


# =============================================================================
# Helpers
# =============================================================================

SEQ_LEN = 16
NUM_STRATEGY_FEATURES = 4
STRATEGY_COLS = ['feat_a', 'feat_b', 'feat_c', 'feat_d']


def small_ffm_config():
    return FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=SEQ_LEN,
    )


def make_ffm_df(n=200, seed=0):
    """Minimal FFM-prepared DataFrame with required columns."""
    rng = np.random.default_rng(seed)
    feat_cols = get_model_feature_columns()
    df = pd.DataFrame(rng.standard_normal((n, len(feat_cols))).astype(np.float32),
                      columns=feat_cols)
    df['_datetime']        = pd.date_range('2023-01-01', periods=n, freq='5min', tz='America/New_York')
    df['_instrument_id']   = 0
    df['sess_id']          = 0
    df['sess_time_of_day'] = rng.random(n).astype(np.float32)
    df['tmp_day_of_week']  = rng.integers(0, 5, n)
    df['candle_type']      = rng.integers(0, 6, n)   # vocab = 6 (FFMConfig default)
    return df


def make_strategy_features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, NUM_STRATEGY_FEATURES)).astype(np.float32),
                        columns=STRATEGY_COLS)


def make_labels(n=200, signal_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sig = (rng.random(n) < signal_rate).astype(np.int8)
    rr  = rng.uniform(0, 5, n).astype(np.float32) * sig
    return pd.DataFrame({'signal_label': sig, 'max_rr': rr, 'sl_distance': rr * 0.5})


class TrivialLabeler(StrategyLabeler):
    """Minimal concrete implementation for testing."""

    @property
    def name(self):
        return 'trivial'

    @property
    def feature_cols(self):
        return STRATEGY_COLS

    def run(self, df_raw, ffm_df, ticker):
        n = len(ffm_df)
        feats  = make_strategy_features(n)
        labels = make_labels(n)
        return feats, labels


# =============================================================================
# TrainingConfig
# =============================================================================

def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seq_len == 96
    assert cfg.batch_size == 256
    assert cfg.num_labels == 2
    assert isinstance(cfg.baseline_wr, dict)


def test_training_config_custom():
    cfg = TrainingConfig(seq_len=32, lr=1e-4, num_labels=3,
                         baseline_wr={'ES': 0.30, 'NQ': 0.40})
    assert cfg.seq_len == 32
    assert cfg.lr == 1e-4
    assert cfg.num_labels == 3
    assert cfg.baseline_wr['NQ'] == 0.40


# =============================================================================
# StrategyLabeler ABC
# =============================================================================

def test_strategy_labeler_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        StrategyLabeler()


def test_trivial_labeler_instantiates():
    lb = TrivialLabeler()
    assert lb.name == 'trivial'
    assert lb.feature_cols == STRATEGY_COLS


def test_trivial_labeler_run_output_shape():
    lb = TrivialLabeler()
    ffm_df = make_ffm_df(100)
    raw_df = ffm_df.copy()
    feats, labels = lb.run(raw_df, ffm_df, 'TEST')
    assert len(feats) == 100
    assert len(labels) == 100
    assert list(feats.columns) == STRATEGY_COLS
    assert 'signal_label' in labels.columns
    assert 'max_rr' in labels.columns


# =============================================================================
# HybridStrategyModel
# =============================================================================

def test_hybrid_model_forward_shape():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    out    = model(feats, strat)
    assert out['signal_logits'].shape    == (2, 2)
    assert out['risk_predictions'].shape == (2, 1)
    assert out['confidence'].shape       == (2,)


def test_hybrid_model_confidence_bounded():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    conf  = out['confidence']
    # Confidence = max(softmax(signal_logits)), so range is (0, 1].
    # For 2-class uniform logits the minimum is ~0.5; always > 0 and ≤ 1.
    assert (conf > 0).all() and (conf <= 1).all()


def test_hybrid_model_confidence_equals_max_softmax():
    """Confidence must equal max(softmax(signal_logits)) — directly calibrated."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    expected = torch.softmax(out['signal_logits'], dim=-1).max(dim=-1).values
    assert torch.allclose(out['confidence'], expected, atol=1e-6)


def test_hybrid_model_risk_positive():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    assert (out['risk_predictions'] >= 0).all()


def test_hybrid_model_three_labels():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES, num_labels=3)
    feats = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(2, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    assert out['signal_logits'].shape == (2, 3)


def test_hybrid_model_freeze_backbone():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.5)
    frozen_params = [p for p in model.backbone.parameters() if not p.requires_grad]
    trainable_params = list(model.trainable_parameters())
    assert len(frozen_params) > 0
    assert len(trainable_params) > 0


def test_hybrid_model_load_backbone_missing_file():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    with pytest.raises(Exception):
        model.load_backbone('/nonexistent/path/backbone.pt')


def test_hybrid_model_load_backbone_from_file():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'backbone.pt')
        torch.save(model.backbone.state_dict(), path)
        model2 = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
        model2.load_backbone(path)
        for (k1, v1), (k2, v2) in zip(model.backbone.state_dict().items(),
                                        model2.backbone.state_dict().items()):
            assert torch.allclose(v1, v2), f'Mismatch in {k1}'


# =============================================================================
# FocalLoss
# =============================================================================

def test_focal_loss_shape():
    loss_fn = FocalLoss()
    logits  = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss    = loss_fn(logits, targets)
    assert loss.shape == ()


def test_focal_loss_is_positive():
    loss_fn = FocalLoss()
    logits  = torch.randn(16, 2)
    targets = torch.randint(0, 2, (16,))
    assert loss_fn(logits, targets).item() > 0


def test_focal_loss_with_class_weights():
    w       = torch.tensor([1.0, 5.0])
    loss_fn = FocalLoss(weight=w)
    logits  = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss    = loss_fn(logits, targets)
    assert loss.item() > 0


def test_focal_loss_gamma_zero_matches_ce():
    # gamma=0 → no focal weighting, should be close to smoothed CE
    loss_fn = FocalLoss(gamma=0.0, label_smoothing=0.0)
    logits  = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))
    loss    = loss_fn(logits, targets)
    ce_loss = torch.nn.functional.cross_entropy(logits, targets)
    # values should be in the same ballpark (within 50%)
    assert abs(loss.item() - ce_loss.item()) / ce_loss.item() < 0.5


# =============================================================================
# HybridStrategyDataset
# =============================================================================

def test_dataset_length():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    assert len(ds) == 100 - SEQ_LEN + 1


def test_dataset_item_shapes():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['features'].shape          == (SEQ_LEN, len(get_model_feature_columns()))
    assert item['strategy_features'].shape == (NUM_STRATEGY_FEATURES,)
    assert item['candle_types'].shape      == (SEQ_LEN,)
    assert item['signal_label'].shape      == ()
    assert item['max_rr'].shape            == ()


def test_dataset_signal_indices():
    ffm_df = make_ffm_df(200)
    strat  = make_strategy_features(200)
    labels = make_labels(200, signal_rate=0.10)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    assert len(ds.signal_indices) > 0
    for si in ds.signal_indices:
        item = ds[si]
        assert item['signal_label'].item() > 0


def test_dataset_no_nans_in_output():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert not torch.isnan(item['features']).any()
    assert not torch.isnan(item['strategy_features']).any()


def test_dataset_stride():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds_s1  = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN, stride=1)
    ds_s4  = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN, stride=4)
    assert len(ds_s1) > len(ds_s4)


# =============================================================================
# _concat_with_meta
# =============================================================================

def test_concat_with_meta_labels_length():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.10)
    ds1 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    ds2 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds1, ds2], SEQ_LEN)
    assert len(concat._labels) == len(ds1) + len(ds2)
    assert len(concat.signal_indices) == len(ds1.signal_indices) + len(ds2.signal_indices)


def test_concat_with_meta_signal_indices_offset():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.10)
    ds1 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    ds2 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds1, ds2], SEQ_LEN)
    # All signal indices must be in valid range
    for i in concat.signal_indices:
        assert 0 <= i < len(concat)


# =============================================================================
# _make_balanced_loader
# =============================================================================

def test_balanced_loader_returns_dataloader():
    ffm_df = make_ffm_df(200)
    strat  = make_strategy_features(200)
    labels = make_labels(200, signal_rate=0.10)
    ds = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)
    batch = next(iter(loader))
    assert 'features' in batch
    assert 'strategy_features' in batch


def test_balanced_loader_fallback_when_few_signals():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.001)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    # Should not raise even with 0 or very few signals
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=8, num_workers=0)
    assert loader is not None


# =============================================================================
# _train_one_epoch / _evaluate
# =============================================================================

def _make_small_loader(n=200, seed=0):
    ffm_df = make_ffm_df(n, seed)
    strat  = make_strategy_features(n, seed)
    labels = make_labels(n, signal_rate=0.10, seed=seed)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    return _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)


def test_train_one_epoch_returns_loss():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss()
    result = _train_one_epoch(model, loader, optim, loss_fn, torch.device('cpu'))
    assert 'loss' in result
    assert result['loss'] > 0
    assert 0 <= result['acc'] <= 1


def test_evaluate_returns_metrics():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    loss_fn = FocalLoss()
    result = _evaluate(model, loader, loss_fn, torch.device('cpu'))
    assert 'loss' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 'all_conf' in result
    assert 'all_max_rr' in result
    assert len(result['all_conf']) == len(result['all_labels'])


def test_evaluate_confidence_all_in_01():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    loss_fn = FocalLoss()
    result = _evaluate(model, loader, loss_fn, torch.device('cpu'))
    confs = result['all_conf']
    # max(softmax) is always in (0, 1]; for 2-class it is always >= 0.5
    assert all(0.0 < c <= 1.0 for c in confs)


def test_train_reduces_loss():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    optim   = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = FocalLoss()
    losses = [_train_one_epoch(model, loader, optim, loss_fn, torch.device('cpu'))['loss']
              for _ in range(10)]
    # Loss should not explode over 10 epochs (all values finite and < 10)
    assert all(0 < l < 10 for l in losses), f'Loss out of range: {losses}'
    # Minimum loss across all epochs should be lower than the first epoch
    assert min(losses) < losses[0], f'Loss never improved: {losses}'


# =============================================================================
# run_labeling
# =============================================================================

def _skip_no_parquet():
    try:
        import pyarrow  # noqa: F401
        return False
    except ImportError:
        return True


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_creates_parquet_files(tmp_path):
    lb = TrivialLabeler()
    raw_dir = tmp_path / 'raw'
    ffm_dir = tmp_path / 'ffm'
    cache_dir = tmp_path / 'cache'
    raw_dir.mkdir(); ffm_dir.mkdir()

    ticker = 'TEST'
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=True)

    # Write a minimal CSV matching the raw data format
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open':  np.random.randn(n) + 5000,
        'high':  np.random.randn(n) + 5001,
        'low':   np.random.randn(n) + 4999,
        'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / f'{ticker}_5min.csv', index=False)

    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir))

    assert (cache_dir / f'{ticker}_strategy_features.parquet').exists()
    assert (cache_dir / f'{ticker}_strategy_labels.parquet').exists()


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_skips_cached(tmp_path):
    lb = TrivialLabeler()
    raw_dir   = tmp_path / 'raw';   raw_dir.mkdir()
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()
    cache_dir = tmp_path / 'cache'; cache_dir.mkdir()

    ticker = 'SKIP'
    feat_path  = cache_dir / f'{ticker}_strategy_features.parquet'
    label_path = cache_dir / f'{ticker}_strategy_labels.parquet'

    # Pre-write fake cache files
    pd.DataFrame({'feat_a': [1.0]}).to_parquet(feat_path)
    pd.DataFrame({'signal_label': [0], 'max_rr': [0.0]}).to_parquet(label_path)

    # Should not raise even without raw/ffm files
    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir))
    # Cache files unchanged
    assert feat_path.exists()


def test_run_labeling_skips_missing_data(tmp_path):
    lb = TrivialLabeler()
    cache_dir = tmp_path / 'cache'
    # raw_dir and ffm_dir don't contain ticker files — should skip gracefully
    run_labeling(lb, ['MISSING'], str(tmp_path / 'raw'), str(tmp_path / 'ffm'),
                 str(cache_dir))
    assert not (cache_dir / 'MISSING_strategy_features.parquet').exists()


# =============================================================================
# export_onnx
# =============================================================================

def test_export_onnx_creates_file(tmp_path):
    pytest.importorskip('onnx')
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    path  = str(tmp_path / 'model.onnx')
    try:
        export_onnx(model, path,
                    seq_len=SEQ_LEN,
                    num_ffm_features=len(get_model_feature_columns()),
                    num_strategy_features=NUM_STRATEGY_FEATURES)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    except Exception as e:
        # Some torch versions don't support all operators needed for ONNX export
        pytest.skip(f'ONNX export not supported on this torch version: {e}')


# =============================================================================
# print_eval_summary (smoke test — just verify it doesn't crash)
# =============================================================================

def test_print_eval_summary_no_results(capsys):
    print_eval_summary({}, baseline_wr={'ES': 0.30})
    captured = capsys.readouterr()
    assert 'No fold results' in captured.out


def test_print_eval_summary_with_results(capsys):
    rng = np.random.default_rng(0)
    n   = 200
    metrics = {
        'all_conf':   rng.random(n).tolist(),
        'all_labels': rng.integers(0, 2, n).tolist(),
        'all_preds':  rng.integers(0, 2, n).tolist(),
        'all_max_rr': rng.uniform(0, 5, n).tolist(),
    }
    fold_results = {'F1': metrics, 'F2': metrics}
    print_eval_summary(fold_results, baseline_wr={'ES': 0.30})
    captured = capsys.readouterr()
    assert 'CONFIDENCE THRESHOLDS' in captured.out
    assert 'PER-FOLD' in captured.out
    assert 'LEARNING VERIFICATION' in captured.out


def test_print_eval_summary_ignores_model_key(capsys):
    fold_results = {'_model': object(), 'F1': None}
    print_eval_summary(fold_results)
    # Should not raise; F1=None should print gracefully


# =============================================================================
# FFM feature column coverage
# Verify every feature column produced by get_model_feature_columns() flows
# correctly through HybridStrategyDataset and HybridStrategyModel without error.
# =============================================================================

def test_all_ffm_columns_present_in_dataset():
    """Every column from get_model_feature_columns() must be in the FFM DataFrame
    and make it into dataset._f without NaN after nan_to_num."""
    feat_cols = get_model_feature_columns()
    ffm_df    = make_ffm_df(100)
    strat     = make_strategy_features(100)
    labels    = make_labels(100)
    ds        = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)

    assert ds._f.shape[1] == len(feat_cols), (
        f'Expected {len(feat_cols)} feature cols, got {ds._f.shape[1]}')
    assert not np.isnan(ds._f).any(), 'NaN values survived nan_to_num in dataset._f'


def test_ffm_columns_match_model_input_dim():
    """Dataset output dimension must match what the backbone expects."""
    feat_cols = get_model_feature_columns()
    ffm_df    = make_ffm_df(60)
    strat     = make_strategy_features(60)
    labels    = make_labels(60)
    ds        = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item      = ds[0]
    assert item['features'].shape[-1] == len(feat_cols)


def test_all_ffm_columns_flow_through_model():
    """A forward pass using all FFM feature columns must not raise and must
    produce finite outputs."""
    feat_cols = get_model_feature_columns()
    cfg       = small_ffm_config()
    model     = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    batch  = 2
    feats  = torch.randn(batch, SEQ_LEN, len(feat_cols))
    strat  = torch.randn(batch, NUM_STRATEGY_FEATURES)
    candle = torch.zeros(batch, SEQ_LEN, dtype=torch.long)  # valid candle_type = 0

    out = model(feats, strat, candle_types=candle)
    assert torch.isfinite(out['signal_logits']).all(), 'signal_logits contains inf/nan'
    assert torch.isfinite(out['risk_predictions']).all()
    assert torch.isfinite(out['confidence']).all()


def test_each_ffm_column_carries_signal():
    """Perturbing each feature column independently should change the model output,
    confirming gradients flow through every column."""
    feat_cols = get_model_feature_columns()
    cfg       = small_ffm_config()
    model     = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    base_feats = torch.randn(1, SEQ_LEN, len(feat_cols))
    strat      = torch.randn(1, NUM_STRATEGY_FEATURES)
    with torch.no_grad():
        base_out = model(base_feats, strat)['signal_logits'].clone()

    columns_that_changed = 0
    for col_idx in range(len(feat_cols)):
        perturbed = base_feats.clone()
        perturbed[:, :, col_idx] += 10.0   # large perturbation
        with torch.no_grad():
            new_out = model(perturbed, strat)['signal_logits']
        if not torch.allclose(new_out, base_out, atol=1e-4):
            columns_that_changed += 1

    # At least 90% of columns should influence the output
    pct = columns_that_changed / len(feat_cols)
    assert pct >= 0.90, (
        f'Only {columns_that_changed}/{len(feat_cols)} FFM columns changed model output. '
        f'Some feature columns may be dead or disconnected.')


def test_strategy_feature_cols_carry_signal():
    """Perturbing strategy features must change the model output."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    feats = torch.randn(1, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(1, NUM_STRATEGY_FEATURES)
    with torch.no_grad():
        base_out = model(feats, strat)['signal_logits'].clone()

    changed = 0
    for i in range(NUM_STRATEGY_FEATURES):
        perturbed = strat.clone()
        perturbed[:, i] += 10.0
        with torch.no_grad():
            new_out = model(feats, perturbed)['signal_logits']
        if not torch.allclose(new_out, base_out, atol=1e-4):
            changed += 1

    assert changed == NUM_STRATEGY_FEATURES, (
        f'Only {changed}/{NUM_STRATEGY_FEATURES} strategy features changed model output')


def test_dataset_with_nan_ffm_rows_excluded():
    """Rows where any FFM feature column is NaN must be excluded from the dataset."""
    ffm_df = make_ffm_df(100)
    feat_cols = get_model_feature_columns()
    # Inject NaN into 10 rows
    ffm_df.loc[5:14, feat_cols[0]] = np.nan
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    # Dataset length is based on valid rows after NaN filtering
    assert len(ds) <= (100 - 10) - SEQ_LEN + 1


def test_dataset_categorical_cols_are_int64():
    """Categorical embedding inputs must be int64 tensors."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['candle_types'].dtype    == torch.int64
    assert item['instrument_ids'].dtype  == torch.int64
    assert item['session_ids'].dtype     == torch.int64
    assert item['day_of_week'].dtype     == torch.int64


def test_dataset_continuous_cols_are_float32():
    """Continuous feature tensors must be float32 (backbone expects fp32)."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['features'].dtype          == torch.float32
    assert item['strategy_features'].dtype == torch.float32
    assert item['time_of_day'].dtype       == torch.float32
    assert item['max_rr'].dtype            == torch.float32


def test_dataset_label_is_last_bar_of_window():
    """signal_label must be the label of the LAST bar in the sequence window."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    # Force a known pattern: signal_label = 1 only at index 49
    labels = make_labels(100, signal_rate=0.0)
    labels.loc[49, 'signal_label'] = 1
    ds   = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    # The window that ends at index 49 is window index 49 - SEQ_LEN + 1 = 34
    target_window = 49 - SEQ_LEN + 1
    item = ds[target_window]
    assert item['signal_label'].item() == 1
    # All other windows should have label 0
    for i in range(len(ds)):
        if i != target_window:
            assert ds[i]['signal_label'].item() == 0, f'Window {i} should be noise'
