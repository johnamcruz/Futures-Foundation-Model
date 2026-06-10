"""Unit tests for futures_foundation.finetune — pytest compatible.

The torch walk-forward trainer was retired with the FFM backbone (tag
`ffm-transformer-final`); these tests cover the surviving torch-free
surface: StrategyLabeler/base labeling, run_labeling, FoldHealthMonitor,
reporting helpers, and the realized-R economics helpers.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from futures_foundation import get_model_feature_columns
from futures_foundation.finetune import (
    StrategyLabeler, TrainingConfig, FoldHealthMonitor,
    run_labeling, print_eval_summary,
    print_fold_progression, summarize_fold_precision,
)
from futures_foundation.finetune.trainer import (
    _validate_labeler_output,
    _print_test_threshold_table,
    _print_confidence_calibration,
)


# =============================================================================
# Helpers
# =============================================================================

SEQ_LEN = 16
NUM_STRATEGY_FEATURES = 4
STRATEGY_COLS = ['feat_a', 'feat_b', 'feat_c', 'feat_d']


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


def make_raw(n=200, seed=0, trend=2.0, start=5000.0):
    """Raw OHLCV with a DatetimeIndex matching make_ffm_df's range and a
    gentle uptrend so long triple-barrier events resolve as wins
    deterministically (borrow #4: base run() needs real df_raw OHLC)."""
    base = start + np.arange(n) * trend
    idx  = pd.date_range('2023-01-01', periods=n, freq='5min',
                          tz='America/New_York')
    return pd.DataFrame(
        {'open': base, 'high': base + 1.0, 'low': base - 1.0,
         'close': base, 'volume': np.full(n, 500.0)}, index=idx)


class TrivialLabeler(StrategyLabeler):
    """Minimal concrete implementation for testing (borrow #4 ABC):
    only detect_events() + compute_features(); base run() does the rest."""

    @property
    def name(self):
        return 'trivial'

    @property
    def feature_cols(self):
        return STRATEGY_COLS

    def compute_features(self, df_raw, ffm_df, ticker):
        return make_strategy_features(len(ffm_df))

    def detect_events(self, df_raw, ffm_df, ticker):
        n = len(ffm_df)
        idx = list(range(20, n - 5, 4))          # many → wins exist on any data
        return pd.DataFrame({
            'bar_idx':     idx,
            'direction':   1,                    # long
            'sl_distance': 1.0,
            'tp_rr':       1.0,                  # TP == SL (>=1 => valid)
        })


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
    ffm_df.index = pd.to_datetime(ffm_df['_datetime'])   # base run() needs DT index
    raw_df = make_raw(100)
    feats, labels = lb.run(raw_df, ffm_df, 'TEST')
    assert len(feats) == 100
    assert len(labels) == 100
    assert list(feats.columns) == STRATEGY_COLS
    # borrow #4 contract: signal_label/max_rr/sl_distance + direction (free)
    assert {'signal_label', 'max_rr', 'sl_distance', 'direction'} <= set(labels.columns)
    assert (labels['signal_label'] > 0).sum() > 0        # uptrend → longs win



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


# ── use_cache / config_dict ───────────────────────────────────────────────────

from futures_foundation.finetune.base import StrategyLabeler as _StrategyLabeler
from futures_foundation.finetune.trainer import _labeling_cache_hash


class _VersionedLabeler(TrivialLabeler):
    """TrivialLabeler with a config_dict for cache tests."""
    def __init__(self, version=1):
        self._version = version

    def config_dict(self):
        return {'version': self._version}


def _write_minimal_cache(cache_dir, tickers):
    """Write fake parquet files and a valid hash file for a given labeler."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lb = _VersionedLabeler(version=1)
    h  = _labeling_cache_hash(lb, tickers, '5min')
    (cache_dir / 'labeling_hash.txt').write_text(h)
    for t in tickers:
        pd.DataFrame({'signal_label': [0], 'max_rr': [0.0]}).to_parquet(
            cache_dir / f'{t}_strategy_labels.parquet'
        )
        pd.DataFrame({'feat_a': [1.0]}).to_parquet(
            cache_dir / f'{t}_strategy_features.parquet'
        )


def test_config_dict_default_returns_empty():
    assert TrivialLabeler().config_dict() == {}


def test_labeling_cache_hash_changes_with_config_dict():
    h1 = _labeling_cache_hash(_VersionedLabeler(version=1), ['ES'], '5min')
    h2 = _labeling_cache_hash(_VersionedLabeler(version=2), ['ES'], '5min')
    assert h1 != h2


def test_labeling_cache_hash_changes_with_tickers():
    lb = _VersionedLabeler()
    h1 = _labeling_cache_hash(lb, ['ES'], '5min')
    h2 = _labeling_cache_hash(lb, ['ES', 'NQ'], '5min')
    assert h1 != h2


def test_labeling_cache_hash_changes_with_timeframe():
    lb = _VersionedLabeler()
    assert _labeling_cache_hash(lb, ['ES'], '5min') != _labeling_cache_hash(lb, ['ES'], '3min')


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_hit_skips(tmp_path, capsys):
    tickers   = ['HIT']
    cache_dir = tmp_path / 'cache'
    _write_minimal_cache(cache_dir, tickers)

    lb = _VersionedLabeler(version=1)
    run_labeling(lb, tickers, str(tmp_path / 'raw'), str(tmp_path / 'ffm'),
                 str(cache_dir), use_cache=True)

    out = capsys.readouterr().out
    assert 'cache hit' in out


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_miss_wipes_and_relabels(tmp_path):
    tickers   = ['ES']
    cache_dir = tmp_path / 'cache'
    _write_minimal_cache(cache_dir, tickers)

    # Stale file that should be wiped on cache miss
    stale = cache_dir / 'stale.txt'
    stale.write_text('should be gone')

    # Different version → hash mismatch → cache invalid
    lb = _VersionedLabeler(version=99)
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    ffm_dir = tmp_path / 'ffm'; ffm_dir.mkdir()
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / 'ES_features.parquet', index=True)
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open': np.random.randn(n) + 5000, 'high': np.random.randn(n) + 5001,
        'low':  np.random.randn(n) + 4999, 'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / 'ES_5min.csv', index=False)

    run_labeling(lb, tickers, str(raw_dir), str(ffm_dir), str(cache_dir), use_cache=True)

    assert not stale.exists(), 'cache dir should have been wiped on hash mismatch'
    assert (cache_dir / 'labeling_hash.txt').exists()


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_writes_hash_file(tmp_path):
    ticker  = 'WR'
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    ffm_dir = tmp_path / 'ffm'; ffm_dir.mkdir()
    cache_dir = tmp_path / 'cache'
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=True)
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open': np.random.randn(n) + 5000, 'high': np.random.randn(n) + 5001,
        'low':  np.random.randn(n) + 4999, 'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / f'{ticker}_5min.csv', index=False)

    lb = _VersionedLabeler(version=7)
    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir), use_cache=True)

    hash_file = cache_dir / 'labeling_hash.txt'
    assert hash_file.exists()
    expected = _labeling_cache_hash(lb, [ticker], '5min')
    assert hash_file.read_text().strip() == expected


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
# _validate_labeler_output — labeler contract enforcement
# =============================================================================

def test_validate_labeler_output_passes_valid():
    """Valid labeler output must pass without raising."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n, signal_rate=0.05)
    _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_misaligned_features():
    """Raises when strategy_features row count != ffm_df row count."""
    n = 100
    feats  = make_strategy_features(n - 5)   # wrong length
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='strategy_features'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_misaligned_labels():
    """Raises when labels_df row count != ffm_df row count."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n - 5, signal_rate=0.05)  # wrong length
    with pytest.raises(ValueError, match='labels_df'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_feature_col():
    """Raises when a declared feature_col is absent from strategy_features."""
    n = 100
    feats  = make_strategy_features(n)[['feat_a', 'feat_b']]  # drop feat_c, feat_d
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='missing columns'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_signal_label_col():
    """Raises when labels_df is missing the 'signal_label' column."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n).drop(columns=['signal_label'])
    with pytest.raises(ValueError, match="signal_label"):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_max_rr_col():
    """Raises when labels_df is missing the 'max_rr' column."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n).drop(columns=['max_rr'])
    with pytest.raises(ValueError, match="max_rr"):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_zero_signals():
    """Raises when there are no positive signal labels — training would be useless."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n, signal_rate=0.0)  # all zeros
    with pytest.raises(ValueError, match='0 signals'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_error_includes_ticker():
    """Error message must include the ticker so multi-ticker logs are easy to triage."""
    n = 100
    feats  = make_strategy_features(n - 1)  # misaligned
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='NQ'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'NQ')



# =============================================================================
# Reporting helpers
# =============================================================================

def _make_fake_test_metrics(n=100, seed=3):
    """Minimal test_metrics dict matching what _evaluate returns."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).tolist()
    preds  = rng.integers(0, 2, n).tolist()
    confs  = rng.uniform(0.5, 1.0, n).tolist()
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    return {
        'loss': 0.05, 'precision': 0.4, 'recall': 0.6, 'f1': 0.48,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': n - tp - fp - fn,
        'all_conf': confs, 'all_labels': labels,
        'all_preds': preds, 'all_max_rr': [1.0] * n,
    }

def test_print_test_threshold_table_smoke(capsys):
    """_print_test_threshold_table must not raise and must print threshold rows."""
    metrics = _make_fake_test_metrics(n=200)
    _print_test_threshold_table(metrics, 'F1')
    out = capsys.readouterr().out
    assert 'F1 test:' in out
    assert '0.70' in out


def test_print_test_threshold_table_recall_uses_total_signals(capsys):
    """Recall must equal htp / n_total_actual_signals, not htp / (htp + hfn_in_mask).

    Bug: hfn was computed only within conf>=thresh mask, so FN bars with low
    confidence were excluded, producing recall=1.0 at every threshold.  Fix:
    denominator is always n_sig = tp + fn from the full test set.
    """
    n = 1000
    rng = np.random.default_rng(42)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.55)  # just above 0.50 threshold

    # 10 actual signals
    sig_idx = rng.choice(n, 10, replace=False)
    labels[sig_idx] = 1

    # Model correctly predicts 6 of them with HIGH confidence
    tp_idx = sig_idx[:6]
    preds[tp_idx]  = 1
    confs[tp_idx]  = 0.90  # above 0.80 thresh

    # Model misses 4 signals with LOW confidence (pred=0, conf just above 0.50)
    fn_idx = sig_idx[6:]
    preds[fn_idx]  = 0
    confs[fn_idx]  = 0.52  # above 0.50 but below 0.80 → excluded from 0.80 mask

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.6, 'f1': 0.55,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_test_threshold_table(metrics, 'FX')
    out = capsys.readouterr().out

    # At thresh=0.80: only the 6 high-conf TPs are in the mask
    # Correct recall = 6 / 10 = 60.0% (not 100.0%)
    # Row format: Thresh  N  Correct  Prec  EV@2R  Recall  Rate  Status
    lines = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert lines, 'No 0.80 threshold row printed'
    recall_str = lines[0].split()[5]  # index 5 is Recall (printed as XX.X%)
    recall_val = float(recall_str.rstrip('%')) / 100.0
    assert abs(recall_val - 0.600) < 0.001, (
        f'Recall at 0.80 should be 60.0% (6/10 total signals), got {recall_str}. '
        'Check that denominator uses n_sig not (htp + hfn_in_mask).'
    )


def test_print_test_threshold_table_none_is_noop(capsys):
    """None test_metrics must silently produce no output."""
    _print_test_threshold_table(None, 'F2')
    assert capsys.readouterr().out == ''


def test_print_test_threshold_table_ev_and_status(capsys):
    """EV@2R and status flag must be printed and reflect precision correctly.

    At P=50% and rr_target=2: EV = 0.50*3 - 1 = +0.50R → VIABLE.
    At P=10% and rr_target=2: EV = 0.10*3 - 1 = -0.70R → NOT VIABLE.
    """
    n = 500
    rng = np.random.default_rng(0)

    # Scenario A: 50% precision at 0.80 threshold (10 TP, 10 FP, 10 correct)
    confs  = np.full(n, 0.40)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)

    sig_idx = rng.choice(n, 20, replace=False)
    labels[sig_idx] = 1
    # 10 TP at high confidence
    for i in sig_idx[:10]:
        preds[i] = 1; confs[i] = 0.90
    # 10 FP at high confidence
    fp_idx = rng.choice([i for i in range(n) if i not in sig_idx], 10, replace=False)
    for i in fp_idx:
        preds[i] = 1; confs[i] = 0.85

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_test_threshold_table(metrics, 'FX', rr_target=2.0)
    out = capsys.readouterr().out

    # EV breakeven note must appear
    assert 'Breakeven' in out, 'Breakeven note missing from output'
    assert 'EV@2R' in out or 'EV@2' in out, 'EV column header missing'

    # At 0.80 threshold with 50% precision: EV = +0.50R → VIABLE
    rows_80 = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert rows_80, 'No 0.80 threshold row printed'
    assert 'VIABLE' in rows_80[0], f'Expected VIABLE status at 50% prec, got: {rows_80[0]}'
    assert '+' in rows_80[0], f'Expected positive EV at 50% prec, got: {rows_80[0]}'


# =============================================================================
# _print_test_threshold_table — AvgMaxRR column
# =============================================================================

def test_threshold_table_includes_avg_rr(capsys):
    """AvgMaxRR column must appear in threshold table output when all_max_rr provided."""
    n = 500
    rng = np.random.default_rng(7)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.40)
    # 20 TPs at 0.85 confidence with max_rr=3.0
    tp_idx = rng.choice(n, 20, replace=False)
    labels[tp_idx] = 1
    preds[tp_idx]  = 1
    confs[tp_idx]  = 0.85
    max_rr = np.zeros(n)
    max_rr[tp_idx] = 3.0

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': max_rr.tolist(),
    }
    _print_test_threshold_table(metrics, 'F1')
    out = capsys.readouterr().out

    assert 'AvgRR' in out, 'AvgRR column header must appear'
    rows_80 = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert rows_80, 'No 0.80 threshold row printed'
    assert '3.00R' in rows_80[0], (
        f'Expected 3.00R avg max RR at 0.80 threshold, got: {rows_80[0]}')


def test_threshold_table_avg_rr_dash_when_no_max_rr(capsys):
    """AvgRR column shows dash when all_max_rr is absent from metrics."""
    metrics = _make_fake_test_metrics(n=200)
    metrics_no_rr = {k: v for k, v in metrics.items() if k != 'all_max_rr'}
    _print_test_threshold_table(metrics_no_rr, 'F1')
    out = capsys.readouterr().out
    assert 'AvgRR' in out
    assert '—' in out


# =============================================================================
# _print_confidence_calibration
# =============================================================================

def _make_calibration_metrics(win_rates_by_band, n_per_band=50, seed=0):
    """Build test_metrics where predicted positives have specified win rates per band.

    win_rates_by_band: list of (lo, mid_conf, win_rate) — one entry per band.
    """
    rng = np.random.default_rng(seed)
    all_conf = []; all_labels = []; all_preds = []

    for lo, mid_conf, wr in win_rates_by_band:
        n_wins   = int(n_per_band * wr)
        n_losses = n_per_band - n_wins
        # wins: label=1, pred=1
        all_conf.extend([mid_conf] * n_wins)
        all_labels.extend([1] * n_wins)
        all_preds.extend([1] * n_wins)
        # losses: label=0, pred=1
        all_conf.extend([mid_conf] * n_losses)
        all_labels.extend([0] * n_losses)
        all_preds.extend([1] * n_losses)

    n = len(all_labels)
    tp = sum(1 for l, p in zip(all_labels, all_preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 1)
    fn = 0
    return {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': all_conf, 'all_labels': all_labels,
        'all_preds': all_preds, 'all_max_rr': [1.0] * n,
    }


def test_confidence_calibration_monotonic_flagged(capsys):
    """Monotonically rising win rates must print '✅ monotonic'."""
    metrics = _make_calibration_metrics([
        (0.50, 0.55, 0.10),
        (0.60, 0.65, 0.20),
        (0.70, 0.75, 0.35),
        (0.80, 0.85, 0.55),
        (0.90, 0.95, 0.75),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    assert '✅ monotonic' in out, f'Expected monotonic flag, got:\n{out}'
    assert 'Confidence calibration' in out


def test_confidence_calibration_non_monotonic_flagged(capsys):
    """Win rate dropping between bands must print non-monotonic warning."""
    metrics = _make_calibration_metrics([
        (0.50, 0.55, 0.10),
        (0.60, 0.65, 0.50),   # high
        (0.70, 0.75, 0.15),   # drops sharply — non-monotonic
        (0.80, 0.85, 0.55),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    assert 'non-monotonic' in out, f'Expected non-monotonic warning, got:\n{out}'


def test_confidence_calibration_filters_predicted_positives(capsys):
    """Only predicted positives (pred > 0) must count — noise predictions excluded."""
    n = 200
    rng = np.random.default_rng(5)
    # Half the bars are predicted noise (pred=0) with high conf — must be ignored
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.85)

    # 20 true positives predicted as signal
    tp_idx = rng.choice(n, 20, replace=False)
    labels[tp_idx] = 1
    preds[tp_idx]  = 1

    # 80 noise bars predicted as signal (FP) — win rate ~20%
    fp_idx = rng.choice([i for i in range(n) if i not in tp_idx], 80, replace=False)
    preds[fp_idx] = 1

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp_c = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp_c, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out

    # Win rate at 0.8-0.9 band: 20 wins / (20+80) = 20%
    # If noise bars (pred=0) were wrongly included, win rate would be lower
    lines = [l for l in out.splitlines() if '0.8' in l and '–' in l]
    assert lines, 'Expected 0.8–0.9 band row'
    assert '20.0%' in lines[0], (
        f'Win rate must reflect predicted positives only (20/100=20%), got: {lines[0]}')


def test_confidence_calibration_none_is_noop(capsys):
    """None test_metrics must produce no output."""
    _print_confidence_calibration(None)
    assert capsys.readouterr().out == ''


def test_confidence_calibration_deploy_marker(capsys):
    """Bands at 0.80+ must include the ◄ deploy marker."""
    metrics = _make_calibration_metrics([
        (0.70, 0.75, 0.20),
        (0.80, 0.85, 0.55),
        (0.90, 0.95, 0.75),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    deploy_lines = [l for l in out.splitlines() if '◄' in l]
    assert len(deploy_lines) >= 1, 'Deploy marker ◄ must appear for 0.80+ bands'
    assert all('0.8' in l or '0.9' in l for l in deploy_lines), (
        'Deploy marker must only appear on 0.80+ bands')



# ── summarize_fold_precision ───────────────────────────────────────────────────

def _make_fold_results(conf_vals, label_vals):
    """Helper: build a minimal fold_results dict from flat arrays."""
    return {
        'fold_1': {'all_conf': list(conf_vals), 'all_labels': list(label_vals)},
        '_model': None,
    }


def test_summarize_fold_precision_signal_count():
    confs  = [0.50, 0.75, 0.85, 0.55, 0.90]
    labels = [1,    1,    0,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['signals'] == 3  # labels > 0


def test_summarize_fold_precision_at_threshold():
    # conf >= 0.80: indices 2 (label=0) and 4 (label=1) → prec = 0.5
    confs  = [0.50, 0.75, 0.85, 0.55, 0.90]
    labels = [1,    1,    0,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['prec_at_80'] == pytest.approx(0.5, abs=0.001)


def test_summarize_fold_precision_none_when_no_trades():
    # No confs reach 0.90
    confs  = [0.50, 0.60, 0.70]
    labels = [1,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['prec_at_90'] is None


def test_summarize_fold_precision_skips_model_key():
    fold_results = {
        '_model': {'weights': 'whatever'},
        'fold_1': {'all_conf': [0.85], 'all_labels': [1]},
    }
    result = summarize_fold_precision(fold_results)
    assert '_model' not in result
    assert 'fold_1' in result


def test_summarize_fold_precision_skips_none_fold():
    fold_results = {
        'fold_1': {'all_conf': [0.85], 'all_labels': [1]},
        'fold_2': None,
    }
    result = summarize_fold_precision(fold_results)
    assert 'fold_1' in result
    assert 'fold_2' not in result


# ── print_fold_progression ────────────────────────────────────────────────────

def _make_progression_results(f1_prec, f2_prec, f3_prec, n_per_fold=20):
    """Build fold_results with controlled P@80 per fold."""
    def _fold(prec):
        n_pos = round(n_per_fold * prec)
        confs  = [0.85] * n_per_fold
        labels = [1] * n_pos + [0] * (n_per_fold - n_pos)
        return {'all_conf': confs, 'all_labels': labels}

    return {
        'F1': _fold(f1_prec),
        'F2': _fold(f2_prec),
        'F3': _fold(f3_prec),
        '_model': None,
    }


def test_print_fold_progression_gate2_pass(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert '✅ PASS' in out
    assert 'F1=' in out and 'F3=' in out


def test_print_fold_progression_gate2_fail(capsys):
    fold_results = _make_progression_results(0.60, 0.55, 0.50)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert '❌ FAIL' in out


def test_print_fold_progression_ref_column(capsys):
    fold_results = _make_progression_results(0.55, 0.60, 0.65)
    ref = {'F1': 0.527, 'F2': 0.577, 'F3': 0.657}
    print_fold_progression(fold_results, ref=ref, ref_label='v17')
    out = capsys.readouterr().out
    assert 'vs v17:' in out


def test_print_fold_progression_no_ref_no_ref_column(capsys):
    fold_results = _make_progression_results(0.55, 0.60, 0.65)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'vs ' not in out


def test_print_fold_progression_missing_fold_prints_dash(capsys):
    fold_results = {'F1': {'all_conf': [0.85], 'all_labels': [1]}, '_model': None}
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'F2' in out and '—' in out


def test_print_fold_progression_custom_gate2_desc(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results, gate2_desc='cross-timeframe transfer')
    out = capsys.readouterr().out
    assert 'cross-timeframe transfer' in out


def test_print_fold_progression_includes_header(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'FOLD-TO-FOLD LEARNING PROGRESSION' in out
    assert 'P@80' in out and 'N@80' in out and 'Delta' in out


# =============================================================================
# FoldHealthMonitor
# =============================================================================

def _make_metrics(all_conf, all_labels, best_epoch=None, feature_importance=None):
    """Build a minimal test_metrics dict for health monitor tests."""
    conf_arr = np.array(all_conf, dtype=float)
    lab_arr  = np.array(all_labels, dtype=int)
    preds    = (conf_arr >= 0.5).astype(int)
    mask_80  = (conf_arr >= 0.80) & (preds > 0)
    n_at_80  = int(mask_80.sum())
    prec_80  = float((lab_arr[mask_80] > 0).mean()) if n_at_80 > 0 else 0.0
    m = {
        'all_conf':   conf_arr.tolist(),
        'all_labels': lab_arr.tolist(),
        'all_preds':  preds.tolist(),
        'tp': int(((preds > 0) & (lab_arr > 0)).sum()),
        'fn': int(((preds == 0) & (lab_arr > 0)).sum()),
        'fp': int(((preds > 0) & (lab_arr == 0)).sum()),
        'tn': int(((preds == 0) & (lab_arr == 0)).sum()),
        'prec_at_80': prec_80,
        'n_at_80':    n_at_80,
        'loss': 0.5,
    }
    if best_epoch is not None:
        m['best_epoch'] = best_epoch
    if feature_importance is not None:
        m['feature_importance'] = np.array(feature_importance, dtype=np.float32)
    return m


def _good_metrics(best_epoch=10):
    """Metrics with high P@80 and no problems."""
    rng = np.random.default_rng(42)
    n = 500
    labels = (rng.random(n) < 0.15).astype(int)
    conf   = np.where(labels, rng.uniform(0.75, 0.95, n), rng.uniform(0.3, 0.65, n))
    return _make_metrics(conf, labels, best_epoch=best_epoch,
                         feature_importance=rng.uniform(0.1, 0.5, 8))


def test_health_monitor_no_warnings_on_healthy_run():
    """No warnings emitted when all signals are healthy."""
    monitor = FoldHealthMonitor()
    # Each fold gets a distinct seed so importance vectors differ (no WEIGHT_LOCK)
    for i, fold in enumerate(['F1', 'F2', 'F3']):
        rng = np.random.default_rng(100 + i)
        n = 500
        labels = (rng.random(n) < 0.15).astype(int)
        conf   = np.where(labels, rng.uniform(0.75, 0.95, n), rng.uniform(0.3, 0.65, n))
        importance = rng.uniform(0.1, 0.5, 8)
        monitor.check(fold, _make_metrics(conf, labels, best_epoch=10,
                                          feature_importance=importance))
    assert len(monitor.warnings) == 0


def test_health_monitor_early_epoch_detected():
    """EARLY_EPOCH fires when best_epoch <= threshold."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _good_metrics(best_epoch=3)
    warnings = monitor.check('F1', metrics)
    assert any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_no_early_epoch_above_threshold():
    """No EARLY_EPOCH when best_epoch is above threshold."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _good_metrics(best_epoch=6)
    warnings = monitor.check('F1', metrics)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_weight_lock_detected():
    """WEIGHT_LOCK fires when feature importance vectors are nearly identical."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=10,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=9,
                       feature_importance=importance * 1.0001)  # nearly identical
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    assert any(w.code == 'WEIGHT_LOCK' for w in warnings)


def test_health_monitor_weight_lock_message_includes_l2():
    """WEIGHT_LOCK message includes both cos_sim and L2 distance."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=10,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=9,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'cos_sim=' in wl.message
    assert 'L2=' in wl.message


def test_health_monitor_weight_lock_early_convergence_suggestion():
    """WEIGHT_LOCK suggestion mentions LR/freeze when best_epoch is low."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=5,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=8,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'LR' in wl.suggestion or 'FREEZE_RATIO' in wl.suggestion


def test_health_monitor_weight_lock_late_convergence_suggestion():
    """WEIGHT_LOCK suggestion mentions train_start when best_epoch is high."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'train_start' in wl.suggestion


def test_health_monitor_no_weight_lock_when_diverged():
    """No WEIGHT_LOCK when feature importance vectors differ substantially."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    imp1 = np.array([0.5, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
    imp2 = np.array([0.1, 0.5, 0.1, 0.2, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6], [1, 0], best_epoch=10, feature_importance=imp1)
    m2 = _make_metrics([0.85, 0.55], [1, 0], best_epoch=9, feature_importance=imp2)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    assert not any(w.code == 'WEIGHT_LOCK' for w in warnings)


def test_health_monitor_p80_decline_detected_after_window():
    """P80_DECLINE fires after p80_decline_window consecutive declines."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def metrics_with_p80(target_p80, seed):
        # Build metrics where P@80 ≈ target_p80 by mixing TPs and FPs at high conf.
        # P@80 = n_tp / (n_tp + n_fp)  →  n_fp = n_tp * (1 - target) / target
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx  = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        n_tp = max(1, len(sig_idx))
        conf[sig_idx] = 0.85  # all positives at high conf
        if target_p80 < 1.0 and target_p80 > 0:
            n_fp = int(n_tp * (1 - target_p80) / target_p80)
            n_fp = min(n_fp, len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85  # inject false positives to dilute P@80
        return _make_metrics(conf, labels, best_epoch=10)

    monitor.check('F1', metrics_with_p80(0.70, seed=10))
    monitor.check('F2', metrics_with_p80(0.55, seed=11))
    warnings = monitor.check('F3', metrics_with_p80(0.40, seed=12))
    assert any(w.code == 'P80_DECLINE' for w in warnings)


def test_health_monitor_no_p80_decline_after_one_dip():
    """No P80_DECLINE on a single fold decline — needs window=2 consecutive."""
    monitor = FoldHealthMonitor(p80_decline_window=2)
    rng = np.random.default_rng(1)

    def quick_metrics(conf_level, best_epoch=10):
        n = 200
        labels = (rng.random(n) < 0.20).astype(int)
        conf = np.where(labels == 1, conf_level, 0.35)
        return _make_metrics(conf, labels, best_epoch=best_epoch)

    monitor.check('F1', quick_metrics(0.85))
    monitor.check('F2', quick_metrics(0.75))  # one decline
    warnings = monitor.check('F3', quick_metrics(0.80))  # recovery
    assert not any(w.code == 'P80_DECLINE' for w in warnings)


def test_health_monitor_none_metrics_skipped():
    """None metrics must not crash the monitor."""
    monitor = FoldHealthMonitor()
    warnings = monitor.check('F1', None)
    assert warnings == []


def test_health_monitor_missing_best_epoch_skips_early_epoch_check():
    """No EARLY_EPOCH if best_epoch key is absent from metrics."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _make_metrics([0.9, 0.4], [1, 0])  # no best_epoch key
    warnings = monitor.check('F1', metrics)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_has_critical_reflects_severity():
    """has_critical() returns True when at least one critical warning exists."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def m(target_p80, seed):
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx   = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        n_tp = max(1, len(sig_idx))
        conf[sig_idx] = 0.85
        if 0 < target_p80 < 1.0:
            n_fp = int(n_tp * (1 - target_p80) / target_p80)
            n_fp = min(n_fp, len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85
        return _make_metrics(conf, labels, best_epoch=10)

    monitor.check('F1', m(0.70, seed=20))
    monitor.check('F2', m(0.55, seed=21))
    monitor.check('F3', m(0.38, seed=22))
    assert monitor.has_critical()


def test_health_monitor_summary_no_crash(capsys):
    """summary() must not crash even with no folds checked."""
    monitor = FoldHealthMonitor()
    monitor.summary()
    out = capsys.readouterr().out
    assert 'FOLD HEALTH SUMMARY' in out


# ── VAL_TEST_GAP ─────────────────────────────────────────────────────────────

def test_health_monitor_val_test_gap_detected():
    """VAL_TEST_GAP fires when val P@80 exceeds test P@80 by more than threshold."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    # Build test metrics with test P@80 ≈ 0.55 (50 TP + 41 FP at high conf)
    n_tp, n_fp = 50, 41
    conf = np.array([0.85] * n_tp + [0.85] * n_fp + [0.40] * (500 - n_tp - n_fp))
    labels = np.array([1] * n_tp + [0] * n_fp + [0] * (500 - n_tp - n_fp))
    m = _make_metrics(conf, labels, best_epoch=10)
    m['val_p80'] = 0.75  # gap = 0.75 - (50/91) ≈ 0.75 - 0.55 = 0.20 > 0.10
    warnings = monitor.check('F1', m)
    assert any(w.code == 'VAL_TEST_GAP' for w in warnings)


def test_health_monitor_no_val_test_gap_within_threshold():
    """No VAL_TEST_GAP when val and test P@80 are within the threshold."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    # test P@80 = 50/70 ≈ 0.71
    n_tp, n_fp = 50, 20
    conf = np.array([0.85] * n_tp + [0.85] * n_fp + [0.40] * (500 - n_tp - n_fp))
    labels = np.array([1] * n_tp + [0] * n_fp + [0] * (500 - n_tp - n_fp))
    m = _make_metrics(conf, labels, best_epoch=10)
    m['val_p80'] = 0.75  # gap ≈ 0.75 - 0.71 = 0.04 < 0.10
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings)


def test_health_monitor_val_test_gap_absent_when_no_val_p80():
    """No VAL_TEST_GAP check when val_p80 is missing (e.g. f1/loss checkpoint)."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    m = _good_metrics(best_epoch=10)  # no val_p80 key
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings)


# ── N_COLLAPSE ────────────────────────────────────────────────────────────────

def test_health_monitor_n_collapse_detected():
    """N_COLLAPSE fires when N above threshold drops more than ratio vs prev fold."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=5)
    # F1: 60 predictions above 0.80
    conf1   = np.array([0.85] * 60 + [0.40] * 440)
    labels1 = np.array([1]    * 60 + [0]    * 440)
    # F2: 20 predictions — 67% drop
    conf2   = np.array([0.85] * 20 + [0.40] * 480)
    labels2 = np.array([1]    * 20 + [0]    * 480)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert any(w.code == 'N_COLLAPSE' for w in warnings)


def test_health_monitor_no_n_collapse_within_ratio():
    """No N_COLLAPSE when N drop is within the allowed ratio."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=5)
    conf1   = np.array([0.85] * 60 + [0.40] * 440)
    labels1 = np.array([1]    * 60 + [0]    * 440)
    conf2   = np.array([0.85] * 40 + [0.40] * 460)  # 33% drop — within 50%
    labels2 = np.array([1]    * 40 + [0]    * 460)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert not any(w.code == 'N_COLLAPSE' for w in warnings)


def test_health_monitor_n_collapse_skips_when_prev_zero_signal():
    """N_COLLAPSE does not fire when the previous fold was already below min_signal_n."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=20)
    # F1: only 10 signals (below min) — not a valid comparison baseline
    conf1   = np.array([0.85] * 10 + [0.40] * 490)
    labels1 = np.array([1]    * 10 + [0]    * 490)
    # F2: 5 signals — even lower, but prev wasn't a viable baseline
    conf2   = np.array([0.85] * 5 + [0.40] * 495)
    labels2 = np.array([1]    * 5 + [0]    * 495)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert not any(w.code == 'N_COLLAPSE' for w in warnings)


# ── CONFIDENCE_FLAT ───────────────────────────────────────────────────────────

def test_health_monitor_confidence_flat_detected():
    """CONFIDENCE_FLAT fires when confidence std is below threshold."""
    monitor = FoldHealthMonitor(conf_flat_threshold=0.05)
    rng = np.random.default_rng(0)
    conf   = np.full(500, 0.50) + rng.uniform(-0.01, 0.01, 500)  # std ≈ 0.006
    labels = (np.arange(500) < 75).astype(int)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    assert any(w.code == 'CONFIDENCE_FLAT' for w in warnings)


def test_health_monitor_no_confidence_flat_with_spread():
    """No CONFIDENCE_FLAT when model shows good confidence spread."""
    monitor = FoldHealthMonitor(conf_flat_threshold=0.05)
    m = _good_metrics(best_epoch=10)  # positives in [0.75,0.95], negatives in [0.3,0.65]
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'CONFIDENCE_FLAT' for w in warnings)


# ── ZERO_SIGNAL_FOLD ──────────────────────────────────────────────────────────

def test_health_monitor_zero_signal_fold_detected():
    """ZERO_SIGNAL_FOLD fires (critical) when N above threshold is below minimum."""
    monitor = FoldHealthMonitor(min_signal_n=20)
    conf   = np.array([0.85] * 5 + [0.40] * 495)
    labels = np.array([1]    * 5 + [0]    * 495)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    zero_warns = [w for w in warnings if w.code == 'ZERO_SIGNAL_FOLD']
    assert len(zero_warns) > 0
    assert zero_warns[0].severity == 'critical'


def test_health_monitor_no_zero_signal_above_minimum():
    """No ZERO_SIGNAL_FOLD when N above threshold meets the minimum."""
    monitor = FoldHealthMonitor(min_signal_n=20)
    conf   = np.array([0.85] * 25 + [0.40] * 475)
    labels = np.array([1]    * 25 + [0]    * 475)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'ZERO_SIGNAL_FOLD' for w in warnings)


# ── _compute_p80 / VAL_TEST_GAP bug fix ───────────────────────────────────────

def test_health_monitor_compute_p80_uses_prec_at_80_field():
    """_compute_p80 should use prec_at_80 from metrics when present (fast path)."""
    # Scenario: all_conf has many values >= 0.80 from high-confidence no-signal bars
    # (mimics real trainer where confidence = max(softmax), not P(signal)).
    # Without the fix, _compute_p80 would compute 0.0; with fix it reads prec_at_80.
    conf   = np.array([0.90] * 200 + [0.30] * 300)  # 200 high-conf bars
    labels = np.array([0]    * 200 + [0]    * 300)   # all no-signal (labels=0)
    m = _make_metrics(conf, labels, best_epoch=10)
    # Override prec_at_80/n_at_80 as the trainer would compute them
    # (trainer applies (conf>=0.80)&(pred>0) mask — here preds=1 for conf>=0.5)
    # prec_at_80 from trainer would reflect signal precision, not all-bar precision
    m['prec_at_80'] = 0.466   # the correct test P@80 (trainer-computed)
    m['n_at_80']    = 146
    p80 = FoldHealthMonitor._compute_p80(m)
    assert abs(p80 - 0.466) < 1e-6, f'Expected 0.466, got {p80}'


def test_health_monitor_val_test_gap_no_false_alarm_from_high_conf_noise():
    """VAL_TEST_GAP must not fire when test P@80 (from prec_at_80) is close to val P@80.

    This is the F2 false-alarm bug: out['confidence'] stores max-softmax (not P(signal)),
    so many no-signal bars have conf>=0.80, making the raw _compute_p80 return 0.0.
    The fix: use prec_at_80 (pre-computed by trainer with (conf>=0.80)&(pred>0)) directly.
    """
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    conf   = np.array([0.90] * 300 + [0.30] * 200)
    labels = np.array([0]    * 300 + [0]    * 200)   # all no-signal in all_conf/all_labels
    m = _make_metrics(conf, labels, best_epoch=12)
    # Trainer-computed fields that reflect actual signal precision
    m['prec_at_80'] = 0.466   # real test P@80 at viable level
    m['n_at_80']    = 146
    m['val_p80']    = 0.481   # val P@80 — gap = 0.481 - 0.466 = 1.5% < 10%
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings), (
        'VAL_TEST_GAP fired as false alarm — _compute_p80 must use prec_at_80 field'
    )


def test_health_monitor_weight_lock_with_train_start_suppresses_train_start_suggestion():
    """WEIGHT_LOCK suggestion must NOT mention train_start when fold_config already has it."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    fold_config = {'name': 'F2', 'train_start': '2023-10-01', 'train_end': '2025-04-01'}
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2, fold_config=fold_config)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'Add train_start' not in wl.suggestion, (
        'Should not suggest adding train_start when it is already configured'
    )
    assert 'strategy_lr_multiplier' in wl.suggestion or 'FREEZE_RATIO' in wl.suggestion


def test_health_monitor_p80_decline_with_train_start_suggests_regime_drift():
    """P80_DECLINE suggestion must say 'regime drift' (not 'Add train_start') when already configured."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def metrics_with_p80(target_p80, seed):
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx  = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        conf[sig_idx] = 0.85
        if target_p80 < 1.0 and target_p80 > 0:
            n_fp = min(int(len(sig_idx) * (1 - target_p80) / target_p80), len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85
        return _make_metrics(conf, labels, best_epoch=12)

    fold_config = {'name': 'F3', 'train_start': '2023-10-01', 'train_end': '2025-04-01'}
    monitor.check('F1', metrics_with_p80(0.70, seed=10))
    monitor.check('F2', metrics_with_p80(0.55, seed=11))
    warnings = monitor.check('F3', metrics_with_p80(0.40, seed=12), fold_config=fold_config)
    pd80 = next(w for w in warnings if w.code == 'P80_DECLINE')
    assert 'regime drift' in pd80.suggestion, (
        'P80_DECLINE suggestion must mention regime drift when train_start already set'
    )
    assert 'Add train_start' not in pd80.suggestion


def test_health_monitor_weight_lock_without_fold_config_still_suggests_train_start():
    """WEIGHT_LOCK still suggests train_start when fold_config is None (backwards compat)."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)  # no fold_config
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'train_start' in wl.suggestion


def test_early_epoch_suppressed_when_training_continued():
    """EARLY_EPOCH must NOT fire when training ran 10+ epochs past best_epoch.

    In phase 2 with p80_patience=20, best_epoch=3 and epochs_trained=23 is
    normal: the model trained actively but P@80 peaked early due to gamma
    dynamics. This is not an anchor pathology.
    """
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    m = _good_metrics(best_epoch=3)
    m['epochs_trained'] = 23  # ran 20 epochs past best — not stalled
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings), (
        'EARLY_EPOCH must not fire when training ran well past best_epoch'
    )


def test_early_epoch_fires_when_training_stalled():
    """EARLY_EPOCH fires when best_epoch=3 and training barely continued (stalled)."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    m = _good_metrics(best_epoch=3)
    m['epochs_trained'] = 8  # only 5 epochs past best — truly stalled
    warnings = monitor.check('F1', m)
    assert any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_summarize_fold_precision_filters_noise_predictions():
    """summarize_fold_precision must count only signal predictions (pred>0) above threshold.

    Without the all_preds filter, high-confidence noise predictions (pred=0 with
    conf=0.95) inflate N and dilute the reported precision — same bug as the health
    monitor false alarm fixed in VAL_TEST_GAP.
    """
    # 10 noise bars predicted as noise with conf=0.95 (should NOT count at 0.80)
    # 5 signal bars predicted as signal with conf=0.85 (should count)
    # 2 signal bars predicted as noise with conf=0.90 (should NOT count)
    confs  = [0.95] * 10 + [0.85] * 5 + [0.90] * 2
    labels = [0]    * 10 + [1]    * 5 + [1]    * 2
    preds  = [0]    * 10 + [1]    * 5 + [0]    * 2  # first 10 and last 2 = no signal pred
    fold_results = {
        'F1': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
    }
    result = summarize_fold_precision(fold_results)
    # Only the 5 signal-predicted bars count; all 5 are correct → prec_at_80 = 1.0
    assert result['F1']['prec_at_80'] == pytest.approx(1.0, abs=0.001), (
        'summarize_fold_precision must exclude noise predictions (pred=0) from P@80'
    )


def test_print_fold_progression_filters_noise_predictions(capsys):
    """print_fold_progression P@80 must exclude high-conf noise predictions.

    Without the pred>0 filter, N@80 is nearly all test bars (conf of noise class
    is very high for most bars), making P@80 ≈ signal_rate ≈ 0.1%.
    """
    # 100 noise bars at conf=0.90 predicted as noise
    # 10 signal bars at conf=0.85 predicted as signal (prec should be 1.0)
    confs  = [0.90] * 100 + [0.85] * 10
    labels = [0]    * 100 + [1]    * 10
    preds  = [0]    * 100 + [1]    * 10
    fold_results = {
        'F1': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
        'F2': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
        'F3': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
    }
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    # P@80 should be 100% (10/10), not ~9% (10/110)
    assert '100.0%' in out, (
        'print_fold_progression must exclude pred=0 bars from P@80 calculation'
    )

# =============================================================================
# Borrow #1 — realized-R economic eval (pure aggregation, no training)
# =============================================================================

from futures_foundation.finetune.trainer import _realized_r_eval


def _ramp(n=30):
    o = np.full(n, 100.0); h = np.full(n, 100.1)
    l = np.full(n, 99.9);  c = np.full(n, 100.0)
    return o, h, l, c


def test_realized_r_eval_empty_is_zeroed():
    o, h, l, c = _ramp()
    r = _realized_r_eval(o, h, l, c, np.full(30, 1.0), [], [], [])
    assert r == {'n': 0, 'mean_r': 0.0, 'win_rate': 0.0,
                 'profit_factor': 0.0, 'max_dd': 0.0, 'no_top1': 0.0}


def test_realized_r_eval_long_winner():
    o, h, l, c = _ramp(30)
    atr = np.full(30, 1.0)
    # signal idx 5 -> entry o[6]=100, risk=1 (sl_dist). Ramp up then drop so
    # the trailing exit locks a positive R.
    for j in range(6, 12):
        h[j] = 100.0 + (j - 5) * 2.0
        l[j] = 99.5 + (j - 5) * 2.0
        c[j] = 99.8 + (j - 5) * 2.0
    h[12], l[12], c[12] = 110.0, 95.0, 96.0          # reversal -> trail hit
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [1.0],
                         trail_atr_k=2.0, activate_r=1.0, max_hold=50)
    assert r['n'] == 1
    assert r['mean_r'] > 0 and r['win_rate'] == 1.0


def test_realized_r_eval_long_stopped_is_negative():
    o, h, l, c = _ramp(20)
    atr = np.full(20, 1.0)
    l[7] = 98.0                                       # entry o[6]=100, sl=99
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [1.0])
    assert r['n'] == 1
    assert r['mean_r'] == pytest.approx(-1.0, abs=1e-6)
    assert r['win_rate'] == 0.0 and r['max_dd'] <= 0.0


def test_realized_r_eval_atr_fallback_when_sl_missing():
    o, h, l, c = _ramp(20)
    atr = np.full(20, 1.0)
    l[7] = 98.0
    # sl_dist NaN -> risk falls back to atr (=1.0): same -1R outcome
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [float('nan')])
    assert r['n'] == 1 and r['mean_r'] == pytest.approx(-1.0, abs=1e-6)


def test_realized_r_eval_short_winner():
    o, h, l, c = _ramp(30)
    atr = np.full(30, 1.0)
    for j in range(6, 12):
        l[j] = 100.0 - (j - 5) * 2.0
        h[j] = 100.5 - (j - 5) * 2.0
        c[j] = 100.2 - (j - 5) * 2.0
    h[12], l[12], c[12] = 105.0, 90.0, 104.0          # reversal up -> trail
    r = _realized_r_eval(o, h, l, c, atr, [5], [False], [1.0],
                         trail_atr_k=2.0, activate_r=1.0, max_hold=50)
    assert r['n'] == 1 and r['mean_r'] > 0


def test_realized_r_eval_no_top1_le_mean_and_dd_nonpos():
    # mix: several -1R + one big winner -> no_top1 (tail removed) < mean
    o = np.full(60, 100.0); h = np.full(60, 100.1)
    l = np.full(60, 99.9);  c = np.full(60, 100.0); atr = np.full(60, 1.0)
    sig, isl, sld = [], [], []
    for s in range(2, 50, 5):
        sig.append(s); isl.append(True); sld.append(1.0)
        l[s + 1] = 98.0                                # each -> ~ -1R
    # turn the last one into a monster winner
    s = sig[-1]
    l[s + 1] = 99.95
    for j in range(s + 1, s + 8):
        h[j] = 100.0 + (j - s) * 5.0; l[j] = 99.9 + (j - s) * 5.0
        c[j] = 100.0 + (j - s) * 5.0
    h[s + 8], l[s + 8], c[s + 8] = 140.0, 95.0, 96.0
    r = _realized_r_eval(o, h, l, c, atr, sig, isl, sld, max_hold=60)
    assert r['n'] == len(sig)
    assert r['no_top1'] <= r['mean_r'] + 1e-9
    assert r['max_dd'] <= 0.0


# =============================================================================
# Borrow #1 (b2) — realized_r threaded labeler→labels→dataset→eval (back-compat)
# =============================================================================

from futures_foundation.finetune.trainer import _print_realized_econ


class _DirectionLabeler(TrivialLabeler):
    """A few fixed long signals (borrow #4 ABC): base run() emits
    `direction` for free → run_labeling computes `realized_r`."""

    @property
    def name(self):
        return 'direction'

    def detect_events(self, df_raw, ffm_df, ticker):
        return pd.DataFrame({
            'bar_idx':     [50, 100, 150],
            'direction':   1,                    # long
            'sl_distance': 2.0,
            'tp_rr':       1.5,
        })


def _labeling_dirs(tmp_path, labeler, ticker='DIR', n=300, trend=0.0):
    raw_dir = tmp_path / 'raw'; ffm_dir = tmp_path / 'ffm'
    cache_dir = tmp_path / 'cache'
    raw_dir.mkdir(); ffm_dir.mkdir()
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=True)
    base = 5000.0 + np.arange(n) * trend
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open':  base,
        'high':  base + 1.0,
        'low':   base - 1.0,
        'close': base,
        'volume': np.full(n, 500.0),
    })
    raw_data.to_csv(raw_dir / f'{ticker}_5min.csv', index=False)
    run_labeling(labeler, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir))
    return cache_dir, ticker


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_adds_realized_r_when_direction(tmp_path):
    """labeler emits `direction` → cached labels gain a `realized_r` column;
    non-signal rows are NaN, signal rows are computed (uptrend → finite)."""
    cache_dir, ticker = _labeling_dirs(tmp_path, _DirectionLabeler(), trend=2.0)
    lab = pd.read_parquet(cache_dir / f'{ticker}_strategy_labels.parquet')
    assert 'realized_r' in lab.columns
    sig_mask = lab['signal_label'].values > 0
    assert np.isnan(lab['realized_r'].values[~sig_mask]).all()
    assert np.isfinite(lab['realized_r'].values[sig_mask]).any()


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_always_emits_direction_and_realized_r(tmp_path):
    """Borrow #4: every conformant labeler emits `direction` via the base
    run() (no opt-out) ⇒ `realized_r` is computed for free. The absent-
    direction back-compat path still exists defensively and is covered at
    the dataset level by test_dataset_emits_realized_r_key_backcompat."""
    cache_dir, ticker = _labeling_dirs(tmp_path, TrivialLabeler(),
                                       ticker='ND', trend=2.0)
    lab = pd.read_parquet(cache_dir / f'{ticker}_strategy_labels.parquet')
    assert 'direction' in lab.columns
    assert 'realized_r' in lab.columns


def test_print_realized_econ_prints_block(capsys):
    """Finite realized R at predicted-positive rows → econ block with PF/WR."""
    n = 200
    conf = np.full(n, 0.40); pred = np.zeros(n, int)
    realized = np.full(n, np.nan)
    idx = np.arange(10, 40)
    conf[idx] = 0.85; pred[idx] = 1
    realized[idx] = np.where(np.arange(30) % 3 == 0, -1.0, 2.0)
    _print_realized_econ(conf, pred, realized.tolist(), 'COMBINED')
    out = capsys.readouterr().out
    assert 'REALIZED-R ECONOMICS' in out
    assert 'NOT MFE' in out
    assert '0.80' in out


def test_print_realized_econ_skips_when_all_nan(capsys):
    """All-NaN realized (no `direction` / pre-borrow cache) → skip w/ notice."""
    n = 50
    conf = np.full(n, 0.85); pred = np.ones(n, int)
    _print_realized_econ(conf, pred, [float('nan')] * n, 'COMBINED')
    out = capsys.readouterr().out
    assert 'skipped' in out and 'relabel with force=True' in out


def test_print_realized_econ_noop_when_absent(capsys):
    """None / empty realized arg → no output at all (full back-compat)."""
    _print_realized_econ(np.array([0.9]), np.array([1]), None, 'F1')
    _print_realized_econ(np.array([0.9]), np.array([1]), [], 'F1')
    assert capsys.readouterr().out == ''


def test_threshold_table_realized_econ_backcompat(capsys):
    """_print_test_threshold_table with no all_realized_r key → no crash,
    no econ block (legacy metrics dicts stay valid)."""
    metrics = _make_fake_test_metrics(n=200)
    metrics.pop('all_realized_r', None)
    _print_test_threshold_table(metrics, 'F1')
    out = capsys.readouterr().out
    assert 'REALIZED-R ECONOMICS' not in out


# =============================================================================
# Borrow #3 — economic product objective for checkpoint selection (opt-in)
# =============================================================================

from futures_foundation.finetune.trainer import (
    _econ_combined_objective, _val_econ_objective,
)


def test_econ_objective_empty_and_below_min_is_zero():
    assert _econ_combined_objective([]) == 0.0
    assert _econ_combined_objective([3, 3, 3, 3]) == 0.0          # < min_trades


def test_econ_objective_nonpositive_edge_is_zero():
    assert _econ_combined_objective([-1, -1, -2, -1, -1, -1]) == 0.0
    assert _econ_combined_objective([1, -1, 1, -1, 1, -1, -1, -1]) == 0.0


def test_econ_objective_is_bounded_no_blowup_when_no_losses():
    """All-winners (downside→0) must NOT explode on the epsilon term."""
    v = _econ_combined_objective([2.0] * 20)
    assert 0.0 < v < 10.0                                          # capped


def test_econ_objective_selective_beats_unselective():
    selective   = _econ_combined_objective([2, 2, 1, 3, 2, 2, 1, 2, 3, 2])
    unselective = _econ_combined_objective([2, -2, 3, -3, 1, -1, 4, -4, 2, -1])
    assert selective > unselective


def test_econ_objective_rewards_more_signals_same_distribution():
    """The √n_factor term: more trades at the same R distribution scores
    higher (the 'maybe the one with most signals' property)."""
    few  = _econ_combined_objective([2, -1] * 5)
    many = _econ_combined_objective([2, -1] * 40)
    assert many > few


def test_econ_objective_lower_downside_scores_higher():
    smooth = _econ_combined_objective([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    jagged = _econ_combined_objective([1, -2, 4, -2, 3, -1, 2, -1, 3, 1])
    assert smooth > jagged


def test_val_econ_objective_nan_when_no_realized_r_backcompat():
    """Back-compat: no/empty/all-NaN realized_r → NaN so the caller falls
    back to the default _p80s priority (labeler emits no `direction`)."""
    assert np.isnan(_val_econ_objective(
        {'all_realized_r': [], 'all_conf': [1.0], 'all_preds': [1]}))
    assert np.isnan(_val_econ_objective(
        {'all_realized_r': [float('nan')] * 4,
         'all_conf': [0.9] * 4, 'all_preds': [1] * 4}))
    assert np.isnan(_val_econ_objective(
        {'all_conf': [0.9], 'all_preds': [1]}))                    # key absent


def test_val_econ_objective_filters_predicted_positive_above_thresh():
    n = 40
    conf = np.full(n, 0.40); pred = np.zeros(n, int)
    rr   = np.full(n, np.nan)
    idx  = np.arange(5, 25)
    conf[idx] = 0.90; pred[idx] = 1
    rr[idx]   = 2.0
    v = _val_econ_objective(
        {'all_realized_r': rr.tolist(), 'all_conf': conf.tolist(),
         'all_preds': pred.tolist()}, thresh=0.80)
    assert v > 0.0
    # nothing predicted-positive above threshold → 0.0 (not NaN: data exists)
    v0 = _val_econ_objective(
        {'all_realized_r': rr.tolist(), 'all_conf': np.full(n, 0.10).tolist(),
         'all_preds': np.zeros(n, int).tolist()}, thresh=0.80)
    assert v0 == 0.0


def test_econ_config_defaults_off():
    """econ_selection defaults OFF; econ_patience default is 10."""
    base = TrainingConfig()
    assert base.econ_selection is False and base.econ_patience == 10


# =============================================================================
# Borrow #4-locked — clean ABC: detect_events()+compute_features(), final run()
# =============================================================================

class _ShortLabeler(TrivialLabeler):
    """One short event — exercises the short barrier path."""
    @property
    def name(self):
        return 'short'

    def detect_events(self, df_raw, ffm_df, ticker):
        return pd.DataFrame({'bar_idx': [30], 'direction': [-1],
                             'sl_distance': [2.0], 'tp_rr': [1.5]})


class _BadTpLabeler(TrivialLabeler):
    """tp_rr < 1.0 — base must clamp to 1.0 (TP>=SL) and warn."""
    def detect_events(self, df_raw, ffm_df, ticker):
        return pd.DataFrame({'bar_idx': [40], 'direction': [1],
                             'sl_distance': [1.0], 'tp_rr': [0.5]})


class _MissingColLabeler(TrivialLabeler):
    def detect_events(self, df_raw, ffm_df, ticker):
        return pd.DataFrame({'bar_idx': [40], 'direction': [1]})  # no sl/tp


class _NoHookLabeler(StrategyLabeler):
    name = 'nohook'
    feature_cols = STRATEGY_COLS                 # detect_events/compute_features absent


def _dt_ffm(n=120):
    f = make_ffm_df(n)
    f.index = pd.to_datetime(f['_datetime'])
    return f


def test_abc_requires_detect_events_and_compute_features():
    with pytest.raises(TypeError):
        _NoHookLabeler()                          # abstract hooks unimplemented


def test_base_run_triple_barrier_long_win_sets_label_and_direction():
    ffm = _dt_ffm(120)
    feats, labels = TrivialLabeler().run(make_raw(120, trend=2.0), ffm, 'T')
    assert {'signal_label', 'max_rr', 'sl_distance', 'direction'} <= set(labels.columns)
    sig = labels['signal_label'].values > 0
    assert sig.sum() > 0                          # uptrend longs win
    assert (labels['direction'].values[sig] == 1).all()
    assert (labels['sl_distance'].values[sig] == 1.0).all()
    assert (labels['max_rr'].values[sig] > 0).all()   # realized R on wins


def test_base_run_entry_is_next_bar_open():
    """Signal at bar i, entry at i+1. A long with a hard drop AT the signal
    bar but rising AFTER must still win (entry uses i+1, not i)."""
    ffm = _dt_ffm(60)
    raw = make_raw(60, trend=2.0)
    raw.iloc[25, raw.columns.get_loc('low')] = 1.0     # crash on signal bar 25 only

    class _L(TrivialLabeler):
        def detect_events(self, d, f, t):
            return pd.DataFrame({'bar_idx': [25], 'direction': [1],
                                 'sl_distance': [1.0], 'tp_rr': [1.0]})

    _, labels = _L().run(raw, ffm, 'T')
    assert labels['signal_label'].iloc[25] == 1        # entry at 26 → unaffected


def test_base_run_short_path():
    ffm = _dt_ffm(80)
    _, labels = _ShortLabeler().run(make_raw(80, trend=-2.0), ffm, 'T')
    assert labels['signal_label'].iloc[30] == 1        # downtrend short wins
    assert labels['direction'].iloc[30] == -1


def test_base_run_clamps_tp_rr_below_one_and_warns():
    ffm = _dt_ffm(80)
    with pytest.warns(UserWarning, match='tp_rr < 1.0 clamped'):
        _, labels = _BadTpLabeler().run(make_raw(80, trend=2.0), ffm, 'T')
    assert labels['signal_label'].iloc[40] in (0, 1)   # resolved, no crash


def test_base_run_missing_event_columns_raises():
    ffm = _dt_ffm(80)
    with pytest.raises(ValueError, match='missing columns'):
        _MissingColLabeler().run(make_raw(80), ffm, 'T')


def test_base_run_features_from_compute_features():
    ffm = _dt_ffm(100)
    feats, _ = TrivialLabeler().run(make_raw(100), ffm, 'T')
    assert list(feats.columns) == STRATEGY_COLS and len(feats) == 100
