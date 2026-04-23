"""Unit tests for labels.py — no torch dependency."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

_LABELS_PATH = Path(__file__).parent.parent / "futures_foundation" / "labels.py"
_spec = importlib.util.spec_from_file_location("ffm_labels", _LABELS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

generate_regime_labels = _mod.generate_regime_labels
generate_volatility_labels = _mod.generate_volatility_labels
generate_structure_labels = _mod.generate_structure_labels
generate_range_labels = _mod.generate_range_labels
generate_all_labels = _mod.generate_all_labels
REGIME_LABELS = _mod.REGIME_LABELS
VOLATILITY_LABELS = _mod.VOLATILITY_LABELS
STRUCTURE_LABELS = _mod.STRUCTURE_LABELS
RANGE_LABELS = _mod.RANGE_LABELS
LABEL_CONFIDENCE_SENTINEL = _mod.LABEL_CONFIDENCE_SENTINEL


def make_features(close_array):
    return pd.DataFrame({"_close": np.asarray(close_array, dtype=float)})


def make_trend(n=400, start=100.0, end=400.0):
    return make_features(np.linspace(start, end, n))


def make_downtrend(n=400, start=400.0, end=100.0):
    return make_features(np.linspace(start, end, n))


# =============================================================================
# Label map completeness
# =============================================================================

def test_label_maps_complete():
    assert set(REGIME_LABELS) == {0, 1, 2, 3}
    assert set(VOLATILITY_LABELS) == {0, 1, 2, 3}
    assert set(STRUCTURE_LABELS) == {0, 1, 2}
    assert set(RANGE_LABELS) == {0, 1, 2, 3, 4}


# =============================================================================
# generate_all_labels — schema
# =============================================================================

def test_all_labels_columns():
    labels = generate_all_labels(make_trend())
    assert set(labels.columns) == {"regime_label", "volatility_label", "structure_label", "range_label"}


def test_all_labels_index_matches():
    features = make_trend(n=300)
    labels = generate_all_labels(features)
    assert len(labels) == len(features)
    assert labels.index.equals(features.index)


# =============================================================================
# generate_regime_labels
# =============================================================================

def test_regime_dtype_and_range():
    labels = generate_regime_labels(make_trend())
    valid = labels.dropna()
    assert str(valid.dtype) == "Int64"
    assert set(valid.unique()).issubset({LABEL_CONFIDENCE_SENTINEL, 0, 1, 2, 3})


def test_regime_trailing_nans():
    labels = generate_regime_labels(make_trend(n=300), horizon=20)
    assert labels.iloc[-20:].isna().all()


def test_regime_uptrend_dominant():
    """Strong monotone uptrend → trending_up (label 0) dominates."""
    labels = generate_regime_labels(make_trend(n=400, start=100, end=800))
    valid = labels.dropna()
    assert (valid == 0).sum() > len(valid) * 0.4


def test_regime_downtrend_dominant():
    """Strong monotone downtrend → trending_down (label 1) dominates."""
    # Exponential decay gives constant pct_change → ret_std → 0 → ret_threshold → 0
    # So even tiny negative fwd_return exceeds the threshold → label 1
    close = 400 * np.exp(-np.linspace(0, 3, 400))
    labels = generate_regime_labels(make_features(close))
    valid = labels.dropna()
    assert (valid == 1).sum() > len(valid) * 0.4


def test_regime_flat_rotational():
    """Near-flat price → low fwd returns → mostly rotational (label 2)."""
    np.random.seed(0)
    close = 100.0 + np.random.randn(400) * 0.001  # tiny noise
    labels = generate_regime_labels(make_features(close))
    valid = labels.dropna()
    assert (valid == 2).sum() > len(valid) * 0.3


def test_regime_output_length():
    features = make_trend(n=250)
    labels = generate_regime_labels(features)
    assert len(labels) == 250


# =============================================================================
# generate_volatility_labels
# =============================================================================

def test_volatility_dtype_and_range():
    labels = generate_volatility_labels(make_trend())
    valid = labels.dropna()
    assert str(valid.dtype) == "Int64"
    assert set(valid.unique()).issubset({0, 1, 2, 3})


def test_volatility_trailing_nans():
    labels = generate_volatility_labels(make_trend(n=300), horizon=10)
    assert labels.iloc[-10:].isna().all()


def test_volatility_quiet_period_lower_than_noisy():
    """Low-vol period should accumulate fewer elevated/extreme labels than high-vol period."""
    np.random.seed(1)
    close = np.concatenate([
        100 + np.cumsum(np.random.randn(300) * 4),   # high-vol section
        500 + np.cumsum(np.random.randn(300) * 0.1), # low-vol section
    ])
    labels = generate_volatility_labels(make_features(close), horizon=10)
    high_sec = labels.iloc[100:280].dropna()
    low_sec = labels.iloc[310:580].dropna()
    high_frac = (high_sec >= 2).sum() / len(high_sec)
    low_frac = (low_sec >= 2).sum() / len(low_sec)
    assert low_frac < high_frac, \
        f"Low-vol section ({low_frac:.2f}) should have fewer elevated labels than high-vol ({high_frac:.2f})"


def test_volatility_output_length():
    features = make_trend(n=250)
    labels = generate_volatility_labels(features)
    assert len(labels) == 250


# =============================================================================
# generate_structure_labels
# =============================================================================

def test_structure_dtype_and_range():
    labels = generate_structure_labels(make_trend())
    valid = labels.dropna()
    assert str(valid.dtype) == "Int64"
    assert set(valid.unique()).issubset({LABEL_CONFIDENCE_SENTINEL, 0, 1, 2})


def test_structure_trailing_nans():
    labels = generate_structure_labels(make_trend(n=300), horizon=20)
    assert labels.iloc[-20:].isna().all()


def test_structure_bullish_when_upside_dominates():
    """
    Bullish label (0) appears when upside >> downside over the forward window.
    Requires downside > 0 (future dips below current) AND upside/downside > 1.5.
    Construct: small dip to 90, then large rally to 300, preceding bars have close=100.
    """
    n = 150
    close = np.full(n, 100.0)
    close[60:70] = np.linspace(100, 90, 10)    # dip: establishes downside
    close[70:110] = np.linspace(90, 300, 40)   # strong rally: establishes upside
    close[110:] = 300.0
    labels = generate_structure_labels(make_features(close), horizon=30)
    # Bars 30-55: horizon window covers bars 31-60+, includes dip and rally
    test_window = labels.iloc[30:58].dropna()
    assert (test_window == 0).any(), "Bullish label should appear when upside >> downside"


def test_structure_confident_labels_appear():
    """Normal volatile data should produce both bullish and bearish structure labels."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(500) * 2)
    labels = generate_structure_labels(make_features(close))
    confident = labels.dropna()
    confident = confident[confident != LABEL_CONFIDENCE_SENTINEL]
    assert {0, 1}.issubset(set(confident.unique())), \
        f"Expected both bullish (0) and bearish (1), got {sorted(confident.unique())}"


def test_structure_downtrend_bearish():
    """Strong downtrend: downside >> upside from current → bearish (label 1)."""
    labels = generate_structure_labels(make_downtrend(n=400, start=800, end=100))
    valid = labels.dropna()
    assert (valid == 1).sum() > len(valid) * 0.5


def test_structure_output_length():
    features = make_trend(n=250)
    labels = generate_structure_labels(features)
    assert len(labels) == 250


# =============================================================================
# generate_range_labels
# =============================================================================

def test_range_dtype_and_range():
    labels = generate_range_labels(make_trend())
    valid = labels.dropna()
    assert str(valid.dtype) == "Int64"
    assert set(valid.unique()).issubset({0, 1, 2, 3, 4})


def test_range_trailing_nans():
    labels = generate_range_labels(make_trend(n=300), horizon=10)
    assert labels.iloc[-10:].isna().all()


def test_range_far_above_gets_label4():
    """Future close far above recent range (clips to 1.0) → label 4."""
    close = np.concatenate([np.linspace(100, 110, 70), np.full(50, 5000.0)])
    labels = generate_range_labels(make_features(close), horizon=5, lookback=20)
    test_bars = labels.iloc[55:65].dropna()
    assert len(test_bars) > 0
    assert (test_bars == 4).all(), f"Expected label 4, got {test_bars.values}"


def test_range_far_below_gets_label0():
    """Future close far below recent range (clips to 0.0) → label 0."""
    # Crash starts at index 70; with horizon=5, bars 65-74 look forward into the crash
    close = np.concatenate([np.linspace(100, 110, 70), np.full(50, 0.001)])
    labels = generate_range_labels(make_features(close), horizon=5, lookback=20)
    test_bars = labels.iloc[65:74].dropna()
    assert len(test_bars) > 0
    assert (test_bars == 0).all(), f"Expected label 0, got {test_bars.values}"


def test_range_all_quintiles_appear():
    """Oscillating price should cover all 5 quintiles."""
    t = np.linspace(0, 8 * np.pi, 500)
    close = 100 + 50 * np.sin(t)
    labels = generate_range_labels(make_features(close), horizon=5, lookback=30)
    valid = labels.dropna()
    assert set(valid.unique()) == {0, 1, 2, 3, 4}, \
        f"Expected all 5 quintiles, got {sorted(valid.unique())}"


def test_range_output_length():
    features = make_trend(n=250)
    labels = generate_range_labels(features)
    assert len(labels) == 250
