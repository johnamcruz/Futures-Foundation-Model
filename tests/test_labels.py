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


def make_features(close_array, struct_1h=None):
    df = pd.DataFrame({"_close": np.asarray(close_array, dtype=float)})
    n = len(df)
    # _1h_structure is always required by generate_structure_labels.
    # Default to 0 (choppy/mixed) so generate_all_labels never raises KeyError.
    if struct_1h is not None:
        val = struct_1h if not np.isscalar(struct_1h) else [struct_1h] * n
    else:
        val = [0] * n
    df["_1h_structure"] = pd.array(val, dtype="Int64")
    return df


def make_trend(n=400, start=100.0, end=400.0):
    return make_features(np.linspace(start, end, n), struct_1h=1)


def make_downtrend(n=400, start=400.0, end=100.0):
    return make_features(np.linspace(start, end, n), struct_1h=-1)


# =============================================================================
# Label map completeness
# =============================================================================

def test_label_maps_complete():
    assert set(REGIME_LABELS) == {0, 1, 2, 3}
    assert set(VOLATILITY_LABELS) == {0, 1, 2, 3}
    assert set(STRUCTURE_LABELS) == {0, 1}   # no mixed class — neutral → SENTINEL
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


def test_regime_sentinel_fires_in_mixed_market():
    """Noisy random-walk market has borderline bars — sentinel fraction must be > 15%."""
    np.random.seed(99)
    close = 100 + np.cumsum(np.random.randn(600) * 1.5)
    labels = generate_regime_labels(make_features(close))
    non_nan = labels.dropna()
    sentinel_frac = (non_nan == LABEL_CONFIDENCE_SENTINEL).sum() / len(non_nan)
    assert sentinel_frac > 0.15, (
        f"Expected >15% ambiguous sentinel bars in mixed market, got {sentinel_frac:.1%}. "
        "Confidence masking may not be firing."
    )


def test_regime_no_sentinel_in_monotone_trend():
    """Strong monotone uptrend: every non-tail bar has a clear signal — sentinel fraction = 0%."""
    labels = generate_regime_labels(make_trend(n=400, start=100, end=800))
    non_nan = labels.dropna()
    sentinel_frac = (non_nan == LABEL_CONFIDENCE_SENTINEL).sum() / len(non_nan)
    assert sentinel_frac < 0.05, (
        f"Expected <5% sentinel in monotone uptrend, got {sentinel_frac:.1%}. "
        "Strong trends should be unambiguously labeled."
    )


def test_regime_volatile_market_fires_class_3():
    """High-vol regime: volatile-expansion label (3) appears — not masked as ambiguous."""
    np.random.seed(11)
    close = 100 + np.cumsum(np.random.randn(600) * 8)  # large per-bar moves
    labels = generate_regime_labels(make_features(close))
    non_nan = labels.dropna()
    assert (non_nan == 3).sum() > 0, (
        "High-volatility series must produce volatile-expansion label (3). "
        "Confidence masking must not suppress class-3 from clear volatility spikes."
    )


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
    n = 400
    vals = [1] * 100 + [-1] * 100 + [0] * 100 + [1] * 100
    features = make_features(np.linspace(100, 400, n), struct_1h=vals)
    labels = generate_structure_labels(features)
    valid = labels.dropna()
    assert str(valid.dtype) == "Int64"
    assert set(valid.unique()).issubset({LABEL_CONFIDENCE_SENTINEL, 0, 1})


def test_structure_trailing_nans():
    features = make_features(np.linspace(100, 400, 300), struct_1h=1)
    labels = generate_structure_labels(features, horizon=48)
    assert labels.iloc[-48:].isna().all()


def test_structure_bullish_when_1h_bullish():
    """When _1h_structure is +1 throughout, all valid labels are bullish (0)."""
    features = make_features(np.linspace(100, 400, 200), struct_1h=1)
    labels = generate_structure_labels(features, horizon=48)
    valid = labels.dropna()
    labeled = valid[valid != LABEL_CONFIDENCE_SENTINEL]
    assert len(labeled) > 0
    assert (labeled == 0).all(), \
        "All non-sentinel labels should be bullish (0) when 1H structure is +1"


def test_structure_bearish_when_1h_bearish():
    """When _1h_structure is -1 throughout, all valid labels are bearish (1)."""
    features = make_features(np.linspace(400, 100, 200), struct_1h=-1)
    labels = generate_structure_labels(features, horizon=48)
    valid = labels.dropna()
    labeled = valid[valid != LABEL_CONFIDENCE_SENTINEL]
    assert len(labeled) > 0
    assert (labeled == 1).all(), \
        "All non-sentinel labels should be bearish (1) when 1H structure is -1"


def test_structure_sentinel_when_choppy():
    """When _1h_structure is 0 (choppy), all valid labels are SENTINEL."""
    features = make_features(np.linspace(100, 200, 200), struct_1h=0)
    labels = generate_structure_labels(features, horizon=48)
    non_nan = labels.dropna()
    assert (non_nan == LABEL_CONFIDENCE_SENTINEL).all(), \
        "All valid labels should be SENTINEL (-100) when 1H structure is 0 (choppy)"


def test_structure_confident_labels_appear():
    """Mixed +1/-1 1H structure produces both bullish and bearish labels."""
    n = 400
    vals = [1] * 150 + [-1] * 100 + [1] * 150
    features = make_features(np.linspace(100, 400, n), struct_1h=vals)
    labels = generate_structure_labels(features, horizon=48)
    valid = labels.dropna()
    confident = valid[valid != LABEL_CONFIDENCE_SENTINEL]
    assert {0, 1}.issubset(set(confident.unique())), \
        f"Expected both bullish (0) and bearish (1), got {sorted(confident.unique())}"


def test_structure_output_length():
    features = make_features(np.linspace(100, 400, 250), struct_1h=1)
    labels = generate_structure_labels(features)
    assert len(labels) == 250


def test_structure_sentinel_fires_when_choppy_values_present():
    """Mixed 1H structure including 0 values produces SENTINEL for those bars."""
    np.random.seed(42)
    n = 600
    vals = ([1, -1, 0] * (n // 3 + 1))[:n]
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    features = make_features(close, struct_1h=vals)
    labels = generate_structure_labels(features)
    non_nan = labels.dropna()
    sentinel_frac = (non_nan == LABEL_CONFIDENCE_SENTINEL).sum() / len(non_nan)
    assert sentinel_frac > 0.10, (
        f"Expected >10% SENTINEL when 1H structure includes 0 (choppy) bars, "
        f"got {sentinel_frac:.1%}."
    )


def test_structure_sentinel_low_in_strong_downtrend():
    """All-bearish 1H structure → mostly bearish labels, low SENTINEL."""
    features = make_features(np.linspace(800, 100, 400), struct_1h=-1)
    labels = generate_structure_labels(features)
    non_nan = labels.dropna()
    sentinel_frac = (non_nan == LABEL_CONFIDENCE_SENTINEL).sum() / len(non_nan)
    bearish_frac = (non_nan == 1).sum() / len(non_nan)
    assert sentinel_frac < 0.30, \
        f"Expected <30% sentinel with uniform bearish 1H structure, got {sentinel_frac:.1%}."
    assert bearish_frac > 0.50, \
        f"Expected >50% bearish labels with bearish 1H structure, got {bearish_frac:.1%}."


def test_structure_no_class_2_produced():
    """Class 2 (old 'mixed' class) must never appear — ambiguous bars get SENTINEL."""
    np.random.seed(42)
    n = 500
    vals = ([1] * 150 + [-1] * 150 + [0] * 200)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    features = make_features(close, struct_1h=vals)
    labels = generate_structure_labels(features)
    non_nan = labels.dropna()
    assert 2 not in set(non_nan.unique()), (
        "Structure label class 2 must not appear. "
        "Ambiguous/choppy bars should be LABEL_CONFIDENCE_SENTINEL, not class 2."
    )


# =============================================================================
# Market-scenario coverage — all four label types across all regime types
# =============================================================================


def test_all_four_label_types_in_trending_market():
    """Trending market: all four label generators run without error and have valid output."""
    labels = generate_all_labels(make_trend(n=400, start=100, end=800))
    valid = labels.dropna()
    assert len(valid) > 0
    assert "regime_label" in labels.columns
    assert "volatility_label" in labels.columns
    assert "structure_label" in labels.columns
    assert "range_label" in labels.columns


def test_all_four_label_types_in_downtrend():
    """Downtrend: all four label generators run without error and produce valid output."""
    labels = generate_all_labels(make_downtrend(n=400, start=800, end=100))
    valid = labels.dropna()
    assert len(valid) > 0
    assert (valid["regime_label"].isin([LABEL_CONFIDENCE_SENTINEL, 0, 1, 2, 3])).all()
    assert (valid["structure_label"].isin([LABEL_CONFIDENCE_SENTINEL, 0, 1])).all()


def test_all_four_label_types_in_volatile_market():
    """High-vol choppy market: all generators run; volatile-expansion labels appear for regime."""
    np.random.seed(11)
    close = 100 + np.cumsum(np.random.randn(600) * 8)
    feat = make_features(close)
    labels = generate_all_labels(feat)
    non_nan = labels.dropna()
    assert len(non_nan) > 0
    assert (non_nan["regime_label"] == 3).any(), "Volatile-expansion (3) must appear in high-vol regime"
    assert (non_nan["volatility_label"] >= 2).any(), "Elevated/extreme vol must appear in high-vol data"


def test_all_four_label_types_in_sideways_market():
    """Sideways choppy market: all generators run and return plausible label distributions."""
    np.random.seed(21)
    close = 100 + np.random.randn(500) * 0.5  # bounded noise, no trend
    feat = make_features(close)
    labels = generate_all_labels(feat)
    non_nan = labels.dropna()
    assert len(non_nan) > 0
    # Rotational regime should be substantial in a flat market
    regime_non_nan = non_nan["regime_label"]
    assert (regime_non_nan == 2).sum() > len(regime_non_nan) * 0.20, (
        "Rotational label (2) should dominate a flat sideways market"
    )


def test_regime_sentinel_does_not_replace_volatile_class():
    """High-vol data: volatile bars are labeled 3, NOT masked as sentinel.
    The sentinel only applies to trending/rotational ambiguity, not to clear volatility expansions.
    """
    np.random.seed(11)
    close = 100 + np.cumsum(np.random.randn(600) * 8)
    labels = generate_regime_labels(make_features(close))
    non_nan = labels.dropna()
    # Clearly volatile bars (fwd_vol >> vol_threshold) should get label 3, not sentinel
    assert (non_nan == 3).sum() > 0, (
        "Volatility-expansion bars should get label 3. "
        "Confidence masking must not suppress class-3 for clear vol expansions."
    )
    assert (non_nan == 3).sum() > (non_nan == LABEL_CONFIDENCE_SENTINEL).sum() * 0.1, (
        "In high-vol market, there should be substantial class-3 labels relative to sentinel."
    )


# =============================================================================
# No-sentinel invariant for reliable heads
# =============================================================================


def test_volatility_has_no_sentinel():
    """Volatility labels are always real classes (0-3) — no confidence masking is applied."""
    np.random.seed(5)
    close = 100 + np.cumsum(np.random.randn(500) * 2)
    labels = generate_volatility_labels(make_features(close))
    non_nan = labels.dropna()
    assert LABEL_CONFIDENCE_SENTINEL not in set(non_nan.unique()), (
        "Volatility head must never produce LABEL_CONFIDENCE_SENTINEL; "
        "it uses percentile ranking, not threshold masking."
    )


def test_range_has_no_sentinel():
    """Range labels are always real classes (0-4) — no confidence masking is applied."""
    np.random.seed(5)
    close = 100 + np.cumsum(np.random.randn(500) * 2)
    labels = generate_range_labels(make_features(close))
    non_nan = labels.dropna()
    assert LABEL_CONFIDENCE_SENTINEL not in set(non_nan.unique()), (
        "Range head must never produce LABEL_CONFIDENCE_SENTINEL; "
        "quintile bucketing is always well-defined."
    )


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
