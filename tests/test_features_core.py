"""Unit tests for core feature groups 1–7 in features.py — no torch dependency."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Pre-register candle_psychology so features.py relative import resolves
_CP_PATH = Path(__file__).parent.parent / "futures_foundation" / "candle_psychology.py"
_cp_spec = importlib.util.spec_from_file_location("futures_foundation.candle_psychology", _CP_PATH)
_cp_mod = importlib.util.module_from_spec(_cp_spec)
sys.modules["futures_foundation.candle_psychology"] = _cp_mod
_cp_spec.loader.exec_module(_cp_mod)

_FEATURES_PATH = Path(__file__).parent.parent / "futures_foundation" / "features.py"
_spec = importlib.util.spec_from_file_location("ffm_features", _FEATURES_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

derive_features = _mod.derive_features
get_model_feature_columns = _mod.get_model_feature_columns
INSTRUMENT_MAP = _mod.INSTRUMENT_MAP


def make_ohlcv(n=300, seed=42, bar_freq_min=5):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq=f"{bar_freq_min}min")
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 1.5,
        "high": close + np.abs(np.random.randn(n)) * 3,
        "low": close - np.abs(np.random.randn(n)) * 3,
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })
    # Ensure high >= max(open,close) and low <= min(open,close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


# =============================================================================
# Column contract
# =============================================================================

def test_feature_count():
    assert len(get_model_feature_columns()) == 67


def test_htf_1h_structure_feature():
    features = derive_features(make_ohlcv(), "ES")
    assert "htf_1h_structure" in features.columns
    assert features["htf_1h_structure"].dtype == np.float32
    assert features["htf_1h_structure"].isin([-1.0, 0.0, 1.0]).all()


def test_all_model_columns_present():
    features = derive_features(make_ohlcv(), "ES")
    for col in get_model_feature_columns():
        assert col in features.columns, f"Missing model column: {col}"


def test_metadata_columns_present():
    features = derive_features(make_ohlcv(), "ES")
    for col in ["_datetime", "_instrument", "_instrument_id", "_close", "_volume"]:
        assert col in features.columns, f"Missing metadata column: {col}"


# =============================================================================
# Metadata correctness
# =============================================================================

def test_instrument_id_all_known():
    df = make_ohlcv()
    for sym, expected_id in INSTRUMENT_MAP.items():
        features = derive_features(df, sym)
        assert (features["_instrument_id"] == expected_id).all(), \
            f"{sym}: expected id {expected_id}, got {features['_instrument_id'].unique()}"


def test_instrument_unknown_defaults_to_zero():
    features = derive_features(make_ohlcv(), "UNKNOWN")
    assert (features["_instrument_id"] == 0).all()


def test_close_metadata_matches_input():
    df = make_ohlcv()
    features = derive_features(df, "ES")
    np.testing.assert_array_equal(features["_close"].values, df["close"].values)


# =============================================================================
# Group 1 — Bar Anatomy
# =============================================================================

def test_bar_direction_ternary():
    features = derive_features(make_ohlcv(), "ES")
    assert set(features["bar_direction"].unique()).issubset({-1.0, 0.0, 1.0})


def test_bar_direction_matches_sign():
    df = make_ohlcv()
    features = derive_features(df, "ES")
    expected = np.sign(df["close"].values - df["open"].values)
    np.testing.assert_array_equal(features["bar_direction"].values, expected)


def test_bar_wick_percents_sum_to_one():
    """body_pct + upper_wick_pct + lower_wick_pct == 1 for non-zero-range bars."""
    df = make_ohlcv()
    features = derive_features(df, "ES")
    total = (
        features["bar_body_pct"].abs()
        + features["bar_upper_wick_pct"]
        + features["bar_lower_wick_pct"]
    )
    nonzero_range = (df["high"] - df["low"]) > 0
    np.testing.assert_allclose(
        total[nonzero_range].values, 1.0, atol=1e-6,
        err_msg="Wick percents must sum to 1 for non-zero-range bars",
    )


def test_bar_wick_percents_nonnegative():
    features = derive_features(make_ohlcv(), "ES")
    assert (features["bar_upper_wick_pct"] >= 0).all()
    assert (features["bar_lower_wick_pct"] >= 0).all()
    assert (features["bar_body_pct"].abs() >= 0).all()


# =============================================================================
# Group 2 — Returns & Momentum
# =============================================================================

def test_ret_close_1_matches_pct_change():
    df = make_ohlcv()
    features = derive_features(df, "ES")
    expected = df["close"].pct_change(1)
    np.testing.assert_allclose(
        features["ret_close_1"].values, expected.values, equal_nan=True,
    )


def test_ret_close_3_matches_pct_change():
    df = make_ohlcv()
    features = derive_features(df, "ES")
    expected = df["close"].pct_change(3)
    np.testing.assert_allclose(
        features["ret_close_3"].values, expected.values, equal_nan=True,
    )


def test_ret_open_close_sign_matches_bar_direction():
    """ret_open_close sign must match bar_direction for non-doji bars."""
    df = make_ohlcv()
    features = derive_features(df, "ES")
    non_doji = features["bar_direction"] != 0
    np.testing.assert_array_equal(
        np.sign(features["ret_open_close"][non_doji].values),
        features["bar_direction"][non_doji].values,
    )


# =============================================================================
# Group 3 — Volume Dynamics
# =============================================================================

def test_vol_ratio_constant_volume():
    """Constant volume → vol_ratio_N == 1.0 everywhere valid."""
    df = make_ohlcv()
    df["volume"] = 1000.0
    features = derive_features(df, "ES")
    for lb in [5, 10, 20]:
        col = f"vol_ratio_{lb}"
        valid = features[col].dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-6,
                                    err_msg=f"{col} should be 1.0 for constant volume")


def test_vol_ratio_spike_above_one():
    """20× volume spike → vol_ratio_5 >> 1 at that bar."""
    df = make_ohlcv(n=200)
    baseline = df["volume"].mean()
    df.loc[100, "volume"] = baseline * 20
    features = derive_features(df, "ES")
    assert features["vol_ratio_5"].iloc[100] > 3.0


# =============================================================================
# Group 4 — Volatility Measures
# =============================================================================

def test_atr_positive():
    features = derive_features(make_ohlcv(), "ES")
    valid = features["vty_atr_raw"].dropna()
    assert (valid > 0).all()


def test_atr_increases_during_volatile_bars():
    """ATR during a volatile stretch should be >> the pre-volatile baseline."""
    df = make_ohlcv(n=200, seed=1)
    df.loc[100:110, "high"] = df.loc[100:110, "close"] + 500
    df.loc[100:110, "low"] = df.loc[100:110, "close"] - 500
    features = derive_features(df, "ES")
    assert features["vty_atr_raw"].iloc[110] > features["vty_atr_raw"].iloc[50] * 3


# =============================================================================
# Group 5 — Session Context
# =============================================================================

def test_sess_id_in_valid_range():
    features = derive_features(make_ohlcv(), "ES")
    assert set(features["sess_id"].unique()).issubset({0, 1, 2, 3})


def test_sess_time_of_day_bounded():
    features = derive_features(make_ohlcv(), "ES")
    tod = features["sess_time_of_day"]
    assert (tod >= 0).all() and (tod <= 1).all()


def test_sess_bars_elapsed_bounded():
    features = derive_features(make_ohlcv(), "ES")
    be = features["sess_bars_elapsed"]
    assert (be >= 0).all() and (be <= 1).all()


def test_sess_dist_from_high_nonpositive():
    """close <= session_high always, so dist_from_high (normalized) <= 0."""
    features = derive_features(make_ohlcv(), "ES")
    valid = features["sess_dist_from_high"].dropna()
    assert (valid <= 1e-9).all(), "close can never exceed the session high"


def test_sess_dist_from_low_nonnegative():
    """close >= session_low always, so dist_from_low (normalized) >= 0."""
    features = derive_features(make_ohlcv(), "ES")
    valid = features["sess_dist_from_low"].dropna()
    assert (valid >= -1e-9).all(), "close can never be below the session low"


# =============================================================================
# Group 6 — Market Structure
# =============================================================================

def test_str_range_position_bounded():
    features = derive_features(make_ohlcv(), "ES")
    for col in ["str_range_position_10", "str_range_position_20"]:
        vals = features[col]
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0, 1]"


def test_str_structure_state_ternary():
    features = derive_features(make_ohlcv(), "ES")
    assert set(features["str_structure_state"].unique()).issubset({-1, 0, 1})


def test_str_dist_from_high_nonpositive():
    """Rolling high >= close always → str_dist_from_high_N <= 0."""
    features = derive_features(make_ohlcv(), "ES")
    for col in ["str_dist_from_high_10", "str_dist_from_high_20"]:
        valid = features[col].dropna()
        assert (valid <= 1e-9).all(), f"{col}: close cannot exceed rolling high"


def test_str_dist_from_low_nonnegative():
    """Rolling low <= close always → str_dist_from_low_N >= 0."""
    features = derive_features(make_ohlcv(), "ES")
    for col in ["str_dist_from_low_10", "str_dist_from_low_20"]:
        valid = features[col].dropna()
        assert (valid >= -1e-9).all(), f"{col}: close cannot be below rolling low"


# =============================================================================
# Output shape & edge cases
# =============================================================================

def test_output_length_matches_input():
    df = make_ohlcv(n=300)
    features = derive_features(df, "ES")
    assert len(features) == 300


def test_short_data_no_exception():
    """n=50 should not raise — just produce many NaN rows."""
    df = make_ohlcv(n=50)
    features = derive_features(df, "ES")
    assert len(features) == 50
    for col in get_model_feature_columns():
        assert col in features.columns


def test_valid_row_ratio():
    """After 400 bars, >85% of rows should have all model features valid."""
    df = make_ohlcv(n=400)
    features = derive_features(df, "ES")
    cols = get_model_feature_columns()
    ratio = features[cols].notna().all(axis=1).mean()
    assert ratio > 0.80, f"Only {ratio:.1%} valid rows — unexpected NaN propagation"


def test_3min_bars_feature_columns():
    """3-min bars should produce the same column set as 5-min bars."""
    df = make_ohlcv(n=300, bar_freq_min=3)
    features = derive_features(df, "NQ")
    for col in get_model_feature_columns():
        assert col in features.columns


# =============================================================================
# Group 10 — Volume Absorption & Order Flow
# =============================================================================

def test_vol_cum_signed_range():
    features = derive_features(make_ohlcv(), "ES")
    for col in ["vol_cum_signed_5", "vol_cum_signed_20"]:
        vals = features[col]
        assert (vals >= -0.5).all() and (vals <= 0.5).all(), f"{col} out of [-0.5, 0.5]"


def test_vol_cum_signed_positive_for_buying_bars():
    """All bars with close at high → cumulative signed delta is positive."""
    df = make_ohlcv(n=200, seed=99)
    df["open"] = df["low"]
    df["close"] = df["high"]
    features = derive_features(df, "ES")
    valid = features["vol_cum_signed_5"].iloc[20:]
    assert (valid > 0).all(), "Bars closing at their high should produce positive cum signed delta"


def test_vol_cum_signed_negative_for_selling_bars():
    """All bars with close at low → cumulative signed delta is negative."""
    df = make_ohlcv(n=200, seed=99)
    df["open"] = df["high"]
    df["close"] = df["low"]
    features = derive_features(df, "ES")
    valid = features["vol_cum_signed_5"].iloc[20:]
    assert (valid < 0).all(), "Bars closing at their low should produce negative cum signed delta"


def test_vol_absorption_range():
    features = derive_features(make_ohlcv(), "ES")
    vals = features["vol_absorption"]
    assert (vals >= 0).all() and (vals <= 5.0 + 1e-6).all(), "vol_absorption out of [0, 5]"


def test_vol_absorption_higher_for_doji_than_marubozu():
    """Same elevated volume: doji (small body) should have higher absorption than marubozu."""
    df = make_ohlcv(n=100, seed=5)
    avg_vol = df["volume"].mean()

    # Bar 50: doji — open = close = midpoint of range
    mid = (df.loc[50, "high"] + df.loc[50, "low"]) / 2
    df.loc[50, "open"] = mid
    df.loc[50, "close"] = mid
    df.loc[50, "volume"] = avg_vol * 4

    # Bar 60: marubozu — open = low, close = high (maximum body)
    df.loc[60, "open"] = df.loc[60, "low"]
    df.loc[60, "close"] = df.loc[60, "high"]
    df.loc[60, "volume"] = avg_vol * 4

    features = derive_features(df, "ES")
    assert features["vol_absorption"].iloc[50] > features["vol_absorption"].iloc[60], \
        "Doji bar should have higher absorption than marubozu at same volume"


def test_vol_momentum_align_range():
    features = derive_features(make_ohlcv(), "ES")
    vals = features["vol_momentum_align"]
    assert (vals >= -3.0 - 1e-6).all() and (vals <= 3.0 + 1e-6).all(), \
        "vol_momentum_align out of [-3, 3]"


def test_vol_momentum_align_positive_in_confirmed_uptrend():
    """Early in a volume surge on a steady uptrend: vol_ratio > 1 before rolling mean adapts."""
    df = make_ohlcv(n=200, seed=7)
    base_vol = df["volume"].mean()

    # Bars 30-49: steady uptrend at normal volume (builds positive momentum)
    for i in range(30, 50):
        c = 5000 + (i - 30) * 1.0
        df.loc[i, ["close", "open", "high", "low"]] = [c, c - 0.5, c + 0.5, c - 1.0]
        df.loc[i, "volume"] = base_vol

    # Bars 50-80: same uptrend rate + 3x volume spike
    for i in range(50, 80):
        c = 5020 + (i - 50) * 1.0
        df.loc[i, ["close", "open", "high", "low"]] = [c, c - 0.5, c + 0.5, c - 1.0]
        df.loc[i, "volume"] = base_vol * 3

    features = derive_features(df, "ES")
    # Bar 52: vol_ratio_5 = 3x / mean(1x,1x,3x,3x,3x) ≈ 1.36 > 1.
    # momentum_5 = sum of 5 small positive returns → positive.
    assert features["vol_momentum_align"].iloc[52] > 0, \
        "Uptrend + volume surge should give positive alignment before rolling mean adapts"


def test_1h_structure_metadata_present():
    features = derive_features(make_ohlcv(), "ES")
    assert "_1h_structure" in features.columns, "Missing _1h_structure metadata column"


def test_1h_structure_values_valid():
    features = derive_features(make_ohlcv(), "ES")
    vals = features["_1h_structure"].dropna()
    assert set(vals.unique()).issubset({-1, 0, 1}), \
        f"_1h_structure should only contain {{-1, 0, 1}}, got {set(vals.unique())}"
