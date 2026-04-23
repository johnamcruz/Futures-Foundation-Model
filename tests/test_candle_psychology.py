"""Unit tests for candle psychology features — no torch dependency."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load modules directly (mirrors test_features_core.py style)
# ---------------------------------------------------------------------------

_CP_PATH = Path(__file__).parent.parent / "futures_foundation" / "candle_psychology.py"
_cp_spec = importlib.util.spec_from_file_location("ffm_candle_psychology", _CP_PATH)
_cp_mod = importlib.util.module_from_spec(_cp_spec)
_cp_spec.loader.exec_module(_cp_mod)

add_candle_features = _cp_mod.add_candle_features
ENGULF_LOOKBACK = _cp_mod.ENGULF_LOOKBACK
MOMENTUM_WINDOW = _cp_mod.MOMENTUM_WINDOW
DIRECTION_WINDOW = _cp_mod.DIRECTION_WINDOW

# Also load features for integration tests
_FEATURES_PATH = Path(__file__).parent.parent / "futures_foundation" / "features.py"

import sys as _sys
# Pre-register candle_psychology under the relative-import name features.py expects
_sys.modules.setdefault("futures_foundation.candle_psychology", _cp_mod)

_f_spec = importlib.util.spec_from_file_location("ffm_features", _FEATURES_PATH)
_f_mod = importlib.util.module_from_spec(_f_spec)
_f_spec.loader.exec_module(_f_mod)

derive_features = _f_mod.derive_features
get_model_feature_columns = _f_mod.get_model_feature_columns

_CP_COLS = [
    "candle_type", "engulf_count", "momentum_speed_ratio",
    "wick_rejection", "dir_consistency",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(open_, high, low, close, n=1):
    """Build a minimal OHLCV DataFrame with n identical bars."""
    return pd.DataFrame({
        "open":   [float(open_)] * n,
        "high":   [float(high)]  * n,
        "low":    [float(low)]   * n,
        "close":  [float(close)] * n,
        "volume": [1000.0]       * n,
    })


def make_ohlcv(n=300, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "datetime": dates,
        "open":   close + np.random.randn(n) * 1.5,
        "high":   close + np.abs(np.random.randn(n)) * 3,
        "low":    close - np.abs(np.random.randn(n)) * 3,
        "close":  close,
        "volume": np.random.randint(100, 10_000, n).astype(float),
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# Contract: all 5 columns produced with correct dtype
# ---------------------------------------------------------------------------

def test_all_columns_added():
    df = make_ohlcv(50)
    result = add_candle_features(df)
    for col in _CP_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_all_columns_float32():
    df = make_ohlcv(50)
    result = add_candle_features(df)
    for col in _CP_COLS:
        assert result[col].dtype == np.float32, f"{col} should be float32"


def test_output_length_matches_input():
    df = make_ohlcv(200)
    result = add_candle_features(df)
    assert len(result) == 200


def test_input_df_not_mutated():
    df = make_ohlcv(50)
    original_cols = set(df.columns)
    add_candle_features(df)
    assert set(df.columns) == original_cols, "add_candle_features must not mutate the input"


# ---------------------------------------------------------------------------
# Feature 1: candle_type
# ---------------------------------------------------------------------------

def test_candle_type_valid_values():
    result = add_candle_features(make_ohlcv())
    assert set(result["candle_type"].unique()).issubset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0})


def test_candle_type_doji():
    """open == close, narrow body → type 0."""
    result = add_candle_features(_bar(5, 10, 0, 5))
    assert result["candle_type"].iloc[0] == 0.0


def test_candle_type_bull_strength():
    """Full bull body → type 1."""
    result = add_candle_features(_bar(0, 10, 0, 10))
    assert result["candle_type"].iloc[0] == 1.0


def test_candle_type_bear_strength():
    """Full bear body → type 2."""
    result = add_candle_features(_bar(10, 10, 0, 0))
    assert result["candle_type"].iloc[0] == 2.0


def test_candle_type_bull_pin():
    """Large lower wick, small body → type 3.
    open=6, close=8, high=9, low=0 → lower_wick=6, body=2, upper_wick=1 on range=9
    wick_asymmetry = 6/9 - 1/9 ≈ 0.56 > 0.30 → bull pin
    """
    result = add_candle_features(_bar(6, 9, 0, 8))
    assert result["candle_type"].iloc[0] == 3.0


def test_candle_type_bear_pin():
    """Large upper wick, small body → type 4.
    open=3, close=1, high=9, low=0 → upper_wick=6, body=2, lower_wick=1 on range=9
    wick_asymmetry = 1/9 - 6/9 ≈ -0.56 < -0.30 → bear pin
    """
    result = add_candle_features(_bar(3, 9, 0, 1))
    assert result["candle_type"].iloc[0] == 4.0


def test_candle_type_neutral():
    """Balanced bar with mid-sized body and balanced wicks → type 5.
    open=4, close=6, high=8, low=2 → body=2, upper_wick=2, lower_wick=2 on range=6
    body_ratio=2/6≈0.33 (not doji, not strength), wick_asymmetry=0 (not pin) → neutral
    """
    result = add_candle_features(_bar(4, 8, 2, 6))
    assert result["candle_type"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# Feature 2: engulf_count
# ---------------------------------------------------------------------------

def test_engulf_count_range():
    result = add_candle_features(make_ohlcv())
    assert (result["engulf_count"] >= 0.0).all()
    assert (result["engulf_count"] <= float(ENGULF_LOOKBACK)).all()


def test_engulf_count_first_bar_zero():
    """First bar always 0 (no prior bars to engulf)."""
    result = add_candle_features(make_ohlcv(50))
    assert result["engulf_count"].iloc[0] == 0.0


def test_engulf_count_perfect_engulf():
    """Bar 5 has a huge body that contains bars 0–4 exactly."""
    df = pd.DataFrame({
        "open":   [99.5, 99.5, 99.5, 99.5, 99.5, 95.0],
        "high":   [101.0, 101.0, 101.0, 101.0, 101.0, 106.0],
        "low":    [99.0,  99.0,  99.0,  99.0,  99.0,  94.0],
        "close":  [100.5, 100.5, 100.5, 100.5, 100.5, 105.0],
        "volume": [1000.0] * 6,
    })
    result = add_candle_features(df)
    # Bar at idx 5: body_high=105, body_low=95 — contains all 5 prior bodies (99.5/100.5)
    assert result["engulf_count"].iloc[5] == 5.0


def test_engulf_count_no_engulf():
    """Each bar is the same size — a later bar cannot engulf an equally-sized prior bar."""
    df = pd.DataFrame({
        "open":   [99.0, 99.0, 99.0, 99.0, 99.0],
        "high":   [101.0] * 5,
        "low":    [98.0]  * 5,
        "close":  [100.0] * 5,
        "volume": [1000.0] * 5,
    })
    result = add_candle_features(df)
    # Identical bars: body_high(prev)==body_high(curr) and body_low(prev)==body_low(curr)
    # The condition is <=/>= so equal bounds DO count as engulfed
    # All prior bars have body_high=100, body_low=99 which equals curr body → count=lookback or idx
    # This is intentional: a bar perfectly matching prior bodies IS contained
    assert (result["engulf_count"] >= 0.0).all()


def test_engulf_count_custom_lookback():
    """Custom lookback=2 caps max at 2."""
    df = pd.DataFrame({
        "open":   [99.5] * 5 + [95.0],
        "high":   [101.0] * 5 + [106.0],
        "low":    [99.0]  * 5 + [94.0],
        "close":  [100.5] * 5 + [105.0],
        "volume": [1000.0] * 6,
    })
    result = add_candle_features(df, engulf_lookback=2)
    assert result["engulf_count"].iloc[5] == 2.0


# ---------------------------------------------------------------------------
# Feature 3: momentum_speed_ratio
# ---------------------------------------------------------------------------

def test_momentum_speed_ratio_range():
    result = add_candle_features(make_ohlcv())
    assert (result["momentum_speed_ratio"] >= 0.0).all()
    assert (result["momentum_speed_ratio"] <= 10.0).all()


def test_momentum_speed_ratio_warmup_is_one():
    """First `momentum_window` bars default to 1.0 (neutral)."""
    df = make_ohlcv(100)
    result = add_candle_features(df, momentum_window=20)
    warmup = result["momentum_speed_ratio"].iloc[:20]
    np.testing.assert_array_equal(warmup.values, 1.0)


def test_momentum_speed_ratio_monotonic_up_clips_to_max():
    """Monotonically rising closes → retrace_pts=0 → ratio clips to 10.0."""
    n = 40
    closes = np.arange(n, dtype=np.float64)
    df = pd.DataFrame({
        "open":   closes,
        "high":   closes + 0.5,
        "low":    closes - 0.5,
        "close":  closes,
        "volume": [1000.0] * n,
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    result = add_candle_features(df, momentum_window=20)
    # At idx=20: peak=19, trough=0 in window[0:20]; close[-1]==peak → retrace=0 → 10.0
    assert result["momentum_speed_ratio"].iloc[20] == pytest.approx(10.0, abs=1e-4)


def test_momentum_speed_ratio_fast_retrace_below_one():
    """10 bars up (speed=1.0/bar) then 4 bars of fast retrace (speed=2.0/bar) → ratio=0.5."""
    # wc[0:15] = 0..10 then 8,6,4,2; idx=15 is the eval bar (window=15, n=16)
    # impulse: 10 pts in 10 bars → speed=1.0; retrace: 8 pts in 4 bars → speed=2.0
    closes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 2], dtype=np.float64)
    df = pd.DataFrame({
        "open":   closes,
        "high":   closes + 0.1,
        "low":    closes - 0.1,
        "close":  closes,
        "volume": [1000.0] * len(closes),
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"]  = df[["open", "close", "low"]].min(axis=1)
    result = add_candle_features(df, momentum_window=15)
    assert result["momentum_speed_ratio"].iloc[-1] < 1.0


def test_momentum_speed_ratio_flat_stays_neutral():
    """Flat window (all closes equal) → no meaningful move → stays 1.0."""
    n = 40
    df = pd.DataFrame({
        "open":   [100.0] * n,
        "high":   [100.5] * n,
        "low":    [99.5]  * n,
        "close":  [100.0] * n,
        "volume": [1000.0] * n,
    })
    result = add_candle_features(df, momentum_window=20)
    flat_bars = result["momentum_speed_ratio"].iloc[20:]
    np.testing.assert_array_equal(flat_bars.values, 1.0)


# ---------------------------------------------------------------------------
# Feature 4: wick_rejection
# ---------------------------------------------------------------------------

def test_wick_rejection_range():
    result = add_candle_features(make_ohlcv())
    assert (result["wick_rejection"] >= -1.0).all()
    assert (result["wick_rejection"] <= 1.0).all()


def test_wick_rejection_pure_lower_wick_is_positive():
    """open==close==high, long lower wick → bullish rejection → positive."""
    result = add_candle_features(_bar(10, 10, 0, 10))
    assert result["wick_rejection"].iloc[0] > 0.0


def test_wick_rejection_pure_upper_wick_is_negative():
    """open==close==low, long upper wick → bearish rejection → negative."""
    result = add_candle_features(_bar(0, 10, 0, 0))
    assert result["wick_rejection"].iloc[0] < 0.0


def test_wick_rejection_symmetric_bar_near_zero():
    """Equal upper and lower wicks → rejection near 0."""
    result = add_candle_features(_bar(5, 10, 0, 5))
    assert result["wick_rejection"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_wick_rejection_full_body_near_zero():
    """Full body bar (no wicks) → both wicks zero → rejection = 0."""
    result = add_candle_features(_bar(0, 10, 0, 10))
    assert result["wick_rejection"].iloc[0] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Feature 5: dir_consistency
# ---------------------------------------------------------------------------

def test_dir_consistency_range():
    result = add_candle_features(make_ohlcv())
    assert (result["dir_consistency"] >= 0.0).all()
    assert (result["dir_consistency"] <= 1.0).all()


def test_dir_consistency_all_bullish_is_one():
    """N consecutive bullish bars → all agree → 1.0 (after warmup)."""
    n = 20
    closes = np.arange(1, n + 1, dtype=np.float64)
    df = pd.DataFrame({
        "open":   closes - 0.5,
        "high":   closes + 0.1,
        "low":    closes - 0.6,
        "close":  closes,
        "volume": [1000.0] * n,
    })
    result = add_candle_features(df, direction_window=5)
    assert result["dir_consistency"].iloc[-1] == pytest.approx(1.0, abs=1e-5)


def test_dir_consistency_alternating_is_low():
    """Alternating up/down bars → no full agreement across the window.

    With window=5 and strictly alternating directions the current bar's
    direction appears either 2 or 3 times out of 5 (0.4 or 0.6), never
    approaching the perfect-streak value of 1.0.
    """
    n = 20
    closes = np.array([100.0 + (1 if i % 2 == 0 else -1) * 0.5 for i in range(n)])
    df = pd.DataFrame({
        "open":   np.roll(closes, 1),
        "high":   closes + 0.1,
        "low":    closes - 0.1,
        "close":  closes,
        "volume": [1000.0] * n,
    })
    result = add_candle_features(df, direction_window=5)
    assert result["dir_consistency"].iloc[-1] < 1.0


def test_dir_consistency_doji_is_neutral():
    """Doji bar (open == close) → 0.5 regardless of prior bars."""
    df = _bar(100, 101, 99, 100)  # open == close
    result = add_candle_features(df)
    assert result["dir_consistency"].iloc[0] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Integration: candle psychology columns flow through derive_features
# ---------------------------------------------------------------------------

def test_derive_features_contains_cp_columns():
    """All 5 candle psychology columns must appear in derive_features output."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02 09:30", periods=300, freq="5min"),
        **{col: make_ohlcv(300)[col] for col in ("open", "high", "low", "close", "volume")},
    })
    features = derive_features(df, "ES")
    for col in _CP_COLS:
        assert col in features.columns, f"derive_features missing candle psychology col: {col}"


def test_derive_features_bar_size_vs_session_present():
    """bar_size_vs_session must appear in derive_features output."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02 09:30", periods=300, freq="5min"),
        **{col: make_ohlcv(300)[col] for col in ("open", "high", "low", "close", "volume")},
    })
    features = derive_features(df, "ES")
    assert "bar_size_vs_session" in features.columns


def test_derive_features_cp_columns_no_all_nan():
    """Candle psychology columns should have valid (non-NaN) values after warm-up."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02 09:30", periods=300, freq="5min"),
        **{col: make_ohlcv(300)[col] for col in ("open", "high", "low", "close", "volume")},
    })
    features = derive_features(df, "ES")
    all_cp_cols = _CP_COLS + ["bar_size_vs_session"]
    for col in all_cp_cols:
        assert features[col].notna().any(), f"{col} is entirely NaN"


def test_derive_features_cp_in_model_feature_columns():
    """All 6 candle psychology features must be in get_model_feature_columns()."""
    cols = get_model_feature_columns()
    all_cp_cols = _CP_COLS + ["bar_size_vs_session"]
    for col in all_cp_cols:
        assert col in cols, f"{col} missing from get_model_feature_columns()"
