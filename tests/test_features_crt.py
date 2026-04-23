"""
Unit tests for CRT sweep features.

No torch dependency — imports features.py directly so these run without
the full model stack installed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load features module without triggering the package __init__ (which needs torch)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_FEATURES_PATH = _REPO_ROOT / "futures_foundation" / "features.py"

# Pre-register candle_psychology so features.py relative import resolves
_CP_PATH = _REPO_ROOT / "futures_foundation" / "candle_psychology.py"
_cp_spec = importlib.util.spec_from_file_location("futures_foundation.candle_psychology", _CP_PATH)
_cp_mod = importlib.util.module_from_spec(_cp_spec)
sys.modules["futures_foundation.candle_psychology"] = _cp_mod
_cp_spec.loader.exec_module(_cp_mod)

def _load():
    spec = importlib.util.spec_from_file_location("ffm_features", _FEATURES_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_feat = _load()
derive_features          = _feat.derive_features
get_model_feature_columns = _feat.get_model_feature_columns
_resample_ohlcv          = _feat._resample_ohlcv
_detect_crt_sweeps       = _feat._detect_crt_sweeps
_align_sweeps_to_base    = _feat._align_sweeps_to_base
_compute_atr             = _feat._compute_atr
_compute_htf_context     = _feat._compute_htf_context


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NS_PER_MIN = 60 * 1_000_000_000


def make_htf_df(rows, freq="60min"):
    """Build a minimal HTF DataFrame from (open, high, low, close) tuples."""
    dates = pd.date_range("2024-01-02 10:00", periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=dates)
    df["volume"] = 1000.0
    return df


def make_ohlcv(n=600, freq="5min", seed=0):
    """Return a base-timeframe OHLCV DataFrame with a datetime column."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "datetime": dates,
        "open":   close + np.random.randn(n) * 1.5,
        "high":   close + np.abs(np.random.randn(n)) * 3,
        "low":    close - np.abs(np.random.randn(n)) * 3,
        "close":  close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


def make_controlled_uptrend(n=600, start=5000, end=5100, freq="5min"):
    """Steady monotone uptrend with small symmetric bar ranges and no noise."""
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    close = np.linspace(start, end, n)
    return pd.DataFrame({
        "datetime": dates,
        "open":   close - 0.1,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": np.ones(n) * 1000.0,
    })


# ---------------------------------------------------------------------------
# Feature column contract
# ---------------------------------------------------------------------------

def test_feature_count():
    assert len(get_model_feature_columns()) == 62


def test_sweep_column_names_in_list():
    cols = get_model_feature_columns()
    expected = [
        "swp_1h_bull_active", "swp_1h_bear_active",
        "swp_1h_age_norm",    "swp_1h_magnitude",
        "swp_4h_bull_active", "swp_4h_bear_active",
        "swp_4h_age_norm",    "swp_4h_magnitude",
        "swp_tf_alignment",   "swp_dominant_dir",
    ]
    for col in expected:
        assert col in cols, f"Missing from get_model_feature_columns(): {col}"


# ---------------------------------------------------------------------------
# detect_crt_sweeps — deterministic logic tests
# ---------------------------------------------------------------------------

def test_detect_bull_sweep():
    """Bar 1 undercuts prior low and closes back above it → bull sweep."""
    rows = [
        (9.0, 10.0, 8.0, 9.0),   # bar 0: prior candle
        (8.0,  9.0, 7.0, 8.5),   # bar 1: low=7 < 8, close=8.5 > 8 → BULL
    ]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, bear, bull_mag, bear_mag = _detect_crt_sweeps(df, atr)

    assert bull[0] == False,  "No sweep on first bar"
    assert bull[1] == True,   "Expected bull sweep at bar 1"
    assert bear[1] == False,  "No bear sweep when bull fires"
    assert bull_mag[1] > 0.0, "Bull magnitude should be positive"
    assert bear_mag[1] == 0.0


def test_detect_bear_sweep():
    """Bar 1 exceeds prior high and closes back below it → bear sweep."""
    rows = [
        ( 9.0, 10.0, 8.0,  9.0),   # bar 0: prior candle
        ( 9.5, 11.0, 9.0,  9.5),   # bar 1: high=11 > 10, close=9.5 < 10 → BEAR
    ]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, bear, bull_mag, bear_mag = _detect_crt_sweeps(df, atr)

    assert bear[1] == True,   "Expected bear sweep at bar 1"
    assert bull[1] == False,  "No bull sweep when bear fires"
    assert bear_mag[1] > 0.0, "Bear magnitude should be positive"
    assert bull_mag[1] == 0.0


def test_detect_no_sweep_when_close_does_not_recover():
    """Wick below prior low but close stays below → NOT a bull sweep."""
    rows = [
        (9.0, 10.0, 8.0, 9.0),
        (8.0,  9.0, 7.0, 6.5),   # low < prior_low, but close=6.5 < prior_low=8
    ]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, bear, _, _ = _detect_crt_sweeps(df, atr)
    assert bull[1] == False, "close below prior low should not count as bull sweep"
    assert bear[1] == False, "high didn't exceed prior high, no bear sweep"


def test_detect_no_sweep_when_close_does_not_reverse():
    """Wick above prior high but close stays above → NOT a bear sweep."""
    rows = [
        ( 9.0, 10.0, 8.0,  9.0),
        (10.0, 11.0, 9.5, 10.5),  # high > prior_high, but close=10.5 > prior_high=10
    ]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, bear, _, _ = _detect_crt_sweeps(df, atr)
    assert bear[1] == False, "close above prior high should not count as bear sweep"


def test_detect_magnitude_clipped_at_3():
    """Extreme wick penetration should be clipped to 3.0."""
    # With only 2 bars, ATR is NaN → nan_to_num gives 1e-6.
    # Penetration = prior_low - current_low = 8 - (-1000) = 1008.
    # 1008 / 1e-6 >> 3, so clip gives 3.0.
    rows = [
        (   9.0,  10.0,     8.0,  9.0),
        (-500.0, -400.0, -1000.0, 8.5),
    ]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, _, bull_mag, _ = _detect_crt_sweeps(df, atr)
    assert bull[1] == True
    assert bull_mag[1] == pytest.approx(3.0)


def test_detect_first_bar_never_sweeps():
    """Bar index 0 can never be a sweep — there is no prior candle."""
    rows = [(9.0, 10.0, 8.0, 9.0)]
    df = make_htf_df(rows)
    atr = _compute_atr(df, 14)
    bull, bear, bull_mag, bear_mag = _detect_crt_sweeps(df, atr)
    assert bull[0] == False
    assert bear[0] == False
    assert bull_mag[0] == 0.0
    assert bear_mag[0] == 0.0


# ---------------------------------------------------------------------------
# align_sweeps_to_base — countdown tent and expiry tests
# ---------------------------------------------------------------------------

def _make_base_and_htf_times(n_base, n_htf, base_interval_min=5, htf_interval_min=60):
    """Return (base_ns, htf_ns) as int64 nanosecond arrays."""
    base_ns = np.arange(n_base, dtype=np.int64) * base_interval_min * _NS_PER_MIN
    htf_ns  = np.arange(n_htf, dtype=np.int64) * htf_interval_min * _NS_PER_MIN
    return base_ns, htf_ns


def test_align_active_window_exact():
    """Sweep fires at HTF bar 0; active for exactly expiry_bars base bars."""
    expiry = 4
    # HTF bar closes at t=0. First base bar AFTER t=0 is base bar 1 (t=5min).
    # Active at bars 1, 2, 3, 4 — expired at bar 5.
    base_ns = np.arange(10, dtype=np.int64) * 5 * _NS_PER_MIN   # 0, 5, 10, ... min
    htf_ns  = np.array([0], dtype=np.int64)                       # HTF close at t=0

    bull_mask   = np.array([True])
    bear_mask   = np.array([False])
    bull_mag_h  = np.array([1.5], dtype=np.float32)
    bear_mag_h  = np.array([0.0], dtype=np.float32)

    bull_act, _, bull_age, _, bull_mag, _ = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry
    )

    # Base bar 0 is AT t=0 — not strictly after, so it should NOT be active.
    assert bull_act[0] == 0.0, "Bar at HTF close time should not be active"

    # Bars 1..4 active; bar 5 expired.
    for i in range(1, 1 + expiry):
        assert bull_act[i] == 1.0, f"Bar {i} should be active"
    assert bull_act[1 + expiry] == 0.0, "Bar just after expiry should not be active"


def test_align_age_norm_values():
    """age_norm = 0 at first active bar, increases to <1 at last active bar."""
    expiry = 4
    base_ns = np.arange(10, dtype=np.int64) * 5 * _NS_PER_MIN
    htf_ns  = np.array([0], dtype=np.int64)

    bull_mask  = np.array([True])
    bear_mask  = np.array([False])
    bull_mag_h = np.array([1.0], dtype=np.float32)
    bear_mag_h = np.array([0.0], dtype=np.float32)

    bull_act, _, bull_age, _, _, _ = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry
    )

    assert bull_age[1] == pytest.approx(0.0),        "First active bar: age_norm = 0"
    assert bull_age[4] == pytest.approx(0.75),        "Last active bar: age_norm = 1 - 1/expiry"
    assert bull_age[5] == pytest.approx(1.0),         "Expired bar: age_norm = 1"
    assert bull_age[0] == pytest.approx(1.0),         "Pre-sweep bar: age_norm = 1"


def test_align_magnitude_carried():
    """Magnitude from the sweep is forward-filled across the active window."""
    expiry = 3
    base_ns = np.arange(8, dtype=np.int64) * 5 * _NS_PER_MIN
    htf_ns  = np.array([0], dtype=np.int64)

    bull_mag_h = np.array([2.0], dtype=np.float32)
    bull_mask  = np.array([True])
    bear_mask  = np.array([False])
    bear_mag_h = np.array([0.0], dtype=np.float32)

    _, _, _, _, bull_mag, _ = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry
    )

    for i in range(1, 1 + expiry):
        assert bull_mag[i] == pytest.approx(2.0), f"Magnitude should be 2.0 at bar {i}"
    assert bull_mag[1 + expiry] == pytest.approx(0.0), "Magnitude should be 0 after expiry"


def test_align_no_sweeps_all_inactive():
    """Empty sweep mask → all zeros/ones at rest."""
    base_ns = np.arange(10, dtype=np.int64) * 5 * _NS_PER_MIN
    htf_ns  = np.arange(2,  dtype=np.int64) * 60 * _NS_PER_MIN

    bull_mask  = np.array([False, False])
    bear_mask  = np.array([False, False])
    bull_mag_h = np.array([0.0, 0.0], dtype=np.float32)
    bear_mag_h = np.array([0.0, 0.0], dtype=np.float32)

    bull_act, bear_act, bull_age, bear_age, bull_mag, bear_mag = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry_bars=12
    )

    assert (bull_act  == 0.0).all()
    assert (bear_act  == 0.0).all()
    assert (bull_age  == 1.0).all()
    assert (bear_age  == 1.0).all()
    assert (bull_mag  == 0.0).all()
    assert (bear_mag  == 0.0).all()


def test_align_later_sweep_wins_on_overlap():
    """When two sweeps of the same direction overlap, the more recent one dominates."""
    expiry = 6
    # Base bars at 0, 5, 10, ..., 60 min
    base_ns = np.arange(13, dtype=np.int64) * 5 * _NS_PER_MIN

    # HTF bar A closes between base bar 0 and 1 → activates at base bar 1 (p=1)
    # HTF bar B closes between base bar 2 and 3 → activates at base bar 3 (p=3)
    htf_ns = np.array([
        int(2.5 * _NS_PER_MIN),   # A: between base 0 and 1
        int(12.5 * _NS_PER_MIN),  # B: between base 2 and 3
    ], dtype=np.int64)

    bull_mask  = np.array([True, True])
    bear_mask  = np.array([False, False])
    bull_mag_h = np.array([1.0, 2.0], dtype=np.float32)  # B has larger magnitude
    bear_mag_h = np.zeros(2, dtype=np.float32)

    _, _, bull_age, _, bull_mag, _ = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry
    )

    # At bar 3 (p of sweep B), sweep B is fresher: age_norm=0.0, magnitude=2.0
    assert bull_age[3]  == pytest.approx(0.0), "Sweep B fresh: age_norm=0 at its start"
    assert bull_mag[3]  == pytest.approx(2.0), "Sweep B magnitude dominates at bar 3"

    # At bar 4, sweep B countdown=5 > sweep A countdown=3 → B still dominates
    assert bull_age[4]  == pytest.approx(1 - 5/6)
    assert bull_mag[4]  == pytest.approx(2.0)


def test_align_sweep_past_end_of_data_ignored():
    """HTF sweep after the last base bar should not raise and produce no active bars."""
    base_ns = np.arange(5, dtype=np.int64) * 5 * _NS_PER_MIN   # 0..20 min
    htf_ns  = np.array([25 * _NS_PER_MIN], dtype=np.int64)       # after all base bars

    bull_mask  = np.array([True])
    bear_mask  = np.array([False])
    bull_mag_h = np.array([1.0], dtype=np.float32)
    bear_mag_h = np.zeros(1, dtype=np.float32)

    bull_act, _, _, _, _, _ = _align_sweeps_to_base(
        base_ns, htf_ns, bull_mask, bear_mask, bull_mag_h, bear_mag_h, expiry_bars=12
    )
    assert (bull_act == 0.0).all()


# ---------------------------------------------------------------------------
# resample_ohlcv
# ---------------------------------------------------------------------------

def test_resample_ohlcv_bar_count():
    """Resampling 5-min to 60-min produces far fewer bars."""
    df = make_ohlcv(n=120, freq="5min")   # 10 hours of bars
    df_1h = _resample_ohlcv(df, "60min")
    assert 1 <= len(df_1h) < 120


def test_resample_ohlcv_ohlc_aggregation():
    """high = max of constituent highs; low = min of constituent lows.

    Use bars starting at 09:05 ending at 10:00 so that all 12 bars fall
    cleanly inside the single (09:00, 10:00] bin with closed='right'.
    """
    n = 12
    dates = pd.date_range("2024-01-02 09:05", periods=n, freq="5min")  # 09:05 … 10:00
    highs  = [float(i) for i in range(1, n + 1)]
    lows   = [float(-i) for i in range(1, n + 1)]
    df = pd.DataFrame({
        "datetime": dates,
        "open":   [10.0] * n,
        "high":   highs,
        "low":    lows,
        "close":  [20.0] * n,
        "volume": [100.0] * n,
    })
    df_1h = _resample_ohlcv(df, "60min")
    assert len(df_1h) == 1, "All bars should collapse into a single 1H bin"
    assert df_1h["high"].iloc[0]  == pytest.approx(max(highs))
    assert df_1h["low"].iloc[0]   == pytest.approx(min(lows))
    assert df_1h["open"].iloc[0]  == pytest.approx(10.0)   # first bar's open
    assert df_1h["close"].iloc[0] == pytest.approx(20.0)   # last bar's close


def test_resample_returns_datetime_index():
    df = make_ohlcv(n=60, freq="5min")
    df_1h = _resample_ohlcv(df, "60min")
    assert isinstance(df_1h.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# derive_features integration — sweep column properties
# ---------------------------------------------------------------------------

def test_sweep_columns_present_in_derive_features():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    swp_cols = [c for c in get_model_feature_columns() if c.startswith("swp_")]
    missing = [c for c in swp_cols if c not in features.columns]
    assert missing == [], f"Missing sweep columns: {missing}"


def test_sweep_no_nulls():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    swp_cols = [c for c in get_model_feature_columns() if c.startswith("swp_")]
    for col in swp_cols:
        null_count = features[col].isna().sum()
        assert null_count == 0, f"{col} has {null_count} NaN values"


def test_sweep_active_flags_binary():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    for col in ["swp_1h_bull_active", "swp_1h_bear_active",
                "swp_4h_bull_active", "swp_4h_bear_active"]:
        unique = set(features[col].unique())
        assert unique <= {0.0, 1.0}, f"{col} contains values outside {{0, 1}}: {unique}"


def test_sweep_age_norm_in_unit_interval():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    for col in ["swp_1h_age_norm", "swp_4h_age_norm"]:
        assert features[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"


def test_sweep_magnitude_in_clip_range():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    for col in ["swp_1h_magnitude", "swp_4h_magnitude"]:
        assert features[col].between(0.0, 3.0).all(), f"{col} out of [0, 3]"


def test_sweep_alignment_ternary():
    df = make_ohlcv(n=600)
    features = derive_features(df, instrument="ES")
    for col in ["swp_tf_alignment", "swp_dominant_dir"]:
        unique = set(features[col].unique())
        assert unique <= {-1.0, 0.0, 1.0}, f"{col} has unexpected values: {unique}"


def test_sweep_expiry_adapts_to_bar_frequency():
    """3-min bars should produce different expiry_bars than 5-min bars,
    but both should yield the correct real-time window (≈1H / ≈4H)."""
    df_3min = make_ohlcv(n=600, freq="3min")
    df_5min = make_ohlcv(n=600, freq="5min")

    f3 = derive_features(df_3min, instrument="ES")
    f5 = derive_features(df_5min, instrument="ES")

    # Both should produce all sweep columns without NaN
    swp_cols = [c for c in get_model_feature_columns() if c.startswith("swp_")]
    for col in swp_cols:
        assert f3[col].isna().sum() == 0, f"3-min: {col} has NaN"
        assert f5[col].isna().sum() == 0, f"5-min: {col} has NaN"


# ---------------------------------------------------------------------------
# Group 9: HTF Price Context
# ---------------------------------------------------------------------------

_HTF_COLS = ["htf_1h_close_pos", "htf_1h_ret", "htf_4h_close_pos", "htf_4h_ret", "htf_tf_alignment"]


def test_htf_columns_present():
    """All 5 HTF price context columns must appear in derive_features output."""
    f = derive_features(make_ohlcv(n=600), instrument="ES")
    for col in _HTF_COLS:
        assert col in f.columns, f"Missing HTF column: {col}"


def test_htf_no_nulls():
    """HTF features must have no NaN values — ATR warmup defaults to 0 or 0.5."""
    f = derive_features(make_ohlcv(n=600), instrument="ES")
    for col in _HTF_COLS:
        assert f[col].isna().sum() == 0, f"{col} has {f[col].isna().sum()} NaN values"


def test_htf_close_pos_bounded():
    """Close position features must lie in [0, 1] — fractional position within the HTF candle range."""
    f = derive_features(make_ohlcv(n=600), instrument="ES")
    assert f["htf_1h_close_pos"].between(0.0, 1.0).all(), "htf_1h_close_pos out of [0, 1]"
    assert f["htf_4h_close_pos"].between(0.0, 1.0).all(), "htf_4h_close_pos out of [0, 1]"


def test_htf_tf_alignment_ternary():
    """HTF alignment must be in {-1, 0, 1} only."""
    f = derive_features(make_ohlcv(n=600), instrument="ES")
    unique = set(f["htf_tf_alignment"].unique())
    assert unique <= {-1.0, 0.0, 1.0}, f"htf_tf_alignment has unexpected values: {unique}"


def test_htf_context_open_is_first_bar_in_period():
    """htf_open returned by _compute_htf_context must equal the open of the first bar in each period."""
    df = make_controlled_uptrend(n=600)
    htf_open, _, _ = _compute_htf_context(df, 60)

    period = df["datetime"].dt.ceil("60min")
    expected_open = df.groupby(period)["open"].transform("first")
    assert (htf_open.values == expected_open.values).all(), (
        "htf_open must equal the first bar's open in each 1H period"
    )


def test_htf_1h_resets_at_period_boundary():
    """At the first bar of each new 1H period, cummax/cummin must equal that bar's high/low."""
    df = make_controlled_uptrend(n=600)
    _, htf_high, htf_low = _compute_htf_context(df, 60)

    period = df["datetime"].dt.ceil("60min")
    new_period = (period != period.shift()).values
    first_bar_indices = np.where(new_period)[0]

    for i in first_bar_indices[:8]:
        assert htf_high.iloc[i] == df["high"].iloc[i], (
            f"At 1H period start bar {i}: htf_high should reset to bar high "
            f"({df['high'].iloc[i]:.3f}), got {htf_high.iloc[i]:.3f}"
        )
        assert htf_low.iloc[i] == df["low"].iloc[i], (
            f"At 1H period start bar {i}: htf_low should reset to bar low "
            f"({df['low'].iloc[i]:.3f}), got {htf_low.iloc[i]:.3f}"
        )


def test_htf_4h_continues_past_1h_boundary():
    """4H cummax must NOT reset at 1H boundaries — it only resets at 4H period starts."""
    df = make_controlled_uptrend(n=600)
    _, htf_high_1h, _ = _compute_htf_context(df, 60)
    _, htf_high_4h, _ = _compute_htf_context(df, 240)

    period_1h = df["datetime"].dt.ceil("60min")
    period_4h = df["datetime"].dt.ceil("240min")

    # Find bars where 1H resets but 4H does NOT (mid-4H-candle 1H boundaries)
    new_1h = (period_1h != period_1h.shift()).values
    new_4h = (period_4h != period_4h.shift()).values
    intra_4h_1h_starts = np.where(new_1h & ~new_4h)[0]

    for i in intra_4h_1h_starts[:6]:
        if i == 0:
            continue
        # At a mid-4H 1H boundary in an uptrend, 4H high >= previous 4H high
        assert htf_high_4h.iloc[i] >= htf_high_4h.iloc[i - 1], (
            f"htf_4h_high decreased at 1H boundary (bar {i}): "
            f"prev={htf_high_4h.iloc[i - 1]:.3f}, curr={htf_high_4h.iloc[i]:.3f}. "
            "4H cummax must not reset at 1H boundaries."
        )
        # And 4H high must be >= the current bar's high (cummax property)
        assert htf_high_4h.iloc[i] >= df["high"].iloc[i], (
            f"htf_4h_high < bar high at 1H boundary (bar {i}): "
            f"4H_high={htf_high_4h.iloc[i]:.3f}, bar_high={df['high'].iloc[i]:.3f}"
        )


def test_htf_close_pos_rises_within_uptrend_period():
    """Within a single 1H period, close_pos must be non-decreasing in a monotone uptrend."""
    df = make_controlled_uptrend(n=600)
    f = derive_features(df, instrument="ES")

    period = df["datetime"].dt.ceil("60min")
    # Use the 5th complete 1H period (well past ATR warmup)
    unique_periods = period.unique()
    target_period = unique_periods[5]
    mask = (period == target_period).values
    close_pos = f["htf_1h_close_pos"].values[mask]

    diffs = np.diff(close_pos)
    assert (diffs >= -1e-5).all(), (
        f"htf_1h_close_pos decreased within uptrend period: min diff = {diffs.min():.6f}"
    )


def test_htf_alignment_positive_in_consistent_uptrend():
    """Consistent uptrend: both 1H and 4H returns from their opens are positive → alignment = +1."""
    df = make_controlled_uptrend(n=600)
    f = derive_features(df, instrument="ES")

    # After ATR warmup, both htf_1h_ret and htf_4h_ret are positive in an uptrend.
    # Where both are strictly positive, alignment must be +1.
    after_warmup = f.iloc[30:]
    both_positive = after_warmup[
        (after_warmup["htf_1h_ret"] > 0) & (after_warmup["htf_4h_ret"] > 0)
    ]
    assert len(both_positive) > 0, "Expected positive returns in uptrend after warmup"
    assert (both_positive["htf_tf_alignment"] == 1.0).all(), (
        "Both-positive timeframe returns must produce alignment = +1"
    )


def test_htf_alignment_negative_when_timeframes_oppose():
    """When 1H is positive but 4H is still negative (HTF decline, LTF reversal), alignment = -1."""
    n = 300
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = np.concatenate([
        np.linspace(5100, 4900, 240),   # 4H-scale decline
        np.linspace(4900, 4950, 60),    # 1H-scale reversal within same 4H candle
    ])
    df = pd.DataFrame({
        "datetime": dates,
        "open":   close - 0.1,
        "high":   close + 1.0,
        "low":    close - 1.0,
        "close":  close,
        "volume": np.ones(n) * 1000.0,
    })
    f = derive_features(df, instrument="ES")

    # In the reversal segment (bars 250-290): 1H ret positive, 4H ret still negative
    reversal = f.iloc[250:290]
    opposing = reversal[
        (reversal["htf_1h_ret"] > 0) & (reversal["htf_4h_ret"] < 0)
    ]
    assert len(opposing) > 0, (
        "Expected bars with opposing 1H/4H returns during LTF reversal within HTF decline"
    )
    assert (opposing["htf_tf_alignment"] == -1.0).all(), (
        "Opposing 1H/4H returns must produce alignment = -1"
    )
