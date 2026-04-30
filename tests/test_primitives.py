"""Unit tests for futures_foundation.primitives."""

import numpy as np
import pandas as pd
import pytest

from futures_foundation.primitives import (
    compute_atr,
    compute_supertrend,
    compute_supertrend_htf,
    compute_ema,
    compute_rsi,
    rolling_mean,
    rolling_atr_percentile,
    detect_pivots,
    apply_rr_barriers,
    best_rr_hit,
    session_mask,
    session_end_mask,
    compute_vwap,
    detect_cisd_signals,
    compute_ote_zones,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flat_bars(n=100, price=100.0, spread=1.0):
    """Returns h, l, c arrays for flat price with fixed spread."""
    c = np.full(n, price)
    h = c + spread
    l = c - spread
    return h, l, c


def _trending_bars(n=100, start=100.0, step=1.0):
    """Returns h, l, c arrays for a clean uptrend."""
    c = np.arange(n, dtype=np.float64) * step + start
    h = c + 0.5
    l = c - 0.5
    return h, l, c


def _ny_index(n=100, start='2024-01-02 09:30', freq='5min'):
    return pd.date_range(start, periods=n, freq=freq, tz='America/New_York')


# ──────────────────────────────────────────────────────────────────────────────
# compute_atr
# ──────────────────────────────────────────────────────────────────────────────

def test_atr_all_positive():
    h, l, c = _flat_bars(50)
    atr = compute_atr(h, l, c, period=14)
    assert np.all(atr > 0)


def test_atr_length_preserved():
    h, l, c = _flat_bars(50)
    atr = compute_atr(h, l, c, period=14)
    assert len(atr) == 50


def test_atr_flat_bars_constant():
    n = 100
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    atr = compute_atr(h, l, c, period=14)
    # TR = 2.0 for every bar → ATR converges to 2.0
    assert abs(atr[-1] - 2.0) < 0.01


def test_atr_wilder_smoothing_second_bar():
    h = np.array([101.0, 102.0])
    l = np.array([99.0,  100.0])
    c = np.array([100.0, 101.0])
    atr = compute_atr(h, l, c, period=14)
    # atr[0] = h[0]-l[0] = 2.0
    # TR[1] = max(h[1]-l[1], |h[1]-c[0]|, |l[1]-c[0]|) = max(2,1,1) = 2
    # atr[1] = (13*2 + 2) / 14 = 2.0
    assert abs(atr[1] - 2.0) < 1e-9


def test_atr_returns_float64():
    h, l, c = _flat_bars(20)
    assert compute_atr(h, l, c, 5).dtype == np.float64


# ──────────────────────────────────────────────────────────────────────────────
# compute_supertrend
# ──────────────────────────────────────────────────────────────────────────────

def test_supertrend_direction_only_pm1():
    h, l, c = _flat_bars(100)
    d, _, _ = compute_supertrend(h, l, c, period=10, mult=2.0)
    assert set(d).issubset({1, -1})


def test_supertrend_length_preserved():
    h, l, c = _flat_bars(50)
    d, st, atr = compute_supertrend(h, l, c, 10, 2.0)
    assert len(d) == len(st) == len(atr) == 50


def test_supertrend_starts_bull():
    h, l, c = _flat_bars(20)
    d, _, _ = compute_supertrend(h, l, c, 10, 2.0)
    assert d[0] == 1


def test_supertrend_uptrend_stays_bull():
    h, l, c = _trending_bars(n=200, step=2.0)
    d, _, _ = compute_supertrend(h, l, c, period=10, mult=3.0)
    # After the first few warmup bars, a strong uptrend should stay bullish
    assert np.all(d[20:] == 1)


def test_supertrend_downtrend_goes_bear():
    h, l, c = _trending_bars(n=200, start=500.0, step=-2.0)
    d, _, _ = compute_supertrend(h, l, c, period=10, mult=3.0)
    assert np.all(d[20:] == -1)


def test_supertrend_st_line_on_correct_side():
    h, l, c = _trending_bars(200, step=1.0)
    d, st, atr = compute_supertrend(h, l, c, 10, 2.0)
    bull_mask = d == 1
    bear_mask = d == -1
    if bull_mask.any():
        assert np.all(st[bull_mask] <= h[bull_mask])
    if bear_mask.any():
        assert np.all(st[bear_mask] >= l[bear_mask])


def test_supertrend_flip_detectable():
    # Flat then sharp drop — should flip bear
    c = np.concatenate([np.full(50, 100.0), np.linspace(100, 50, 50)])
    h = c + 0.5
    l = c - 0.5
    d, _, _ = compute_supertrend(h, l, c, period=5, mult=1.0)
    assert -1 in d[50:]


# ──────────────────────────────────────────────────────────────────────────────
# compute_supertrend_htf
# ──────────────────────────────────────────────────────────────────────────────

def test_supertrend_htf_length_matches_input():
    idx = _ny_index(n=200)
    c   = np.linspace(100, 150, 200)
    h   = c + 1.0
    l   = c - 1.0
    v   = np.ones(200)
    df  = pd.DataFrame({'open': c, 'high': h, 'low': l, 'close': c, 'volume': v}, index=idx)
    result = compute_supertrend_htf(df, period=10, mult=3.0, tf='1h')
    assert len(result) == 200


def test_supertrend_htf_values_pm1():
    idx = _ny_index(n=200)
    c   = np.linspace(100, 150, 200)
    h   = c + 1.0
    l   = c - 1.0
    df  = pd.DataFrame({'open': c, 'high': h, 'low': l, 'close': c, 'volume': np.ones(200)}, index=idx)
    result = compute_supertrend_htf(df, 10, 3.0, '1h')
    assert set(result).issubset({1, -1})


# ──────────────────────────────────────────────────────────────────────────────
# compute_ema
# ──────────────────────────────────────────────────────────────────────────────

def test_ema_length_preserved():
    arr = np.random.randn(100)
    assert len(compute_ema(arr, 10)) == 100


def test_ema_seeds_at_first_value():
    arr = np.array([5.0, 6.0, 7.0])
    ema = compute_ema(arr, 2)
    assert ema[0] == 5.0


def test_ema_constant_array_returns_constant():
    arr = np.full(50, 7.0)
    ema = compute_ema(arr, 10)
    assert np.allclose(ema, 7.0)


def test_ema_faster_period_reacts_quicker():
    # After a jump, faster EMA should be closer to new value
    arr = np.concatenate([np.full(50, 100.0), np.full(50, 200.0)])
    ema_fast = compute_ema(arr, 5)
    ema_slow = compute_ema(arr, 20)
    assert ema_fast[-1] > ema_slow[-1]


def test_ema_returns_float64():
    arr = np.ones(20)
    assert compute_ema(arr, 5).dtype == np.float64


# ──────────────────────────────────────────────────────────────────────────────
# compute_rsi
# ──────────────────────────────────────────────────────────────────────────────

def test_rsi_length_preserved():
    c = np.random.randn(100).cumsum() + 100
    assert len(compute_rsi(c, 14)) == 100


def test_rsi_range_0_to_100():
    c = np.random.randn(200).cumsum() + 100
    rsi = compute_rsi(c, 14)
    valid = rsi[~np.isnan(rsi)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_rsi_first_period_nans():
    c = np.arange(1, 101, dtype=np.float64)
    rsi = compute_rsi(c, 14)
    assert np.all(np.isnan(rsi[:14]))
    assert not np.isnan(rsi[14])


def test_rsi_all_up_returns_100():
    c = np.arange(1, 101, dtype=np.float64)
    rsi = compute_rsi(c, 14)
    assert rsi[-1] == 100.0


def test_rsi_all_down_returns_0():
    c = np.arange(100, 0, -1, dtype=np.float64)
    rsi = compute_rsi(c, 14)
    assert rsi[-1] == 0.0


def test_rsi_short_array_returns_nans():
    c = np.array([100.0, 101.0, 102.0])
    rsi = compute_rsi(c, 14)
    assert np.all(np.isnan(rsi))


# ──────────────────────────────────────────────────────────────────────────────
# rolling_mean
# ──────────────────────────────────────────────────────────────────────────────

def test_rolling_mean_length_preserved():
    arr = np.arange(50, dtype=np.float64)
    assert len(rolling_mean(arr, 5)) == 50


def test_rolling_mean_exact_values():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_mean(arr, 3)
    assert abs(out[2] - 2.0) < 1e-9   # mean(1,2,3)
    assert abs(out[3] - 3.0) < 1e-9   # mean(2,3,4)
    assert abs(out[4] - 4.0) < 1e-9   # mean(3,4,5)


def test_rolling_mean_expanding_start():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = rolling_mean(arr, 3)
    assert abs(out[0] - 1.0) < 1e-9   # only 1 element
    assert abs(out[1] - 1.5) < 1e-9   # mean(1,2)


def test_rolling_mean_constant_array():
    arr = np.full(30, 5.0)
    assert np.allclose(rolling_mean(arr, 7), 5.0)


# ──────────────────────────────────────────────────────────────────────────────
# rolling_atr_percentile
# ──────────────────────────────────────────────────────────────────────────────

def test_atr_percentile_range():
    h, l, c = _flat_bars(300)
    atr  = compute_atr(h, l, c, 14)
    rank = rolling_atr_percentile(atr, 200)
    assert np.all(rank >= 0) and np.all(rank <= 1)


def test_atr_percentile_length():
    h, l, c = _flat_bars(100)
    atr = compute_atr(h, l, c, 14)
    assert len(rolling_atr_percentile(atr, 50)) == 100


def test_atr_percentile_rising_atr_gives_high_rank():
    # Increasing ATR values: last bar should have rank near 1.0
    atr  = np.arange(1, 201, dtype=np.float64)
    rank = rolling_atr_percentile(atr, 100)
    assert rank[-1] > 0.95


def test_atr_percentile_first_bar_is_zero():
    atr  = np.ones(50)
    rank = rolling_atr_percentile(atr, 20)
    assert rank[0] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# detect_pivots
# ──────────────────────────────────────────────────────────────────────────────

def _pivot_pattern(n=50, period=3):
    """Price with a single clear pivot high at bar 10 and pivot low at bar 20."""
    h = np.full(n, 100.0)
    l = np.full(n, 99.0)
    h[10] = 110.0   # clear pivot high
    l[20] = 89.0    # clear pivot low
    return h, l


def test_detect_pivots_finds_high():
    h, l = _pivot_pattern()
    ph, _ = detect_pivots(h, l, period=3)
    assert 10 in ph


def test_detect_pivots_finds_low():
    h, l = _pivot_pattern()
    _, pl = detect_pivots(h, l, period=3)
    assert 20 in pl


def test_detect_pivots_empty_short_array():
    h = np.array([1.0, 2.0])
    l = np.array([0.5, 1.5])
    ph, pl = detect_pivots(h, l, period=3)
    assert len(ph) == 0 and len(pl) == 0


def test_detect_pivots_flat_no_unique_pivots():
    h = np.full(20, 100.0)
    l = np.full(20, 99.0)
    ph, pl = detect_pivots(h, l, period=3)
    assert len(ph) == 0 and len(pl) == 0


def test_detect_pivots_returns_bar_indices_not_window_offsets():
    h, l = _pivot_pattern(n=50, period=3)
    ph, _ = detect_pivots(h, l, period=3)
    # Pivot at bar 10 is confirmed at bar 10+3=13, but returned index is the pivot bar (10)
    assert 10 in ph


# ──────────────────────────────────────────────────────────────────────────────
# apply_rr_barriers
# ──────────────────────────────────────────────────────────────────────────────

def _make_bars_for_rr(n=50, entry_close=100.0, direction='long'):
    """Bars that guarantee a 3R hit for a long/short from bar 5."""
    c = np.full(n, entry_close)
    if direction == 'long':
        h = c + 5.0    # plenty of room to hit targets
        l = c - 0.1    # stop at -1, well below stop at 99.0
    else:
        h = c + 0.1
        l = c - 5.0
    return h, l, c


def test_rr_barriers_target_hit_long():
    h, l, c = _make_bars_for_rr(50, 100.0, 'long')
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=True,
                            entry_price=100.0, sl_price=99.0,
                            rr_targets=[1.0, 2.0, 3.0])
    assert res[2.0]['hit']
    assert res[2.0]['outcome'] == 'target_hit'
    assert res[2.0]['realized_rr'] == 2.0


def test_rr_barriers_stop_hit_long():
    n = 50
    c = np.full(n, 100.0)
    h = c + 0.5
    l = c - 2.0     # immediately breaches stop at 99.0
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=True,
                            entry_price=100.0, sl_price=99.0,
                            rr_targets=[1.0, 2.0])
    assert res[1.0]['outcome'] == 'stopped'
    assert res[1.0]['realized_rr'] == -1.0


def test_rr_barriers_target_hit_short():
    h, l, c = _make_bars_for_rr(50, 100.0, 'short')
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=False,
                            entry_price=100.0, sl_price=101.0,
                            rr_targets=[1.0, 2.0])
    assert res[2.0]['hit']
    assert res[2.0]['realized_rr'] == 2.0


def test_rr_barriers_session_end_exit():
    n = 20
    c = np.full(n, 100.0)
    h = c + 0.5
    l = c - 0.1
    is_session_end       = np.zeros(n, dtype=bool)
    is_session_end[10]   = True
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=True,
                            entry_price=100.0, sl_price=99.0,
                            rr_targets=[3.0],
                            is_session_end=is_session_end)
    assert res[3.0]['outcome'] == 'session_end'


def test_rr_barriers_lookahead_limits_scan():
    # Bars that could hit 3R but only if we look far enough forward.
    # h = entry + 0.5, so 3R target (entry + 3 * stop_dist) is unreachable in any bar.
    n = 50
    c = np.full(n, 100.0)
    h = c + 0.5    # never reaches 3R target (103.0 with sl_price=99.0)
    l = c - 0.1
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=True,
                            entry_price=100.0, sl_price=99.0,
                            rr_targets=[3.0], lookahead=3)
    assert res[3.0]['outcome'] == 'data_end'


def test_rr_barriers_invalid_stop_returns_invalid():
    h, l, c = _make_bars_for_rr(20)
    res = apply_rr_barriers(h, l, c, entry_idx=5, is_long=True,
                            entry_price=100.0, sl_price=100.0,   # zero stop dist
                            rr_targets=[1.0])
    assert res[1.0]['outcome'] == 'invalid'


def test_rr_barriers_data_end():
    h = np.array([100.5, 100.5, 100.5])
    l = np.array([99.9,  99.9,  99.9])
    c = np.array([100.0, 100.0, 100.0])
    res = apply_rr_barriers(h, l, c, entry_idx=0, is_long=True,
                            entry_price=100.0, sl_price=99.0,
                            rr_targets=[5.0])
    assert res[5.0]['outcome'] == 'data_end'


# ──────────────────────────────────────────────────────────────────────────────
# best_rr_hit
# ──────────────────────────────────────────────────────────────────────────────

def test_best_rr_hit_returns_highest():
    results = {
        1.0: {'hit': True,  'outcome': 'target_hit', 'realized_rr': 1.0},
        2.0: {'hit': True,  'outcome': 'target_hit', 'realized_rr': 2.0},
        3.0: {'hit': False, 'outcome': 'stopped',    'realized_rr': -1.0},
    }
    assert best_rr_hit(results) == 2.0


def test_best_rr_hit_none_hit_returns_zero():
    results = {
        1.0: {'hit': False, 'outcome': 'stopped', 'realized_rr': -1.0},
        2.0: {'hit': False, 'outcome': 'stopped', 'realized_rr': -1.0},
    }
    assert best_rr_hit(results) == 0.0


def test_best_rr_hit_respects_min_rr():
    results = {
        1.0: {'hit': True, 'outcome': 'target_hit', 'realized_rr': 1.0},
        2.0: {'hit': True, 'outcome': 'target_hit', 'realized_rr': 2.0},
    }
    assert best_rr_hit(results, min_rr=2.0) == 2.0
    assert best_rr_hit(results, min_rr=3.0) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# session_mask / session_end_mask
# ──────────────────────────────────────────────────────────────────────────────

def test_session_mask_inside():
    idx  = _ny_index(n=24, start='2024-01-02 07:00', freq='1h')
    mask = session_mask(idx, start_hour=7, end_hour=16)
    # 07:00–15:00 = 9 hours inside (07,08,...,15), 16:00 is outside
    assert mask[0]   # 07:00 inside
    assert not mask[9]  # 16:00 outside


def test_session_mask_outside_pre_market():
    idx  = _ny_index(n=4, start='2024-01-02 06:00', freq='1h')
    mask = session_mask(idx, start_hour=7, end_hour=16)
    assert not mask[0]   # 06:00 outside


def test_session_end_mask_marks_last_bar():
    idx = pd.date_range('2024-01-02 07:00', periods=24, freq='1h', tz='America/New_York')
    end = session_end_mask(idx, start_hour=7, end_hour=16)
    # Last bar inside session is 15:00 (hour 8 in the index, 0-indexed)
    assert end[8]    # 15:00 is last inside bar
    assert not end[9]  # 16:00 already outside


def test_session_end_mask_length():
    idx = _ny_index(30)
    assert len(session_end_mask(idx)) == 30


# ──────────────────────────────────────────────────────────────────────────────
# compute_vwap
# ──────────────────────────────────────────────────────────────────────────────

def test_vwap_length_preserved():
    n    = 50
    h, l, c = _flat_bars(n, 100.0)
    v    = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    assert len(compute_vwap(h, l, c, v, mask)) == n


def test_vwap_constant_price_equals_price():
    n    = 30
    h, l, c = _flat_bars(n, 100.0, spread=0.5)
    v    = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    vwap = compute_vwap(h, l, c, v, mask)
    # TP = (100.5 + 99.5 + 100) / 3 = 100.0
    assert np.allclose(vwap, 100.0)


def test_vwap_nan_before_first_session():
    n    = 30
    h, l, c = _flat_bars(n, 100.0)
    v    = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[10] = True   # session starts at bar 10
    vwap = compute_vwap(h, l, c, v, mask)
    assert np.all(np.isnan(vwap[:10]))
    assert not np.isnan(vwap[10])


def test_vwap_resets_on_new_session():
    n    = 30
    h    = np.full(n, 101.0)
    l    = np.full(n, 99.0)
    c    = np.full(n, 100.0)
    v    = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[0]  = True
    mask[15] = True   # new session at bar 15
    vwap = compute_vwap(h, l, c, v, mask)
    # Both sessions have constant price so VWAP stays at 100
    assert np.allclose(vwap[15:], 100.0)


# ──────────────────────────────────────────────────────────────────────────────
# detect_cisd_signals
# ──────────────────────────────────────────────────────────────────────────────

def _bearish_cisd_bars():
    """
    Craft a minimal bearish CISD sequence:
      bar 0: bull candle (potential seed)
      bar 1: bear candle (c < o) — creates a bearish potential
      bar 2: bull candle (c > o) — creates a potential zone at o[2]
      bar 3: strong bear bar — closes below o[2] with body+close_str filter
    """
    n = 20
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)

    # bar 1: bear candle → creates bullish potential (next bar can flip)
    o[1] = 100.0; c[1] = 98.0; h[1] = 100.5; l[1] = 97.5

    # bar 2: bull flip candle → creates bear potential at o[2]=100
    o[2] = 98.0; c[2] = 102.0; h[2] = 102.5; l[2] = 97.5

    # bars 3-5: more bear candles to build up sweep distance
    for i in range(3, 6):
        o[i] = 102.0; c[i] = 103.0; h[i] = 103.5; l[i] = 101.5

    # bar 6: strong bear displacement: sweeps well above then closes far below o[2]
    o[6] = 104.0; h[6] = 107.0; l[6] = 96.0; c[6] = 97.0
    return o, h, l, c


def test_cisd_signals_length():
    o, h, l, c = _bearish_cisd_bars()
    res = detect_cisd_signals(o, h, l, c)
    assert len(res['cisd_signal']) == len(o)


def test_cisd_signals_values_0_1_2():
    o, h, l, c = _bearish_cisd_bars()
    res = detect_cisd_signals(o, h, l, c)
    assert set(res['cisd_signal']).issubset({0, 1, 2})


def test_cisd_no_signal_on_flat_bars():
    n = 50
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    res = detect_cisd_signals(o, h, l, c)
    assert np.all(res['cisd_signal'] == 0)


def test_cisd_displacement_str_nonnegative():
    o, h, l, c = _bearish_cisd_bars()
    res = detect_cisd_signals(o, h, l, c)
    assert np.all(res['displacement_str'] >= 0)


def test_cisd_origin_level_nan_where_no_signal():
    o, h, l, c = _bearish_cisd_bars()
    res = detect_cisd_signals(o, h, l, c)
    no_signal = res['cisd_signal'] == 0
    assert np.all(np.isnan(res['origin_level'][no_signal]))


def test_cisd_tight_filters_suppress_signals():
    o, h, l, c = _bearish_cisd_bars()
    # With very strict body/close_str filters nothing passes
    res = detect_cisd_signals(o, h, l, c, body_ratio_min=0.99, close_str_min=0.99)
    assert np.all(res['cisd_signal'] == 0)


def test_cisd_expiry_removes_stale_potentials():
    o, h, l, c = _bearish_cisd_bars()
    # With expiry_bars=1 the potential expires before bar 6 can fire
    res = detect_cisd_signals(o, h, l, c, expiry_bars=1)
    assert np.all(res['cisd_signal'] == 0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_ote_zones
# ──────────────────────────────────────────────────────────────────────────────

def test_ote_zones_length():
    o, h, l, c = _bearish_cisd_bars()
    cisd = detect_cisd_signals(o, h, l, c)
    ft, fb = compute_ote_zones(cisd, h, l, c)
    assert len(ft) == len(fb) == len(c)


def test_ote_zones_nan_where_no_signal():
    n = 20
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.0)
    c = np.full(n, 100.0)
    cisd = detect_cisd_signals(o, h, l, c)
    ft, fb = compute_ote_zones(cisd, h, l, c)
    assert np.all(np.isnan(ft))
    assert np.all(np.isnan(fb))


def test_ote_zones_top_above_bot():
    o, h, l, c = _bearish_cisd_bars()
    cisd = detect_cisd_signals(o, h, l, c)
    ft, fb = compute_ote_zones(cisd, h, l, c)
    valid = ~np.isnan(ft)
    if valid.any():
        assert np.all(ft[valid] >= fb[valid])
