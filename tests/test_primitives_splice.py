"""Roll-splice detection primitives (primitives/splice.py) — the DATA-HYGIENE suite.

Verifies the jump detector fires exactly on synthetic roll-splice bars (never on ordinary
volatility), that the prefix-count range queries are O(1)-correct against a brute-force
reference, and that window/outcome tainting matches the exact ranges the pivot strategy relies
on: [i-seq+1, i] for the input window, [i+1, min(i+1+vert, n-1)] for the outcome path.
"""
import numpy as np
import pytest

from futures_foundation.primitives.splice import (
    detect_splice_jumps, splice_prefix_counts, splice_taints_window, splice_taints_outcome)


def _series(n, jump_at=None, jump_mult=20.0, atr_val=1.0, seed=0):
    """A flat-ATR random-walk close series with an optional single splice jump inserted at
    `jump_at` (bar-to-bar move = jump_mult * atr_val, far beyond normal walk noise)."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))       # normal noise << 1 ATR
    atr = np.full(n, atr_val)
    if jump_at is not None:
        close[jump_at:] += jump_mult * atr_val              # shift everything from jump_at on
    return close, atr


# ---------------------------------------------------------------- detect_splice_jumps
def test_no_jump_in_clean_series():
    close, atr = _series(200)
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert jump.dtype == bool and len(jump) == 200
    assert not jump.any()


def test_single_jump_detected_at_exact_bar():
    close, atr = _series(200, jump_at=100, jump_mult=20.0)
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert jump.sum() == 1
    assert jump[100] and not jump[99] and not jump[101]


def test_bar_zero_never_flagged():
    """No prior bar to jump from -- bar 0 is always False regardless of its value."""
    close, atr = _series(50)
    close[0] = 1e9                                          # absurd value, no diff computed
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert not jump[0]


def test_ordinary_volatility_does_not_trigger():
    """A bar-to-bar move of a few ATR (real, tradeable volatility) must NOT be flagged as a
    splice -- the threshold must separate genuine vol from roll artifacts."""
    close, atr = _series(200, jump_at=100, jump_mult=3.0)   # 3 ATR move -- normal for a fast market
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert not jump.any()


def test_threshold_is_atr_relative_not_absolute():
    """The SAME absolute price jump is a splice at low ATR but ordinary vol at high ATR --
    proves the detector scales with the instrument/regime, not a fixed price delta."""
    n = 50
    close_lowatr = np.concatenate([np.full(25, 100.0), np.full(25, 105.0)])   # jump = 5
    atr_lowatr = np.full(n, 0.5)                                              # 5 / 0.5 = 10 ATR
    close_hiatr = close_lowatr.copy()
    atr_hiatr = np.full(n, 5.0)                                              # 5 / 5.0 = 1 ATR
    assert detect_splice_jumps(close_lowatr, atr_lowatr, jump_atr=8.0)[25]
    assert not detect_splice_jumps(close_hiatr, atr_hiatr, jump_atr=8.0)[25]


def test_zero_or_negative_atr_never_flags_warmup():
    """atr<=0 (indicator warmup) must never produce a flagged bar via division blow-up."""
    close, atr = _series(50, jump_at=20, jump_mult=20.0)
    atr[:20] = 0.0                                          # warmup window -> atr<=0
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert jump[20]                                         # the real jump still fires...
    assert not jump[:20].any()                              # ...and warmup bars never do


def test_multiple_jumps_all_detected():
    close, atr = _series(300, jump_at=100, jump_mult=15.0, seed=1)
    close2, _ = _series(300, jump_at=200, jump_mult=15.0, seed=1)
    close = close.copy()
    close[200:] += (close2[200:] - close2[:100].mean())     # stack a 2nd jump at 200
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    assert jump[100] and jump[200]
    assert jump.sum() == 2


# ---------------------------------------------------------------- splice_prefix_counts
def test_prefix_counts_length_and_endpoints():
    close, atr = _series(100, jump_at=50, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    assert len(prefix) == 101                               # n+1
    assert prefix[0] == 0
    assert prefix[-1] == 1                                  # total jump count


def test_prefix_counts_match_bruteforce_cumsum():
    close, atr = _series(400, jump_at=150, jump_mult=12.0, seed=3)
    close2, _ = _series(400, jump_at=300, jump_mult=12.0, seed=7)
    close = close.copy(); close[300:] += (close2[300:] - close2[280:300].mean())
    jump = detect_splice_jumps(close, atr, jump_atr=8.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    brute = np.concatenate([[0], np.cumsum(jump)])
    np.testing.assert_array_equal(prefix, brute)


# ---------------------------------------------------------------- splice_taints_window / _outcome
def test_taints_window_true_when_jump_inside_range():
    close, atr = _series(200, jump_at=100, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    seq = 20
    assert splice_taints_window(prefix, i=105, seq=seq)      # window [86,105] contains bar 100
    assert not splice_taints_window(prefix, i=95, seq=seq)   # window [76,95] does not


def test_taints_window_exact_boundary_bars():
    """The jump bar sitting exactly at the window's first or last bar still counts (inclusive
    range) -- a candidate one bar away from clean must not be silently exempted."""
    close, atr = _series(200, jump_at=100, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    seq = 20
    assert splice_taints_window(prefix, i=100 + seq - 1, seq=seq)   # jump = the FIRST bar of window
    assert splice_taints_window(prefix, i=100, seq=seq)             # jump = the LAST bar of window
    assert not splice_taints_window(prefix, i=100 + seq, seq=seq)   # jump one bar before window starts


def test_taints_outcome_true_when_jump_inside_forward_path():
    close, atr = _series(200, jump_at=150, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    n = len(close)
    assert splice_taints_outcome(prefix, i=140, vert=20, n=n)       # [141,160] contains 150
    assert not splice_taints_outcome(prefix, i=160, vert=20, n=n)   # [161,180] does not


def test_taints_outcome_clamps_at_series_end():
    """i+1+vert past the series end must clamp to n-1, never index out of bounds, and still
    correctly detect a jump sitting right at the tail."""
    close, atr = _series(100, jump_at=95, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    n = len(close)
    assert splice_taints_outcome(prefix, i=90, vert=50, n=n)        # vert overshoots past n-1
    assert not splice_taints_outcome(prefix, i=96, vert=50, n=n)    # entry AFTER the only jump


def test_taints_window_clamps_at_series_start():
    """i-seq+1 going negative (early-series candidate) must clamp to bar 0, not wrap/error."""
    close, atr = _series(50, jump_at=2, jump_mult=20.0)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    assert splice_taints_window(prefix, i=5, seq=50)                # window clamps to [0,5], has jump@2
    assert not splice_taints_window(prefix, i=40, seq=5)            # window [36,40], clean


def test_clean_series_never_taints_anything():
    close, atr = _series(300, seed=42)
    prefix = splice_prefix_counts(close, atr, jump_atr=8.0)
    n = len(close)
    for i in range(130, 170, 7):
        assert not splice_taints_window(prefix, i=i, seq=128)
        assert not splice_taints_outcome(prefix, i=i, vert=120, n=n)
