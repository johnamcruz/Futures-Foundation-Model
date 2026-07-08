"""Correlated-sibling pairing primitives (primitives/pairs.py) — the CAUSALITY suite.

Cross-STREAM alignment is the one place a leak can hide in the sibling experiment (two clocks,
gaps, halts). Same rigor as the aggregation-window primitive got: future bars must be physically
unreachable, gaps must fall back to earlier bars, staleness must drop the anchor, and the window
content must be byte-exact.
"""
import numpy as np
import pytest

from futures_foundation.primitives.pairs import (
    DEFAULT_SIBLINGS, parse_siblings, asof_sibling_index, sibling_ohlcv_window)


def _sib(ts):
    """Sibling stream whose values ENCODE their index (content checks read the values)."""
    n = len(ts)
    i = np.arange(n, dtype=float)
    return {'ts': np.asarray(ts), 'o': i, 'h': i + 0.5, 'l': i - 0.5, 'c': i + 0.25,
            'v': np.full(n, 7.0)}


# ---------------------------------------------------------------- the pair map
def test_default_siblings_total_and_sane():
    assert set(DEFAULT_SIBLINGS) == {'ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN'}
    for k, v in DEFAULT_SIBLINGS.items():
        assert v in DEFAULT_SIBLINGS and v != k            # partner is in-universe, never itself
    assert DEFAULT_SIBLINGS['ES'] == 'NQ' and DEFAULT_SIBLINGS['NQ'] == 'ES'
    assert DEFAULT_SIBLINGS['GC'] == 'SI' and DEFAULT_SIBLINGS['ZB'] == 'ZN'


def test_parse_siblings_off_on_and_overrides():
    assert parse_siblings('') is None and parse_siblings('0') is None
    assert parse_siblings('1') == DEFAULT_SIBLINGS
    m = parse_siblings('ES:YM,CL:')                        # repoint ES, EXCLUDE CL
    assert m['ES'] == 'YM' and 'CL' not in m
    assert m['GC'] == 'SI'                                 # untouched pairs keep the default


# ---------------------------------------------------------------- asof alignment (causality)
def test_asof_picks_the_simultaneous_bar():
    ts = np.arange(0, 100, 3)                              # same 3-unit grid as the own stream
    assert asof_sibling_index(30, ts) == 10                # equal stamp = the simultaneous bar


def test_asof_never_returns_a_future_bar():
    ts = np.arange(0, 100, 3)
    for own_t in (29, 30, 31, 32):                         # sweep around a boundary
        j = asof_sibling_index(own_t, ts)
        assert ts[j] <= own_t                              # THE causality invariant


def test_asof_gap_falls_back_to_earlier_bar():
    ts = np.array([0, 3, 6, 15, 18])                       # bars 9 and 12 missing (halt)
    assert asof_sibling_index(12, ts) == 2                 # falls back to the 6-bar, not forward to 15


def test_asof_staleness_guard_drops_the_anchor():
    ts = np.array([0, 3, 6, 15, 18])
    assert asof_sibling_index(12, ts, max_gap=9) == 2      # 12-6=6 <= 9: ok
    assert asof_sibling_index(12, ts, max_gap=5) == -1     # 6 > 5: STALE -> dropped, never used


def test_asof_before_first_bar_is_dropped():
    assert asof_sibling_index(-1, np.arange(0, 30, 3)) == -1


# ---------------------------------------------------------------- the sibling window
def test_window_content_ends_at_aligned_bar_and_is_exact():
    sib = _sib(np.arange(0, 300, 3))
    w = sibling_ohlcv_window(60, sib, seq=8)               # aligned bar = index 20
    assert w.shape == (5, 8) and w.dtype == np.float32
    assert w[0, -1] == 20.0 and w[0, 0] == 13.0            # opens encode indices 13..20
    assert w[1, -1] == 20.5 and w[2, -1] == 19.5 and w[4, -1] == 7.0


def test_window_never_contains_future_values():
    sib = _sib(np.arange(0, 300, 3))
    w = sibling_ohlcv_window(61, sib, seq=8)               # own ts between sibling stamps
    assert w[0].max() == 20.0                              # nothing past the aligned (<=61) bar


def test_window_insufficient_history_returns_none():
    sib = _sib(np.arange(0, 30, 3))                        # only 10 bars
    assert sibling_ohlcv_window(9, sib, seq=8) is None     # aligned idx 3 -> 4 bars < seq
    assert sibling_ohlcv_window(27, sib, seq=8) is not None


def test_window_stale_returns_none():
    sib = _sib(np.array([0, 3, 6, 15, 18, 21, 24, 27, 30, 33]))
    assert sibling_ohlcv_window(12, sib, seq=3, max_gap=5) is None   # gap-stale -> dropped


def test_window_nan_safe():
    sib = _sib(np.arange(0, 300, 3))
    sib['v'] = sib['v'].copy(); sib['v'][15] = np.nan
    w = sibling_ohlcv_window(60, sib, seq=8)
    assert np.isfinite(w).all()
