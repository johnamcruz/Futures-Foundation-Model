"""Supertrend-flip zigzag pivots (detect_supertrend_zigzag_pivots) — the ported [JL] zigzag core.

The trigger-variant contract: same schema as detect_atr_zigzag_pivots (drop-in for every consumer),
strictly causal (confirm = the flip bar, extreme lookback entirely in the past), pivots alternate
long/short, and the confirmed extreme is byte-exact the leg's max/min.
"""
import numpy as np

from futures_foundation.primitives.detection import (
    detect_atr_zigzag_pivots, detect_supertrend_zigzag_pivots)


def _waves(n_legs=6, leg=40, amp=30.0, seed=0):
    """Alternating up/down ramps with mild noise — unambiguous swings, flips guaranteed."""
    rng = np.random.default_rng(seed)
    px = [100.0]
    for k in range(n_legs):
        d = 1.0 if k % 2 == 0 else -1.0
        for _ in range(leg):
            px.append(px[-1] + d * (amp / leg) + rng.normal(0, 0.05))
    c = np.array(px)
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + 0.1
    l = np.minimum(o, c) - 0.1
    return o, h, l, c


def test_schema_matches_atr_zigzag():
    o, h, l, c = _waves()
    ours = detect_atr_zigzag_pivots(o, h, l, c)
    st = detect_supertrend_zigzag_pivots(o, h, l, c)
    assert len(st) >= 3                                    # the waves produce real flips
    assert set(st[0]) == set(ours[0])                      # identical dict keys = drop-in trigger
    for p in st:
        assert p['direction'] in (-1, 1)
        assert isinstance(p['confirm'], int) and isinstance(p['origin'], int)
        assert p['R'] >= 0


def test_causal_confirm_and_past_extreme():
    o, h, l, c = _waves()
    n = len(c)
    for p in detect_supertrend_zigzag_pivots(o, h, l, c):
        assert p['confirm'] + 1 < n                        # a causal entry bar exists
        assert p['origin'] <= p['leg_end'] <= p['confirm']  # extremes are IN THE PAST of the confirm
        assert p['leg_end'] < p['confirm'] or p['leg_end'] == p['confirm']


def test_directions_alternate_and_match_the_leg():
    o, h, l, c = _waves()
    piv = detect_supertrend_zigzag_pivots(o, h, l, c)
    dirs = [p['direction'] for p in piv]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))     # zigzag: long/short alternate
    for p in piv:
        # long entries confirm a pivot LOW (leg_end is the segment's lowest low), shorts a HIGH
        seg = slice(p['origin'], p['confirm'] + 1)
        if p['direction'] == 1:
            assert l[p['leg_end']] == l[seg].min()
        else:
            assert h[p['leg_end']] == h[seg].max()


def test_confirmed_extreme_is_exact_leg_extreme():
    o, h, l, c = _waves(seed=3)
    piv = detect_supertrend_zigzag_pivots(o, h, l, c)
    for a, b in zip(piv, piv[1:]):
        # between consecutive pivots, the later pivot's extreme covers (prev confirm, confirm]
        seg = slice(a['confirm'] + 1, b['confirm'] + 1)
        if b['direction'] == 1:
            assert l[b['leg_end']] <= l[seg].min() + 1e-9
        else:
            assert h[b['leg_end']] >= h[seg].max() - 1e-9


def test_factor_controls_flip_count():
    # the sweepable knob: a tighter band (smaller factor) flips more often -> more pivots
    o, h, l, c = _waves(n_legs=8, seed=1)
    tight = detect_supertrend_zigzag_pivots(o, h, l, c, factor=1.5)
    loose = detect_supertrend_zigzag_pivots(o, h, l, c, factor=4.0)
    assert len(tight) >= len(loose)


def test_short_series_and_flat_are_safe():
    z = np.full(50, 100.0)
    assert detect_supertrend_zigzag_pivots(z, z + 0.1, z - 0.1, z) == []   # no flips on flat
    assert detect_supertrend_zigzag_pivots(z[:2], z[:2], z[:2], z[:2]) == []


def test_deterministic():
    o, h, l, c = _waves(seed=7)
    a = detect_supertrend_zigzag_pivots(o, h, l, c)
    b = detect_supertrend_zigzag_pivots(o, h, l, c)
    assert a == b
