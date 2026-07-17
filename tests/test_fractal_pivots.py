"""Williams-fractal pivots (detect_fractal_pivots) — the time-confirmation trigger variant.

Contract: the shared pivot schema, strictly causal (confirm = extreme+k,
entry bar exists), extremes unique in their 2k+1 window, byte-exact detection.
"""
import numpy as np

from futures_foundation.primitives.detection import (
    detect_fractal_pivots)


def _v(center=30, n=64, depth=5.0):
    """A clean V: strictly falling into `center`, strictly rising after — one unambiguous low."""
    t = np.arange(n, dtype=float)
    c = 100 + np.abs(t - center) * (depth / center)
    h = c + 0.2
    l = c - 0.2
    return h, l


def test_finds_the_v_bottom():
    h, l = _v(center=30)
    piv = detect_fractal_pivots(h, l, k=2)
    lows = [p for p in piv if p['direction'] == 1]
    assert any(p['origin'] == 30 for p in lows)            # THE bottom is detected
    p = next(p for p in lows if p['origin'] == 30)
    assert p['confirm'] == 32                              # confirm = extreme + k
    assert p['leg_end'] == 30


def test_causal_confirm_and_unique_extreme():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 1, 500).cumsum()
    h = c + np.abs(rng.normal(0, 0.3, 500))
    l = c - np.abs(rng.normal(0, 0.3, 500))
    n = len(c)
    for p in detect_fractal_pivots(h, l, k=3):
        i, cf = p['origin'], p['confirm']
        assert cf == i + 3 and cf + 1 < n                  # time confirmation + causal entry exists
        if p['direction'] == 1:
            seg = l[i - 3:i + 4]
            assert l[i] == seg.min() and (seg == l[i]).sum() == 1   # unique lowest low
        else:
            seg = h[i - 3:i + 4]
            assert h[i] == seg.max() and (seg == h[i]).sum() == 1


def test_flat_series_has_no_pivots():
    z = np.full(100, 100.0)
    assert detect_fractal_pivots(z + 0.1, z - 0.1, k=2) == []   # ties are never pivots


def test_min_bars_apart_suppresses_clusters():
    rng = np.random.default_rng(2)
    c = 100 + rng.normal(0, 1, 600).cumsum()
    h = c + np.abs(rng.normal(0, 0.3, 600))
    l = c - np.abs(rng.normal(0, 0.3, 600))
    dense = detect_fractal_pivots(h, l, k=2)
    sparse = detect_fractal_pivots(h, l, k=2, min_bars_apart=10)
    assert len(sparse) < len(dense)
    for d in (1, -1):
        idx = [p['origin'] for p in sparse if p['direction'] == d]
        assert all(b - a >= 10 for a, b in zip(idx, idx[1:]))


def test_deterministic():
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 1, 300).cumsum()
    h, l = c + 0.3, c - 0.3
    assert detect_fractal_pivots(h, l, k=2) == detect_fractal_pivots(h, l, k=2)


# ---------------------------------------------------------------- fractal-zigzag hybrid
from futures_foundation.primitives.detection import detect_fractal_zigzag_pivots


def _rw(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    return o, h, l, c


def test_hybrid_alternates_and_respects_leg_rule():
    from futures_foundation.pipeline._primitives import compute_atr
    o, h, l, c = _rw()
    piv = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.25)
    assert len(piv) > 3
    dirs = [p['direction'] for p in piv]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))     # strict alternation
    atr = compute_atr(h, l, c, 20)
    for a, b in zip(piv, piv[1:]):
        pa = l[a['origin']] if a['direction'] == 1 else h[a['origin']]
        pb = l[b['origin']] if b['direction'] == 1 else h[b['origin']]
        assert abs(pb - pa) >= 1.25 * atr[b['origin']] - 1e-9   # significance rule


def test_hybrid_keep_first_no_hindsight():
    # two same-direction fractals, second deeper: the FIRST must be kept (live-tradeable rule);
    # swapping in the deeper one is the measured ~9-WR-pt lookahead.
    o, h, l, c = _rw(seed=6)
    piv = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.0)
    from futures_foundation.primitives.detection import detect_fractal_pivots
    raw = detect_fractal_pivots(h, l, k=2)
    kept = {p['origin'] for p in piv}
    # walk the causal state machine independently and confirm identical keeps
    from futures_foundation.pipeline._primitives import compute_atr
    atr = compute_atr(h, l, c, 20)
    last_d, last_px, expect = 0, None, set()
    for p in raw:
        i, d = p['origin'], p['direction']
        px = l[i] if d == 1 else h[i]
        if not (np.isfinite(atr[i]) and atr[i] > 0) or d == last_d:
            continue
        if last_px is not None and abs(px - last_px) < 1.0 * atr[i]:
            continue
        expect.add(i); last_d, last_px = d, px
    assert kept == expect


def test_hybrid_stricter_leg_fewer_pivots():
    o, h, l, c = _rw(seed=7)
    loose = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=0.5)
    tight = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=2.5)
    assert len(tight) < len(loose)


# ---------------------------------------------------------------- THE LOOKAHEAD PROOF
def test_truncation_invariance_no_lookahead():
    """THE SHOWSTOPPER TEST: a causal detector's decisions at bar t are identical whether or not
    the future exists. For random cut points t: pivots with confirm <= t computed on the FULL
    series must equal those computed on data[:t+1]. Any lookahead (future bars influencing the
    extreme choice, the alternation state, or the leg rule) breaks this equality."""
    from futures_foundation.pipeline._primitives import compute_atr
    o, h, l, c = _rw(n=3000, seed=11)

    def confirmed_by(data_end, t):
        piv = detect_fractal_zigzag_pivots(o[:data_end], h[:data_end], l[:data_end],
                                           c[:data_end], k=2, min_leg_atr=1.25)
        return [(p['confirm'], p['direction'], p['origin']) for p in piv if p['confirm'] <= t]

    rng = np.random.default_rng(0)
    for t in sorted(rng.integers(300, 2990, 25)):
        full = confirmed_by(3000, t)
        trunc = confirmed_by(t + 1, t)
        # the last pivot near the boundary may differ only via the cf+1<n guard (needs an entry
        # bar) — drop pivots confirmed exactly at the boundary from both sides before comparing
        full = [p for p in full if p[0] < t]
        trunc = [p for p in trunc if p[0] < t]
        assert full == trunc, f'LOOKAHEAD at cut {t}: full={full[-3:]} trunc={trunc[-3:]}'


def test_live_edge_emits_newest_bar_confirm():
    """live_edge=True: a pivot whose confirm IS the newest bar is emitted (bar-by-bar consumers
    need gate[-1] to be able to fire); default (training) drops it — everything else identical."""
    h, l = _v(center=30, n=33)                             # extreme@30, confirm=32 = last bar
    assert not any(p['confirm'] == 32 for p in detect_fractal_pivots(h, l, k=2))
    live = detect_fractal_pivots(h, l, k=2, live_edge=True)
    p = next(p for p in live if p['confirm'] == 32)
    assert p['origin'] == 30 and p['direction'] == 1
    # away from the edge the two modes are byte-identical
    o, hh, ll, c = _rw(seed=9)
    train = detect_fractal_zigzag_pivots(o, hh, ll, c, k=2, min_leg_atr=1.25)
    live = detect_fractal_zigzag_pivots(o, hh, ll, c, k=2, min_leg_atr=1.25, live_edge=True)
    assert [p for p in live if p['confirm'] + 1 < len(c)] == train


def test_truncation_invariance_live_edge():
    """The live consumer's proof: at every cut t, live_edge=True on data[:t+1] must emit exactly
    the full-series pivots with confirm <= t — INCLUDING one confirming on bar t itself. This is
    the no-drift guarantee for the bot's bar-by-bar port."""
    o, h, l, c = _rw(n=3000, seed=13)
    full = detect_fractal_zigzag_pivots(o, h, l, c, k=2, min_leg_atr=1.25, live_edge=True)
    rng = np.random.default_rng(2)
    for t in sorted(rng.integers(300, 2990, 25)):
        want = [(p['confirm'], p['direction'], p['origin']) for p in full if p['confirm'] <= t]
        piv = detect_fractal_zigzag_pivots(o[:t + 1], h[:t + 1], l[:t + 1], c[:t + 1],
                                           k=2, min_leg_atr=1.25, live_edge=True)
        got = [(p['confirm'], p['direction'], p['origin']) for p in piv]
        assert got == want, f'LIVE-EDGE DRIFT at cut {t}'


def test_truncation_invariance_pure_fractal():
    """Same proof for the raw fractal detector."""
    o, h, l, c = _rw(n=2000, seed=12)

    def confirmed_by(data_end, t):
        piv = detect_fractal_pivots(h[:data_end], l[:data_end], k=3)
        return [(p['confirm'], p['direction'], p['origin']) for p in piv if p['confirm'] < t]

    rng = np.random.default_rng(1)
    for t in sorted(rng.integers(200, 1990, 15)):
        assert confirmed_by(2000, t) == confirmed_by(t + 1, t), f'LOOKAHEAD at cut {t}'
