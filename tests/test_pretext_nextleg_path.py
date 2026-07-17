"""Stage-2.7 (nextleg_path) — the PATH target's contract.

r1 = deepest pullback within the newborn leg / that leg's OWN extent. The whole design claim is
that it is PURE CANDLE STRUCTURE: unitless, scale-free, direction-symmetric, no ATR/cost/R. If any
of those break it stops being SSL and becomes a strategy label smuggled into the pretext — the
shape that lost in turn-electra. These tests are the guard.

Also asserts 2.7 is ADDITIVE: 2.6's t1/t2 must come out byte-identical, because nextleg is the
graduated backbone and the 2.6-vs-2.7 A/B is only honest if 2.6 is untouched.

The path math is torch-free on purpose (pretext/nextleg_path.py) so these run in the normal suite;
target assembly needs the fractal detector from a _torch module, so it gates on
CHRONOS_TORCH_TESTS=1 like the other SSL trainers (libomp isolation).
"""
import os

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.nextleg_path import NextLegPathTask, leg_retrace

torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1 (libomp isolation)')

# a leg that peaks at 105 (best progress 5 of a 10-point leg), falls back to 100, then completes
ROUGH_H = np.array([100., 105., 105., 103., 108., 110.])
ROUGH_L = np.array([100., 103., 100., 100., 104., 109.])


def test_retrace_clean_leg_is_near_zero():
    h = np.array([100., 102., 104., 106., 108., 110.])
    l = np.array([100., 101., 103., 105., 107., 109.])
    assert leg_retrace(h, l, 0, 5, d=1) < 0.15


def test_retrace_measures_giveback_fraction():
    assert leg_retrace(ROUGH_H, ROUGH_L, 0, 5, d=1) == pytest.approx(0.5)


def test_retrace_is_direction_symmetric():
    up = leg_retrace(ROUGH_H, ROUGH_L, 0, 5, d=1)
    dn = leg_retrace(200. - ROUGH_L, 200. - ROUGH_H, 0, 5, d=-1)      # mirrored down-leg
    assert dn == pytest.approx(up)


def test_retrace_is_scale_free():
    """THE design claim: a ratio of two distances from the SAME leg, so scaling prices changes
    nothing. This is what makes it instrument- and TF-agnostic WITHOUT touching ATR."""
    base = leg_retrace(ROUGH_H, ROUGH_L, 0, 5, d=1)
    assert leg_retrace(ROUGH_H * 10, ROUGH_L * 10, 0, 5, d=1) == pytest.approx(base)
    assert leg_retrace(ROUGH_H * 0.1, ROUGH_L * 0.1, 0, 5, d=1) == pytest.approx(base)


def test_retrace_degenerate_leg_is_nan_not_zero():
    """No extent -> unresolved. MUST be NaN so the anchor is dropped; 0.0 would read as 'clean'."""
    flat = np.full(4, 100.)
    assert np.isnan(leg_retrace(flat, flat, 0, 3, d=1))
    assert np.isnan(leg_retrace(flat, flat, 0, 0, d=1))               # single bar


def test_retrace_capped():
    h = np.array([100., 150., 100., 101.])
    l = np.array([100., 100., 1., 100.])                              # violent giveback, tiny extent
    assert leg_retrace(h, l, 0, 3, d=1, cap=2.0) <= 2.0


def test_task_registered_and_2_6_untouched():
    assert 'nextleg_path' in PRETEXTS
    t = get_pretext('nextleg_path')
    assert isinstance(t, NextLegPathTask)
    assert t.trainer == 'train_ssl_nextleg_path'
    assert get_pretext('nextleg').trainer == 'train_ssl_nextleg'      # 2.6 still resolves to 2.6
    cfg = {'context_lengths': (64, 100, 150, 200), 'leg_cap': 256}
    assert t.reserve(cfg) == get_pretext('nextleg').reserve(cfg)      # inherited, not re-derived


def _rw(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    return np.stack([c, c + np.abs(rng.normal(0, .3, n)), c - np.abs(rng.normal(0, .3, n)),
                     c, np.abs(rng.normal(1e3, 50, n))], 1).astype(np.float32)


@torch_test
def test_2_6_targets_are_byte_identical():
    """2.7 is ADDITIVE. nextleg is the shipped backbone; if t1/t2 move, the A/B is meaningless."""
    from futures_foundation.finetune.pretext._torch.nextleg import _leg_targets as t26
    from futures_foundation.finetune.pretext._torch.nextleg_path import _leg_path_targets
    big = _rw()
    c26, g26, ok26 = t26(big, 2, 256)
    c27, g27, ok27 = _leg_path_targets(big, 2, 256)
    assert np.array_equal(c26, c27)
    assert np.allclose(g26, g27[:, :2])
    assert ok27.sum() <= ok26.sum()                       # unresolved r1 only ever DROPS anchors


@torch_test
def test_path_targets_shape_and_range():
    from futures_foundation.finetune.pretext._torch.nextleg_path import _leg_path_targets
    _c, g, ok = _leg_path_targets(_rw(), 2, 256)
    assert g.shape[1] == 3
    r1 = g[ok][:, 2]
    assert len(r1) > 100
    assert (r1 >= 0).all() and (r1 <= 2.0).all() and np.isfinite(r1).all()


@torch_test
def test_trainer_signature_matches_2_6_and_swallows_cfg():
    """The orchestrator hands EVERY task the whole shared cfg (base.py: `**kw` from cfg), so a
    trainer must NAME what it wants and swallow the rest in **_ignore. A **kw passthrough forwards
    keys like `seq` down to BaseTrainer -> TypeError at Colab runtime, ~2 min in. Caught in prod;
    this is the guard. 2.7 must accept exactly 2.6's params + the retrace lever."""
    import inspect
    from futures_foundation.finetune._ssl_torch import train_ssl_nextleg, train_ssl_nextleg_path
    s26 = inspect.signature(train_ssl_nextleg).parameters
    s27 = inspect.signature(train_ssl_nextleg_path).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in s27.values()), \
        'train_ssl_nextleg_path must end in **_ignore to swallow the shared cfg'
    missing = set(s26) - set(s27)
    assert not missing, f'2.7 dropped 2.6 params (cfg keys would fall into _ignore): {missing}'
    assert {'retrace_w', 'retrace_cap'} <= set(s27)
    # the exact key that broke the Colab run
    assert 'seq' not in s27 and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in s27.values())
