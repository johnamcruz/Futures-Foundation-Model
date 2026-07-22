"""Contracts for stage-2.8's future-only ordered path race."""
import os

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.nextleg_race import (
    NextLegRaceTask, RACE_LEVELS, ordered_adverse_curve)


torch_test = pytest.mark.skipif(
    os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='torch test — set CHRONOS_TORCH_TESTS=1')


def _curve(h, l, c=None, confirm=1, end=None, d=1):
    h, l = np.asarray(h, float), np.asarray(l, float)
    c = np.asarray(c if c is not None else (h + l) / 2, float)
    return ordered_adverse_curve(h, l, c, confirm, len(h) - 1 if end is None else end, d)


def test_race_is_future_only():
    """Changing any bar at/before confirmation cannot change a future target."""
    h = np.array([99., 100., 103., 106., 110.])
    l = np.array([98., 99., 99.5, 102., 108.])
    c = np.array([98.5, 100., 102., 105., 109.])
    base = _curve(h, l, c, confirm=1)
    h[:2] += np.array([500., 200.])
    l[:2] -= np.array([500., 200.])
    # close[confirm] is the causal reference and must stay fixed; earlier closes are irrelevant.
    c[0] += 1000.
    assert np.allclose(_curve(h, l, c, confirm=1), base)


def test_race_distinguishes_adverse_before_from_after_progress():
    # Same reference and eventual +10 extent. In clean, the +10 target occurs before the plunge;
    # in rough, the plunge occurs first. Ordered curves must distinguish them.
    c = np.full(5, 100.)
    clean_h = np.array([100., 100., 110., 110., 110.])
    clean_l = np.array([100., 100., 100., 95., 109.])
    rough_h = np.array([100., 100., 101., 110., 110.])
    rough_l = np.array([100., 100., 95., 99., 109.])
    clean = _curve(clean_h, clean_l, c, confirm=1)
    rough = _curve(rough_h, rough_l, c, confirm=1)
    assert clean[-1] == pytest.approx(0.0)
    assert rough[-1] == pytest.approx(0.5)
    assert np.all(rough >= clean)


def test_same_bar_is_conservatively_adverse():
    h = np.array([100., 100., 110.])
    l = np.array([100., 100., 95.])
    c = np.array([100., 100., 100.])
    assert _curve(h, l, c, confirm=1)[-1] == pytest.approx(0.5)


def test_race_is_scale_and_direction_invariant():
    h = np.array([100., 100., 102., 106., 110.])
    l = np.array([100., 100., 98., 101., 108.])
    c = np.array([100., 100., 101., 105., 109.])
    up = _curve(h, l, c, confirm=1, d=1)
    assert np.allclose(_curve(h * 10, l * 10, c * 10, confirm=1, d=1), up)
    assert np.allclose(_curve(200 - l, 200 - h, 200 - c, confirm=1, d=-1), up)


def test_invalid_or_unresolved_path_is_nan():
    flat = np.full(5, 100.)
    assert np.isnan(_curve(flat, flat, flat, confirm=1)).all()
    assert np.isnan(_curve(flat, flat, flat, confirm=2, end=2)).all()


def test_task_is_additive_and_inherits_nextleg_reserve():
    assert 'nextleg_race' in PRETEXTS
    task = get_pretext('nextleg_race')
    assert isinstance(task, NextLegRaceTask)
    assert task.trainer == 'train_ssl_nextleg_race'
    cfg = {'context_lengths': (64, 100, 150, 200), 'leg_cap': 256}
    assert task.reserve(cfg) == get_pretext('nextleg').reserve(cfg) == 712
    assert get_pretext('nextleg').trainer == 'train_ssl_nextleg'


def test_shared_config_keeps_race_overrides():
    """loop_ssl's allowlist must not silently discard the new experiment's knobs."""
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(pretext='nextleg_race', race_w=.5, race_cap=1.5,
                    race_levels=(.20, .40, .80, 1.0))
    assert cfg['pretext'] == 'nextleg_race'
    assert cfg['race_w'] == .5
    assert cfg['race_cap'] == 1.5
    assert cfg['race_levels'] == (.20, .40, .80, 1.0)


def _rw(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    return np.stack([c, c + np.abs(rng.normal(0, .3, n)),
                     c - np.abs(rng.normal(0, .3, n)), c,
                     np.abs(rng.normal(1e3, 50, n))], 1).astype(np.float32)


@torch_test
def test_nextleg_targets_are_unchanged_and_race_is_bounded():
    from futures_foundation.finetune.pretext._torch.nextleg import _leg_targets
    from futures_foundation.finetune.pretext._torch.nextleg_race import _leg_race_targets
    big = _rw()
    c26, t26, ok26 = _leg_targets(big, 2, 256)
    c28, t28, ok28 = _leg_race_targets(big, 2, 256)
    assert np.array_equal(c26, c28)
    assert np.allclose(t26, t28[:, :2])
    assert ok28.sum() <= ok26.sum()
    race = t28[ok28, 2:]
    assert race.shape[1] == len(RACE_LEVELS)
    assert len(race) > 100 and np.isfinite(race).all()
    assert (race >= 0).all() and (race <= 2.0).all()
    assert np.all(np.diff(race, axis=1) >= -1e-7)          # ordered MAE can only ratchet


@torch_test
def test_trainer_accepts_shared_config_without_forwarding_it():
    import inspect
    from futures_foundation.finetune._ssl_torch import train_ssl_nextleg, train_ssl_nextleg_race
    s26 = inspect.signature(train_ssl_nextleg).parameters
    s28 = inspect.signature(train_ssl_nextleg_race).parameters
    assert set(s26) <= set(s28)
    assert {'race_w', 'race_cap', 'race_levels'} <= set(s28)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in s28.values())


@torch_test
def test_base_nextleg_validation_is_fixed_and_preserves_training_rng():
    """Early stopping must compare identical pivots and not consume training draws."""
    import torch
    from futures_foundation.finetune.pretext._torch.nextleg import _NextLegTrainer

    class _Net(torch.nn.Module):
        def forward_all(self, ctx):
            return ctx[:, :, :2], ctx[:, :1, :2].flatten(1)

        def embed(self, ctx):
            return torch.cat((ctx.mean((1, 2))[:, None], ctx.std((1, 2))[:, None]), dim=1)

    t = object.__new__(_NextLegTrainer)
    t.net = _Net()
    t.dev = 'cpu'
    t.gen = torch.Generator().manual_seed(17)
    t.va = torch.arange(64)
    t.batch = 8
    t.mse_weight = t.leg_w = 1.0

    def make_batch(_starts, gen=None):
        x = torch.randn(8, 5, 4, generator=gen)
        return x, x[:, :, :2] + 0.1, x[:, :1, :2].flatten(1) + 0.2

    t.make_batch = make_batch
    before = t.gen.get_state().clone()
    v1, e1 = t.val_eval()
    v2, e2 = t.val_eval()
    assert v1 == pytest.approx(v2)
    assert e1 == pytest.approx(e2)
    assert torch.equal(t.gen.get_state(), before)
    assert t.net.training is True


@torch_test
def test_race_validation_is_fixed_and_restores_training_rng():
    """Repeated validation uses identical anchors and must not perturb subsequent training draws."""
    import torch
    from futures_foundation.finetune.pretext._torch.nextleg_race import _NextLegRaceTrainer

    class _Net(torch.nn.Module):
        def forward_all(self, ctx):
            # Deterministic predictions whose rows depend on the sampled validation batch.
            candles = ctx[:, :, :2]
            legs = torch.cat((ctx[:, :1, :2].flatten(1),
                              ctx[:, :1, :4].flatten(1)), dim=1)
            return candles, legs

        def embed(self, ctx):
            return torch.cat((ctx.mean((1, 2), keepdim=False)[:, None],
                              ctx.std((1, 2), keepdim=False)[:, None]), dim=1)

    t = object.__new__(_NextLegRaceTrainer)
    t.net = _Net()
    t.gen = torch.Generator().manual_seed(17)
    t.va = torch.arange(64)
    t.batch = 8
    t.mse_weight, t.leg_w, t.race_w = 1.0, 1.0, 0.25

    def make_batch(_starts):
        x = torch.randn(8, 5, 4, generator=t.gen)
        candles = x[:, :, :2] + 0.1
        legs = torch.cat((x[:, :1, :2].flatten(1) + 0.2,
                          x[:, :1, :4].flatten(1) + 0.3), dim=1)
        return x, candles, legs

    t.make_batch = make_batch
    before = t.gen.get_state().clone()
    v1, e1 = t.val_eval()
    assert torch.equal(t.gen.get_state(), before)
    v2, e2 = t.val_eval()
    assert v1 == pytest.approx(v2)
    assert e1['race_loss'] == pytest.approx(e2['race_loss'])
    assert torch.equal(t.gen.get_state(), before)
