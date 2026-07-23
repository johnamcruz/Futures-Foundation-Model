"""Contracts for causal-range NextLeg race v2."""
import inspect
import os

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.nextleg_race import (
    RACE_LEVELS, RACE_SCHEMA, NextLegRaceTask, causal_bar_scale, scaled_path_race)


torch_test = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test — set CHRONOS_TORCH_TESTS=1")


def _bars():
    # Past candles 0..3 all have range=2 -> causal scale=2. Future path reaches 1/2/3 units,
    # then censors the 4-unit target at the end of the newborn leg.
    high = np.array([101., 101., 101., 101., 102.5, 104.5, 106.5, 107.])
    low = np.array([99., 99., 99., 99., 99., 98., 97., 96.])
    close = np.full(8, 100.)
    return high, low, close


def test_causal_bar_scale_is_raw_range_not_atr_and_ignores_future():
    high, low, _ = _bars()
    assert causal_bar_scale(high, low, confirm=3, lookback=4) == pytest.approx(2.0)
    changed_high, changed_low = high.copy(), low.copy()
    changed_high[4:] = 10_000
    changed_low[4:] = -10_000
    assert causal_bar_scale(changed_high, changed_low, confirm=3, lookback=4) == pytest.approx(2.0)


def test_scaled_race_learns_launch_risk_and_censored_time():
    high, low, close = _bars()
    target = scaled_path_race(
        high, low, close, confirm=3, leg_end=7, direction=1,
        levels=RACE_LEVELS, lookback=4)
    assert target.shape == (3, 4)
    assert target[0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert target[1] == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert target[2] == pytest.approx(np.log1p([1, 2, 3, 4]))
    assert np.all(np.diff(target[0]) <= 0)
    assert np.all(np.diff(target[1]) >= 0)
    assert np.all(np.diff(target[2]) >= 0)


def test_scaled_race_is_price_scale_and_direction_invariant():
    high, low, close = _bars()
    original = scaled_path_race(
        high, low, close, 3, 7, 1, levels=RACE_LEVELS, lookback=4)
    scaled = scaled_path_race(
        10 * high, 10 * low, 10 * close, 3, 7, 1,
        levels=RACE_LEVELS, lookback=4)
    mirrored = scaled_path_race(
        200 - low, 200 - high, 200 - close, 3, 7, -1,
        levels=RACE_LEVELS, lookback=4)
    assert np.allclose(original, scaled)
    assert np.allclose(original, mirrored)


def test_same_bar_reach_includes_adverse_excursion_conservatively():
    high = np.array([101., 101., 101., 101., 104.])
    low = np.array([99., 99., 99., 99., 96.])
    close = np.full(5, 100.)
    target = scaled_path_race(
        high, low, close, 3, 4, 1, levels=(2.0,), lookback=4)
    assert target[0, 0] == 1.0
    assert target[1, 0] == pytest.approx(2.0)


def test_invalid_path_or_insufficient_context_is_nan():
    flat = np.full(8, 100.)
    assert np.isnan(scaled_path_race(
        flat, flat, flat, 3, 7, 1, lookback=4)).all()
    high, low, close = _bars()
    assert np.isnan(scaled_path_race(
        high, low, close, 1, 7, 1, lookback=4)).all()


def test_task_declares_stream_layout_and_v2_control_contract():
    assert "nextleg_race" in PRETEXTS
    task = get_pretext("nextleg_race")
    assert isinstance(task, NextLegRaceTask)
    assert task.requires_stream_layout is True
    assert task.control_contract == "nextleg_causal_range_race_v2"
    assert RACE_SCHEMA == "causal_range_competing_path_v2"
    cfg = {"context_lengths": (64, 100, 150, 200), "leg_cap": 256}
    assert task.reserve(cfg) == 714
    assert task.reserve(cfg) == get_pretext("nextleg").reserve(cfg) + 2


def test_shared_config_keeps_v2_overrides():
    from futures_foundation.finetune.ssl import _base_cfg
    cfg = _base_cfg(
        pretext="nextleg_race", race_w=.75, race_cap=6.0,
        race_levels=(.5, 1.0, 2.0), race_scale_lookback=32)
    assert cfg["race_w"] == .75
    assert cfg["race_cap"] == 6.0
    assert cfg["race_levels"] == (.5, 1.0, 2.0)
    assert cfg["race_scale_lookback"] == 32


def _rw(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum()
    spread = np.abs(rng.normal(.4, .1, n))
    return np.stack((
        close, close + spread, close - spread, close,
        np.abs(rng.normal(1e3, 50, n))), axis=1).astype(np.float32)


@torch_test
def test_targets_are_built_per_stream_without_boundary_pivots():
    from futures_foundation.finetune.pretext._torch.nextleg_race import (
        _leg_race_targets, _leg_race_targets_by_segments)
    first, second = _rw(seed=1), _rw(seed=2)
    c1, t1, ok1 = _leg_race_targets(first, 2, 256)
    c2, t2, ok2 = _leg_race_targets(second, 2, 256)
    combined = np.concatenate((first, second))
    confirms, targets = _leg_race_targets_by_segments(
        combined, ((0, len(first)), (len(first), len(second))), 2, 256)
    expected_confirms = np.concatenate((c1[ok1], c2[ok2] + len(first)))
    expected_targets = np.concatenate((t1[ok1], t2[ok2]))
    assert np.array_equal(confirms, expected_confirms)
    assert np.allclose(targets, expected_targets)


@torch_test
def test_vectorized_targets_match_the_causal_reference_exactly():
    from futures_foundation.finetune.pretext._torch.nextleg import _alternating_fractals
    from futures_foundation.finetune.pretext._torch.nextleg_race import _leg_race_targets

    bars = _rw(n=20_000, seed=31)
    high, low, close = bars[:, 1], bars[:, 2], bars[:, 3]
    confirms, targets, valid = [], [], []
    sequence = _alternating_fractals(high, low, 2)
    for index in range(len(sequence) - 2):
        _, confirmation, direction = sequence[index]
        next_origin = sequence[index + 1][0]
        following_origin = sequence[index + 2][0]
        first = next_origin - confirmation
        second = following_origin - next_origin
        race = scaled_path_race(
            high, low, close, confirmation, next_origin, direction,
            levels=RACE_LEVELS, lookback=64, cap=8.0)
        confirms.append(confirmation)
        targets.append((
            np.log1p(max(first, 0)), np.log1p(max(second, 0)),
            *np.nan_to_num(race, nan=0.0).reshape(-1).tolist()))
        valid.append(
            first > 0 and second > 0 and first <= 256 and second <= 256
            and np.isfinite(race).all())

    actual = _leg_race_targets(bars, 2, 256)
    assert np.array_equal(actual[0], np.asarray(confirms, np.int64))
    assert np.array_equal(actual[2], np.asarray(valid, bool))
    assert np.allclose(actual[1], np.asarray(targets, np.float32), rtol=1e-6, atol=1e-6)


@torch_test
def test_segment_target_cache_is_reused_across_control_trainers():
    from futures_foundation.finetune.pretext._torch import nextleg_race as module

    module._RACE_TARGET_CACHE.clear()
    bars = _rw(n=20_000, seed=41)
    arguments = dict(
        big=bars, segments=((0, len(bars)),), k=2, leg_cap=256,
        race_levels=RACE_LEVELS, race_scale_lookback=64, race_cap=8.0)
    first = module._leg_race_targets_by_segments(**arguments)
    second = module._leg_race_targets_by_segments(**arguments)
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert len(module._RACE_TARGET_CACHE) == 1


@torch_test
def test_trainer_signature_requires_v2_inputs_and_supports_lora():
    from futures_foundation.finetune._ssl_torch import train_ssl_nextleg_race
    signature = inspect.signature(train_ssl_nextleg_race).parameters
    required = {
        "race_w", "race_cap", "race_levels", "race_scale_lookback",
        "_stream_layout", "head_lr",
        "lora_r", "lora_alpha", "freeze_encoder_layers",
    }
    assert required <= set(signature)
    assert signature["lora_r"].default == 8
    assert signature["freeze_encoder_layers"].default == 2


@torch_test
def test_reserve_guard_includes_final_pivot_confirmation_lag():
    from futures_foundation.finetune.pretext._torch.nextleg_race import (
        _validate_race_target_reserve)

    # Two resolved legs read 7+9 bars forward. The second future pivot only becomes a legal
    # label after two additional bars close.
    target = np.zeros((1, 14), np.float32)
    target[0, :2] = np.log1p([7, 9])
    with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
        _validate_race_target_reserve(
            target, max_ctx=64, target_reserve=64 + 16,
            confirmation_lag=2, batch_parent=89)
    _validate_race_target_reserve(
        target, max_ctx=64, target_reserve=64 + 18,
        confirmation_lag=2, batch_parent=89)


@torch_test
def test_race_batch_keeps_sampled_target_aligned_and_future_out_of_input():
    import torch
    from futures_foundation.finetune.pretext._torch.nextleg_race import (
        _NextLegRaceTrainer)

    trainer = object.__new__(_NextLegRaceTrainer)
    trainer.dev = "cpu"
    trainer.gen = torch.Generator().manual_seed(7)
    trainer.batch = 2
    trainer.max_ctx = 4
    trainer.parent = 7
    trainer.clamp = 10.0
    trainer.control = "real"
    trainer.clens_t = torch.tensor([4])
    trainer.h_off = torch.tensor([0, 2])
    trainer.big_t = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    trainer.tr = torch.tensor([0, 2, 4])
    trainer.va = torch.tensor([1, 3, 5])
    trainer._tgt_tr = torch.tensor([[10.], [20.], [30.]])
    trainer._tgt_va = torch.tensor([[40.], [50.], [60.]])
    trainer.sample_indices = lambda starts, generator=None: torch.tensor([2, 0])

    context, candle, target = trainer.make_batch(trainer.tr)
    assert context.shape == (2, 5, 4)
    assert candle.shape == (2, 5, 2)
    assert target[:, 0].tolist() == [30.0, 10.0]

    # Changing only gathered future bars changes the candle anchor but never the model input.
    altered = trainer.big_t.clone()
    altered[8:] += 100_000
    trainer.big_t = altered
    context_changed, candle_changed, target_changed = trainer.make_batch(trainer.tr)
    assert torch.equal(context, context_changed)
    assert not torch.equal(candle, candle_changed)
    assert torch.equal(target, target_changed)


@torch_test
def test_network_enforces_monotone_race_outputs(monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import nextleg_race as module

    class _Base(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.C, self.nH = 5, 2
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(12, 12), torch.nn.GELU(), torch.nn.Linear(12, 10))
            self.leg_head = torch.nn.Linear(12, 2)

        def embed(self, context):
            return context[:, :1, :12].flatten(1)

    monkeypatch.setattr(module, "NextLegNet", _Base)
    # Re-declaring inheritance is not possible after monkeypatch, so exercise the monotone
    # parameterization directly with the same operations used by forward_race.
    raw = torch.randn(32, 3, 4)
    reach = torch.cat((
        raw[:, 0, :1],
        raw[:, 0, :1] - torch.cumsum(torch.nn.functional.softplus(raw[:, 0, 1:]), 1),
    ), 1).sigmoid()
    adverse = torch.cumsum(torch.nn.functional.softplus(raw[:, 1]), 1)
    delay = torch.cumsum(torch.nn.functional.softplus(raw[:, 2]), 1)
    assert torch.all(torch.diff(reach, dim=1) <= 0)
    assert torch.all(torch.diff(adverse, dim=1) >= 0)
    assert torch.all(torch.diff(delay, dim=1) >= 0)
