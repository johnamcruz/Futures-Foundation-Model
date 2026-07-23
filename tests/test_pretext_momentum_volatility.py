"""Causal contracts for momentum-volatility coupling SSL."""
from __future__ import annotations

import numpy as np
import pytest

from futures_foundation.finetune.pretext import PRETEXTS, get_pretext
from futures_foundation.finetune.pretext.momentum_volatility import (
    MV_COMPRESSION,
    MV_NOISY_EXPANSION,
    MV_TREND_EXPANSION,
    MV_TREND_WEAKENING,
    MOMENTUM_VOLATILITY_SCHEMA,
    MomentumVolatilityTask,
    momentum_volatility_targets,
    transition_class,
)


def _bars():
    close = np.asarray([
        98., 98.5, 99., 99.5, 100.,  # decision=4: established positive momentum
        101., 103., 102., 96.,
    ])
    width = np.asarray([
        2., 2., 2., 2., 2.,
        3., 4., 3., 5.,
    ])
    return close + width / 2, close - width / 2, close


def test_targets_use_only_completed_context_for_scale_and_past_direction():
    high, low, close = _bars()
    original = momentum_volatility_targets(
        high, low, close, decision=4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3,
        momentum_threshold=.5, expansion_threshold=1.1)
    changed_high, changed_low = high.copy(), low.copy()
    changed_high[5:] += 10_000
    changed_low[5:] -= 10_000
    changed = momentum_volatility_targets(
        changed_high, changed_low, close, decision=4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3,
        momentum_threshold=.5, expansion_threshold=1.1)

    assert original.causal_scale == pytest.approx(2.0)
    assert changed.causal_scale == pytest.approx(original.causal_scale)
    assert changed.past_momentum == pytest.approx(original.past_momentum)
    assert changed.volatility[0] != pytest.approx(original.volatility[0])


def test_targets_are_price_scale_and_direction_invariant():
    high, low, close = _bars()
    original = momentum_volatility_targets(
        high, low, close, 4, horizons=(2, 4),
        scale_lookback=4, momentum_lookback=3)
    scaled = momentum_volatility_targets(
        10 * high, 10 * low, 10 * close, 4, horizons=(2, 4),
        scale_lookback=4, momentum_lookback=3)
    mirrored = momentum_volatility_targets(
        200 - low, 200 - high, 200 - close, 4, horizons=(2, 4),
        scale_lookback=4, momentum_lookback=3)

    np.testing.assert_allclose(original.momentum_strength, scaled.momentum_strength)
    np.testing.assert_allclose(original.volatility, scaled.volatility)
    np.testing.assert_array_equal(original.coupling, scaled.coupling)
    np.testing.assert_allclose(
        original.momentum_strength, mirrored.momentum_strength)
    np.testing.assert_allclose(original.volatility, mirrored.volatility)
    np.testing.assert_array_equal(original.coupling, mirrored.coupling)


def test_future_after_largest_horizon_cannot_change_any_target():
    high, low, close = _bars()
    original = momentum_volatility_targets(
        high, low, close, 4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3)
    high[-2:], low[-2:], close[-2:] = 1e6, -1e6, -1e6
    changed = momentum_volatility_targets(
        high, low, close, 4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3)

    np.testing.assert_allclose(
        original.momentum_strength, changed.momentum_strength)
    np.testing.assert_allclose(original.volatility, changed.volatility)
    np.testing.assert_array_equal(original.coupling, changed.coupling)


@pytest.mark.parametrize(
    "momentum_strength,volatility_ratio,expected",
    [
        (0.8, 1.5, MV_TREND_EXPANSION),
        (0.8, .8, MV_TREND_WEAKENING),
        (0.1, 1.5, MV_NOISY_EXPANSION),
        (0.1, .8, MV_COMPRESSION),
    ],
)
def test_transition_class_preserves_all_four_economic_states(
    momentum_strength, volatility_ratio, expected,
):
    assert transition_class(
        momentum_strength, volatility_ratio,
        momentum_threshold=.5,
        expansion_threshold=1.1,
    ) == expected


def test_path_efficiency_distinguishes_persistent_move_from_round_trip():
    high, low, close = _bars()
    persistent = momentum_volatility_targets(
        high, low, close, 4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3)

    round_trip_close = close.copy()
    round_trip_close[5:7] = [103.0, 101.0]
    round_trip = momentum_volatility_targets(
        high, low, round_trip_close, 4, horizons=(2,),
        scale_lookback=4, momentum_lookback=3)

    assert persistent.momentum_strength[0] > round_trip.momentum_strength[0]


def test_insufficient_history_or_future_fails_closed():
    high, low, close = _bars()
    insufficient_history = momentum_volatility_targets(
        high, low, close, 2, horizons=(2,),
        scale_lookback=4, momentum_lookback=3)
    insufficient_future = momentum_volatility_targets(
        high, low, close, 4, horizons=(8,),
        scale_lookback=4, momentum_lookback=3)

    assert not insufficient_history.valid.any()
    assert not insufficient_future.valid.any()
    assert np.isnan(insufficient_history.momentum_strength).all()
    assert np.isnan(insufficient_future.volatility).all()
    assert (insufficient_history.coupling == -1).all()


def test_task_is_registered_with_own_control_contract_and_split_reserve():
    assert "momentum_volatility" in PRETEXTS
    task = get_pretext("momentum_volatility")
    assert isinstance(task, MomentumVolatilityTask)
    assert task.control_contract == "momentum_volatility_transition_v3"
    assert MOMENTUM_VOLATILITY_SCHEMA == "causal_momentum_volatility_v3"
    cfg = {
        "context_lengths": (64, 100, 150, 200),
        "horizons": (5, 10, 20, 25),
    }
    assert task.reserve(cfg) == 225


def test_shared_ssl_config_preserves_mv_v3_encoder_objective_knobs():
    from futures_foundation.finetune.ssl import _base_cfg

    cfg = _base_cfg(
        transition_contrastive_weight=2.0,
        parent_retention_weight=.75,
        contrastive_temperature=.2,
        scale_lookback=48,
        probe_baseline_ckpt="parent.pt",
    )
    assert cfg["transition_contrastive_weight"] == 2.0
    assert cfg["parent_retention_weight"] == .75
    assert cfg["contrastive_temperature"] == .2
    assert cfg["scale_lookback"] == 48
    assert cfg["probe_baseline_ckpt"] == "parent.pt"


def test_control_gate_requires_real_momentum_volatility_and_coupling_edge():
    task = MomentumVolatilityTask()
    real = {
        "mv_momentum_strength_corr": .2,
        "mv_volatility_corr": .3,
        "mv_transition_auc": .65,
        "mv_transition_worst_auc": .58,
    }
    controls = {
        "shuffle": {
            "mv_momentum_strength_corr": .01,
            "mv_volatility_corr": .02,
            "mv_transition_auc": .51,
            "mv_transition_worst_auc": .50,
        },
        "random": {
            "mv_momentum_strength_corr": -.01,
            "mv_volatility_corr": .0,
            "mv_transition_auc": .49,
            "mv_transition_worst_auc": .48,
        },
    }
    passed, margins, temporal = task.compare_control_evidence(real, controls)
    assert passed
    assert margins["shuffle"]["mv_transition_auc"] == pytest.approx(.14)
    assert temporal == pytest.approx(.14)

    failed, _, _ = task.compare_control_evidence(
        {**real, "mv_transition_auc": .49}, controls)
    assert not failed

    assert task.control_evidence(real, None) == real


def test_mv_gate_retains_parent_context_without_demanding_unrelated_probe_lift():
    task = MomentumVolatilityTask()
    retained = {
        "mean_core_delta": -.001,
        "descriptive_delta": -.002,
        "fwd_absmove_delta": .001,
        "fwd_dir_delta": -.003,
        "learns_regime_vol_structure": True,
    }
    passed, detail = task.gate(
        retained, std=.8, margin=.002, dir_margin=0.0, forecast_skill=.05)
    assert passed
    assert detail["mv_parent_retention_ok"]

    regressed = {**retained, "fwd_dir_delta": -.006}
    passed, detail = task.gate(
        regressed, std=.8, margin=.002, dir_margin=0.0, forecast_skill=.05)
    assert not passed
    assert not detail["mv_parent_retention_ok"]
