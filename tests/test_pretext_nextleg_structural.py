"""Causality and target-contract tests for Structural NextLeg SSL."""
from __future__ import annotations

import inspect

import numpy as np


def _bars(n=24):
    close = np.linspace(8.0, 9.0, n)
    out = np.column_stack((close, close + 0.2, close - 0.2, close, np.ones(n)))
    # Alternating extrema: H10, L5, HH12, HL7, LH11, LL4, LH10.
    for index, direction, price in (
        (2, -1, 10.0), (5, 1, 5.0), (8, -1, 12.0), (11, 1, 7.0),
        (14, -1, 11.0), (17, 1, 4.0), (20, -1, 10.0),
    ):
        if direction == -1:
            out[index, 1] = price
        else:
            out[index, 2] = price
    return out.astype(np.float32)


def _pivots(offset=0):
    # k=1: every origin is known one completed bar later.
    return [(offset + origin, offset + origin + 1, direction)
            for origin, direction in ((2, -1), (5, 1), (8, -1), (11, 1),
                                      (14, -1), (17, 1), (20, -1))]


def test_structure_states_encode_hh_hl_lh_ll_sequences():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        CONTRACTING, DOWNTREND, UPTREND, structural_targets_from_pivots)

    targets = structural_targets_from_pivots(_bars(), _pivots(), k=1, leg_cap=16)
    row = np.flatnonzero(targets.confirms == 12)[0]  # confirmed HL at origin 11
    assert targets.current_state[row] == UPTREND       # HH + HL
    assert targets.next_state[row] == CONTRACTING      # LH + HL
    row = np.flatnonzero(targets.confirms == 15)[0]  # confirmed LH at origin 14
    assert targets.current_state[row] == CONTRACTING
    assert targets.next_state[row] == DOWNTREND        # LH + LL


def test_current_structure_is_invariant_to_every_bar_after_confirmation():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        structural_targets_from_pivots)

    bars = _bars()
    first = structural_targets_from_pivots(bars, _pivots(), k=1, leg_cap=16)
    changed = bars.copy()
    changed[13:, 0:4] *= 7.0
    second = structural_targets_from_pivots(changed, _pivots(), k=1, leg_cap=16)
    i = np.flatnonzero(first.confirms == 12)[0]
    j = np.flatnonzero(second.confirms == 12)[0]
    assert first.current_state[i] == second.current_state[j]


def test_target_records_future_pivot_confirmation_not_only_extreme():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        structural_targets_from_pivots)

    targets = structural_targets_from_pivots(_bars(), _pivots(), k=1, leg_cap=16)
    row = np.flatnonzero(targets.confirms == 12)[0]
    assert targets.future_ends[row] == 18  # second future pivot origin=17, confirm=18
    assert targets.future_ends[row] > 17


def test_segmented_target_builder_cannot_create_cross_stream_pivots(monkeypatch):
    from futures_foundation.finetune.pretext import nextleg_structural as structural

    calls = []

    def detector(segment, k, leg_cap, *, event_horizon):
        calls.append((len(segment), event_horizon))
        return structural.structural_targets_from_pivots(
            segment, _pivots(), k=k, leg_cap=leg_cap, event_horizon=event_horizon)

    big = np.concatenate((_bars(), _bars()), axis=0)
    result = structural.structural_targets_by_segments(
        big, ((0, 24), (24, 24)), k=1, leg_cap=16,
        event_horizon=3, builder=detector)
    assert calls == [(24, 3), (24, 3)]
    assert np.all(result.confirms[:len(result.confirms) // 2] < 24)
    assert np.all(result.confirms[len(result.confirms) // 2:] >= 24)
    assert np.all(result.future_ends[:len(result.future_ends) // 2] < 24)
    assert np.all(result.future_ends[len(result.future_ends) // 2:] >= 24)


def test_structural_reserve_includes_both_legs_and_confirmation_lag():
    from futures_foundation.finetune.pretext import get_pretext

    task = get_pretext('nextleg_structural')
    cfg = {'context_lengths': (64, 200), 'leg_cap': 256, 'leg_k': 3}
    assert task.reserve(cfg) == 200 + 2 * 256 + 3


def test_structural_future_guard_fails_closed_at_split_boundary():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        validate_structural_reserve)

    validate_structural_reserve(
        np.array([100]), np.array([106]), max_ctx=64, target_reserve=70)
    try:
        validate_structural_reserve(
            np.array([100]), np.array([107]), max_ctx=64, target_reserve=70)
    except AssertionError as exc:
        assert 'TEMPORAL LEAK' in str(exc)
    else:
        raise AssertionError('future confirmation outside reserve was accepted')


def test_structural_target_is_scale_invariant():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        structural_targets_from_pivots)

    bars = _bars()
    first = structural_targets_from_pivots(bars, _pivots(), k=1, leg_cap=16)
    scaled = bars.copy()
    scaled[:, :4] *= 10.0
    second = structural_targets_from_pivots(scaled, _pivots(), k=1, leg_cap=16)
    np.testing.assert_array_equal(first.current_state, second.current_state)
    np.testing.assert_array_equal(first.next_state, second.next_state)
    np.testing.assert_allclose(first.excursions, second.excursions, atol=1e-6)


def test_bos_choch_uses_only_future_closes_and_past_confirmed_levels():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        BEARISH_CHOCH, BULLISH_BOS, UPTREND, first_bos_choch_event)

    # Decision close is element 0; scanning begins at element 1.  The established levels
    # (high=12, low=7) were confirmed before this array is presented to the target function.
    event, delay = first_bos_choch_event(
        np.array([9.0, 10.0, 12.2, 6.5]), UPTREND,
        last_high=12.0, last_low=7.0, horizon=3)
    assert (event, delay) == (BULLISH_BOS, 2)  # continuation occurred before reversal

    event, delay = first_bos_choch_event(
        np.array([9.0, 6.8, 12.5, 13.0]), UPTREND,
        last_high=12.0, last_low=7.0, horizon=3)
    assert (event, delay) == (BEARISH_CHOCH, 1)


def test_bos_choch_never_reads_past_declared_horizon():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        NO_BREAK, UPTREND, first_bos_choch_event)

    event, delay = first_bos_choch_event(
        np.array([9.0, 10.0, 11.0, 12.5]), UPTREND,
        last_high=12.0, last_low=7.0, horizon=2)
    assert (event, delay) == (NO_BREAK, 2)


def test_bos_choch_is_masked_when_structure_is_not_directional():
    from futures_foundation.finetune.pretext.nextleg_structural import (
        CONTRACTING, MASKED_BREAK, first_bos_choch_event)

    event, delay = first_bos_choch_event(
        np.array([9.0, 13.0, 6.0]), CONTRACTING,
        last_high=12.0, last_low=7.0, horizon=2)
    assert (event, delay) == (MASKED_BREAK, 0)


def test_structural_span_is_centered_on_origin_and_never_crosses_decision():
    from futures_foundation.finetune.pretext.nextleg_structural import structural_span_bounds

    # Context index 63 is the decision/confirmation candle and k=2 places the origin at 61.
    assert structural_span_bounds(64, confirmation_lag=2, span_width=5) == (59, 64)
    try:
        structural_span_bounds(64, confirmation_lag=2, span_width=7)
    except ValueError as exc:
        assert "confirmed pivot formation" in str(exc)
    else:
        raise AssertionError("span was allowed to cross the causal confirmation boundary")


def test_structural_control_gate_requires_above_chance_classification():
    from futures_foundation.finetune.pretext import get_pretext

    task = get_pretext("nextleg_structural")
    strong = dict(forecast_skill=.03, leg_corr1=.10, leg_corr2=.05,
                  current_structure_bal_acc=.35,
                  next_structure_bal_acc=.30, break_bal_acc=.25,
                  excursion_corr=.05, span_skill=.10)
    controls = {"shuffle": {key: value - .01 for key, value in strong.items()}}
    assert task.compare_control_evidence(strong, controls)[0]
    weak = {**strong, "next_structure_bal_acc": .24}
    weak_controls = {"shuffle": {key: value - .01 for key, value in weak.items()}}
    assert not task.compare_control_evidence(weak, weak_controls)[0]


def test_shared_config_and_trainer_signature_keep_every_structural_knob():
    from futures_foundation.finetune import ssl
    from futures_foundation.finetune._ssl_torch import train_ssl_nextleg_structural

    overrides = dict(
        pretext="nextleg_structural", structure_current_w=.1, structure_next_w=.8,
        excursion_w=.3, structure_event_w=.9, structure_event_horizon=96,
        structure_span_w=.4, structure_span_width=3, structure_span_prob=.75,
        head_lr=2e-4, warm_trainer_ckpt="parent.trainer.pt", freeze_encoder=True)
    cfg = ssl._base_cfg(**overrides)
    assert all(cfg[key] == value for key, value in overrides.items())
    signature = inspect.signature(train_ssl_nextleg_structural).parameters
    assert set(overrides) - {"pretext"} <= set(signature)
    assert any(value.kind is inspect.Parameter.VAR_KEYWORD for value in signature.values())


def test_warm_loader_requires_complete_matching_nextleg_heads(tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch.nextleg_structural import (
        _load_warm_nextleg_heads)

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.leg_head = torch.nn.Linear(2, 2)
            self.structure_head = torch.nn.Linear(2, 2)

    source, target = TinyNet(), TinyNet()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7.0)
        for parameter in target.parameters():
            parameter.fill_(1.0)
    path = tmp_path / "nextleg.trainer.pt"
    torch.save({"model_state": source.state_dict()}, path)
    loaded = _load_warm_nextleg_heads(target, path)
    assert loaded and all(key.startswith(("adapter.", "decoder.", "leg_head."))
                          for key in loaded)
    assert all(torch.allclose(target.state_dict()[key], source.state_dict()[key])
               for key in loaded)
    assert torch.all(target.structure_head.weight == 1.0)  # structural head stays new

    broken = dict(source.state_dict())
    broken.pop("leg_head.bias")
    torch.save({"model_state": broken}, path)
    try:
        _load_warm_nextleg_heads(target, path)
    except RuntimeError as exc:
        assert "missing task tensors" in str(exc)
    else:
        raise AssertionError("partial parent task state was silently accepted")


def test_combined_objective_backpropagates_through_every_head():
    import torch
    from futures_foundation.finetune.pretext._torch.nextleg_structural import (
        _StructuralNextLegTrainer)

    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candles = torch.nn.Parameter(torch.zeros(1, 5, 2))
            self.duration = torch.nn.Parameter(torch.zeros(1, 2))
            self.structure = torch.nn.Parameter(torch.zeros(1, 2, 4))
            self.excursion = torch.nn.Parameter(torch.zeros(1, 2))
            self.breaks = torch.nn.Parameter(torch.zeros(1, 5))
            self.delay = torch.nn.Parameter(torch.zeros(1))
            self.span = torch.nn.Parameter(torch.zeros(1, 5, 5))

        def forward_structural(self, context):
            n = len(context)
            return tuple(parameter.expand(n, *parameter.shape[1:]) for parameter in (
                self.candles, self.duration, self.structure, self.excursion,
                self.breaks, self.delay, self.span))

    trainer = object.__new__(_StructuralNextLegTrainer)
    trainer.net = Heads()
    trainer.mse_weight = trainer.leg_w = 1.0
    trainer.structure_current_w = trainer.structure_next_w = 1.0
    trainer.excursion_w = trainer.structure_event_w = trainer.structure_span_w = 1.0
    trainer.event_weights = torch.ones(5)
    n = 5
    targets = torch.tensor([
        [1., 1., 0., 1., 1., 1., 0., 0.],
        [1., 1., 1., 2., 1., 1., 1., 1.],
        [1., 1., 2., 3., 1., 1., 2., 1.],
        [1., 1., 3., 0., 1., 1., 3., 1.],
        [1., 1., 0., 1., 1., 1., 3., 1.],
    ])
    batch = (torch.zeros(n, 5, 64), torch.ones(n, 5, 2), targets,
             torch.ones(n, 5, 5), torch.ones(n, dtype=torch.bool))
    trainer.compute_loss(batch).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0
               for parameter in trainer.net.parameters())
