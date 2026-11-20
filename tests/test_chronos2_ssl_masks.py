"""Cross-modal / tail masks: roles from channel names, modality isolation, price-only loss, redraw, determinism."""
import numpy as np
import pytest
from futures_foundation.finetune.classifiers.chronos2.ssl_masks import (
    block_mask, flow_flags, modal_time_mask, selection_masks, tail_mask)

NAMES = [f"NQ.{c}" for c in ("open", "high", "low", "close", "volume")] + [f"NQ.flow_{c}" for c in ("buy_vol", "sell_vol", "delta")]


def test_roles_come_from_channel_names_not_positions():
    shuffled = ["NQ.flow_delta"] + NAMES[:5] + ["ES.open"]
    assert flow_flags(shuffled).tolist() == [True, False, False, False, False, False, False]
    assert flow_flags(NAMES).sum() == 3


def test_block_mask_is_contiguous_blocks_near_ratio_and_hides_step_zero():
    m = block_mask(256, 0.4, np.random.default_rng(0))
    assert m[0] and 0.35 <= m.mean() <= 0.65
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, m, False])))[::2]
    assert runs.max() >= 8


def test_each_window_hides_flow_only_price_only_or_both_shared():
    is_flow = flow_flags(NAMES)
    out = modal_time_mask(600, is_flow, 256, 0.4, np.random.default_rng(1))
    kinds = {"flow": 0, "price": 0, "shared": 0}
    for w in out:
        price, flow = w[~is_flow].any(), w[is_flow].any()
        if price and flow:
            kinds["shared"] += 1; assert (w == w[0]).all()
        elif flow:
            kinds["flow"] += 1
        else:
            kinds["price"] += 1
    assert all(v > 120 for v in kinds.values())
    assert out[:, :, 0].any(axis=1).all()


def test_cross_works_when_flow_is_not_the_trailing_channels():
    names = ["NQ.flow_delta", "NQ.open", "NQ.high", "NQ.low", "NQ.close", "NQ.volume", "ES.open"]
    is_flow = flow_flags(names)
    out = modal_time_mask(300, is_flow, 64, 0.4, np.random.default_rng(2))
    only_flow = [w for w in out if w[is_flow].any() and not w[~is_flow].any()]
    assert only_flow and all(not w[1:].any() for w in only_flow)      # only channel 0 is hidden


def test_deterministic_and_refuses_without_both_roles():
    f = flow_flags(NAMES)
    a = modal_time_mask(8, f, 64, 0.4, np.random.default_rng(3)); b = modal_time_mask(8, f, 64, 0.4, np.random.default_rng(3))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        modal_time_mask(2, np.zeros(5, bool), 64, 0.4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        modal_time_mask(2, np.ones(5, bool), 64, 0.4, np.random.default_rng(0))


def test_tail_mask_hides_only_the_last_bars_of_every_variate():
    out = tail_mask(300, 8, 256, np.random.default_rng(0))
    for w in out:
        assert (w == w[0]).all()
        n = int(w[0].sum()); assert 2 <= n <= 12 and w[0, -n:].all() and not w[0, :-n].any()
    assert {int(w[0].sum()) for w in out} == set(range(2, 13))


def test_tail_loss_counts_price_variates_only_and_never_flow():
    finite = np.ones((50, 8, 256), bool); is_flow = flow_flags(NAMES)
    sel, vis = selection_masks(finite, is_flow, "tail", 0.4, np.random.default_rng(0))
    assert not sel[:, is_flow, :].any() and sel[:, ~is_flow, :].any()
    assert not (sel & vis).any()
    assert vis[:, :, :240].all()                                         # only the tail is hidden


def test_tail_on_a_price_only_corpus_still_scores_all_price_variates():
    finite = np.ones((10, 5, 64), bool)
    sel, vis = selection_masks(finite, np.zeros(5, bool), "tail", 0.4, np.random.default_rng(0))
    assert sel[:, :, -2:].all()


def test_cross_selection_without_flow_returns_none_and_respects_finite():
    finite = np.ones((4, 5, 64), bool)
    assert selection_masks(finite, np.zeros(5, bool), "cross", 0.4, np.random.default_rng(0)) is None
    finite = np.ones((200, 8, 64), bool); finite[:, 5:, :20] = False          # flow missing early in every window
    sel, vis = selection_masks(finite, flow_flags(NAMES), "cross", 0.4, np.random.default_rng(1))
    assert not (sel & ~finite).any() and not (vis & ~finite).any()


def test_a_variate_never_loses_every_visible_step():
    # flow has only 12 finite steps; a block of 8-48 could hide all of them
    finite = np.ones((400, 8, 64), bool); finite[:, 5:, :52] = False
    sel, vis = selection_masks(finite, flow_flags(NAMES), "cross", 0.5, np.random.default_rng(5))
    assert (vis[:, 5:, :].any(-1)).all()


def test_selection_masks_rejects_other_modes():
    with pytest.raises(ValueError):
        selection_masks(np.ones((1, 8, 16), bool), flow_flags(NAMES), "time", 0.4, np.random.default_rng(0))


def test_redraw_gives_back_a_visible_step_to_a_variate_whose_only_finite_step_is_the_first():
    finite = np.ones((50, 8, 64), bool); finite[:, 5:, 1:] = False        # flow finite only at step 0
    sel, vis = selection_masks(finite, flow_flags(NAMES), "cross", 0.5, np.random.default_rng(0))
    assert vis[:, 5:, 0].all() and not sel[:, 5:, 0].any()
    sel, vis = selection_masks(finite, flow_flags(NAMES), "tail", 0.5, np.random.default_rng(0))
    assert vis[:, 5:, 0].all() and not sel[:, 5:, :].any()
