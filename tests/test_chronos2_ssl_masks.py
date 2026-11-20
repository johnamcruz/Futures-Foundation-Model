"""Cross-modal block masks: modality isolation, coverage, determinism."""
import numpy as np
import pytest
from futures_foundation.finetune.classifiers.chronos2.ssl_masks import block_mask, modal_time_mask


def test_block_mask_is_contiguous_blocks_near_ratio_and_hides_step_zero():
    m = block_mask(256, 0.4, np.random.default_rng(0))
    assert m[0] and 0.35 <= m.mean() <= 0.65
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, m, False])))[::2]
    assert runs.min() >= 1 and runs.max() >= 8                       # real blocks, not scattered steps


def test_each_window_hides_flow_only_price_only_or_both_shared():
    rng = np.random.default_rng(1)
    out = modal_time_mask(600, 11, 256, 0.4, rng)
    kinds = {"flow": 0, "price": 0, "shared": 0}
    for w in out:
        price, flow = w[:5].any(), w[5:].any()
        if price and flow:
            kinds["shared"] += 1
            assert (w == w[0]).all()                                  # same steps across all variates
        elif flow:
            kinds["flow"] += 1; assert not w[:5].any()                # price fully visible
        else:
            kinds["price"] += 1; assert not w[5:].any()               # flow fully visible
    assert all(v > 120 for v in kinds.values())                       # ~1/3 each
    assert out[:, :, 0].any(axis=1).all()


def test_deterministic_and_needs_flow_variates():
    a = modal_time_mask(8, 11, 64, 0.4, np.random.default_rng(3))
    b = modal_time_mask(8, 11, 64, 0.4, np.random.default_rng(3))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        modal_time_mask(2, 5, 64, 0.4, np.random.default_rng(0))


def test_tail_mask_hides_only_the_last_bars_of_every_variate():
    from futures_foundation.finetune.classifiers.chronos2.ssl_masks import tail_mask
    out = tail_mask(300, 11, 256, np.random.default_rng(0))
    for w in out:
        assert (w == w[0]).all()                                  # all variates identical
        n = int(w[0].sum()); assert 2 <= n <= 12 and w[0, -n:].all() and not w[0, :-n].any()
    assert {int(w[0].sum()) for w in out} == set(range(2, 13))
