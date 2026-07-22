"""Leakage, parity, and checkpoint tests for related-series Mantis context."""
from __future__ import annotations

import numpy as np
import pytest

from futures_foundation.finetune.related_series import (
    RelatedSeriesLayout, parse_timeframe_pairs, timeframe_minutes)


def _timestamps(minutes):
    base = np.datetime64("2025-01-02T00:00:00", "ns")
    return base + np.asarray(minutes).astype("timedelta64[m]")


def _layout():
    records, base = [], 0
    specs = [
        ("NQ@1min", "NQ", "1min", np.arange(0, 40)),
        ("NQ@3min", "NQ", "3min", np.arange(0, 40, 3)),
        ("NQ@5min", "NQ", "5min", np.arange(0, 40, 5)),
        ("NQ@15min", "NQ", "15min", np.arange(0, 46, 15)),
        ("ES@3min", "ES", "3min", np.arange(0, 40, 3)),
    ]
    for sid, ticker, tf, minutes in specs:
        records.append({"sid": sid, "ticker": ticker, "tf": tf, "base": base,
                        "ts": _timestamps(minutes)})
        base += len(minutes)
    return RelatedSeriesLayout.from_assembled(records)


def _stream(layout, sid):
    return next(stream for stream in layout.streams if stream.sid == sid)


def test_timeframe_minutes_is_strict():
    assert timeframe_minutes("1min") == 1
    assert timeframe_minutes("15 minutes") == 15
    with pytest.raises(ValueError):
        timeframe_minutes("hourly")
    with pytest.raises(ValueError):
        timeframe_minutes(0)


def test_timeframe_pair_parser_supports_bidirectional_and_directional_groups():
    assert parse_timeframe_pairs("1min=5min,3min=15min") == {
        "1min": ("5min",), "5min": ("1min",),
        "3min": ("15min",), "15min": ("3min",),
    }
    assert parse_timeframe_pairs("1min:3min+5min+15min,3min:5min+15min") == {
        "1min": ("3min", "5min", "15min"),
        "3min": ("5min", "15min"),
    }
    assert parse_timeframe_pairs("") == {}
    assert parse_timeframe_pairs(None) is None
    with pytest.raises(ValueError, match="cannot pair with itself"):
        parse_timeframe_pairs("3min=3min")


def test_alignment_uses_closed_bars_for_lower_and_higher_timeframes():
    layout = _layout()
    primary = _stream(layout, "NQ@3min")
    # seq=2, primary bars stamped 0 and 3 -> decision close is minute 6.
    plan = layout.align([primary.base], 2, related_tfs=("1min", "5min", "15min"),
                        siblings="NQ:ES,ES:NQ")
    roles = {name: i for i, name in enumerate(plan.role_names)}
    nq1, nq5, nq15 = (_stream(layout, sid) for sid in ("NQ@1min", "NQ@5min", "NQ@15min"))
    es3 = _stream(layout, "ES@3min")

    # Last 1m bar open=5 closes=6; its two-bar window starts at open=4.
    assert plan.starts[0, roles["same_ticker@1min"]] == nq1.base + 4
    # 5m open=0 closes=5 but has only one bar of history, so seq=2 correctly masks it.
    assert not plan.mask[0, roles["same_ticker@5min"]]
    # The 15m bar stamped 0 does not close until minute 15: physically unavailable at minute 6.
    assert not plan.mask[0, roles["same_ticker@15min"]]
    # Same-timeframe ES bar stamped 3 closes with NQ and is available; window stamps 0,3.
    assert plan.starts[0, roles["sibling"]] == es3.base

    # Later: NQ bars stamped 12,15 decide at minute 18. The 15m bar stamped 0 is closed, but
    # seq=2 still needs the next 15m bar (which closes at 30), so it remains unavailable.
    later = layout.align([primary.base + 4], 2, related_tfs=("15min",), siblings="0")
    assert not later.mask[0, 1]
    # With seq=1, the completed bar is usable and the unfinished 15m@15 bar is unreachable.
    later_one = layout.align([primary.base + 5], 1, related_tfs=("15min",), siblings="0")
    assert later_one.mask[0, 1]
    assert later_one.starts[0, 1] == nq15.base


def test_alignment_enforces_exact_timeframe_pairs_instead_of_all_contexts():
    layout = _layout()
    nq3 = _stream(layout, "NQ@3min")
    nq15 = _stream(layout, "NQ@15min")
    # The 3m decision at minute 18 may use completed 15m context, but must not receive otherwise
    # available 1m or 5m context under the declared 3m<->15m / 1m<->5m experiment.
    plan = layout.align(
        [nq3.base + 5], 1, related_tfs=("1min", "3min", "5min", "15min"),
        tf_pairs="1min=5min,3min=15min", siblings="0")
    roles = {name: i for i, name in enumerate(plan.role_names)}
    assert plan.mask[0, roles["same_ticker@15min"]]
    assert plan.starts[0, roles["same_ticker@15min"]] == nq15.base
    assert not plan.mask[0, roles["same_ticker@1min"]]
    assert not plan.mask[0, roles["same_ticker@5min"]]


def test_alignment_never_uses_future_sibling_and_masks_stale_context():
    records = [
        {"sid": "NQ@3min", "ticker": "NQ", "tf": "3min", "base": 0,
         "ts": _timestamps([0, 3, 6, 9])},
        # ES has only a future-stamped bar relative to NQ's first decision.
        {"sid": "ES@3min", "ticker": "ES", "tf": "3min", "base": 4,
         "ts": _timestamps([9, 12, 15, 18])},
    ]
    layout = RelatedSeriesLayout.from_assembled(records)
    plan = layout.align([0], 1, related_tfs=(), siblings={"NQ": "ES"})
    assert not plan.mask[0, 1]

    # A past sibling may also be too stale; increasing tolerance makes the same causal bar usable.
    records[1] = {**records[1], "ts": _timestamps([-30, -27, -24, -21])}
    stale = RelatedSeriesLayout.from_assembled(records)
    assert not stale.align([0], 1, related_tfs=(), siblings={"NQ": "ES"},
                           max_gap_factor=2).mask[0, 1]
    assert stale.align([0], 1, related_tfs=(), siblings={"NQ": "ES"},
                       max_gap_factor=20).mask[0, 1]


def test_alignment_rejects_primary_window_crossing_stream_boundary():
    layout = _layout()
    primary = _stream(layout, "NQ@3min")
    with pytest.raises(ValueError, match="crosses"):
        layout.align([primary.end - 1], 2, related_tfs=(), siblings="0")


def test_ssl_assemble_returns_related_metadata_without_changing_default_contract():
    from futures_foundation.finetune.ssl import assemble
    from futures_foundation.finetune.ssl_data import WindowStartPool

    ts = _timestamps(np.arange(30))
    streams = [{"sid": "NQ@1min", "ticker": "NQ", "tf": "1min", "ts": ts,
                "ohlcv": np.arange(150, dtype=np.float32).reshape(30, 5)}]
    plain = assemble(streams, seq=3, max_jitter=1, val_frac=0.2,
                     holdout_start="2026-01-01", verbose=False)
    related = assemble(streams, seq=3, max_jitter=1, val_frac=0.2,
                       holdout_start="2026-01-01", verbose=False,
                       return_related_layout=True)
    assert len(plain) == 3 and len(related) == 4
    for left, right in zip(plain, related[:3]):
        assert np.array_equal(left, right)
    assert related[3].streams[0].sid == "NQ@1min"

    uniform = assemble(streams, seq=3, max_jitter=1, val_frac=0.2,
                       holdout_start="2026-01-01", verbose=False,
                       sampling_mode="uniform_stream", return_related_layout=True)
    assert isinstance(uniform[1], WindowStartPool)
    assert np.array_equal(np.asarray(uniform[1]), plain[1])
    assert uniform[3].streams[0].sid == "NQ@1min"


def test_fusion_is_exact_primary_at_initialization_and_ignores_masked_members():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.pretext._torch.related_series import RelatedSeriesFusion

    torch.manual_seed(4)
    fusion = RelatedSeriesFusion(8, num_roles=3, num_heads=2).eval()
    embeddings = torch.randn(5, 3, 8)
    mask = torch.tensor([[1, 1, 0]] * 5, dtype=torch.bool)
    roles = torch.arange(3)
    assert torch.equal(fusion(embeddings, mask, roles), embeddings[:, 0])

    with torch.no_grad():
        fusion.gate.fill_(1.0)
    first = fusion(embeddings, mask, roles)
    changed_masked = embeddings.clone()
    changed_masked[:, 2] = 1e6
    assert torch.allclose(first, fusion(changed_masked, mask, roles), atol=0, rtol=0)
    changed_related = embeddings.clone()
    changed_related[:, 1] += 10
    assert not torch.allclose(first, fusion(changed_related, mask, roles))


def test_fusion_does_not_mix_unrelated_batch_examples():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.pretext._torch.related_series import RelatedSeriesFusion

    torch.manual_seed(8)
    fusion = RelatedSeriesFusion(8, num_roles=2, num_heads=2).eval()
    with torch.no_grad():
        fusion.gate.fill_(0.7)
    values = torch.randn(2, 2, 8)
    mask = torch.ones(2, 2, dtype=torch.bool)
    before = fusion(values, mask)
    values[1] += 1000
    after = fusion(values, mask)
    assert torch.allclose(before[0], after[0], atol=0, rtol=0)
    assert not torch.allclose(before[1], after[1])


def test_related_controls_never_corrupt_primary_context():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.pretext._torch.related_nextleg import _apply_related_control

    contexts = torch.arange(2 * 3 * 2 * 8, dtype=torch.float32).reshape(2, 3, 2, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    for mode in ("shuffle", "random", "drop"):
        changed, changed_mask = _apply_related_control(contexts, mask, mode)
        assert torch.equal(changed[:, 0], contexts[:, 0])
        assert changed_mask[:, 0].all()
    shuffled, _ = _apply_related_control(contexts, mask, "shuffle")
    assert not torch.equal(shuffled[:, 1:], contexts[:, 1:])
    dropped, dropped_mask = _apply_related_control(contexts, mask, "drop")
    assert torch.equal(dropped, contexts)
    assert not dropped_mask[:, 1:].any()
    with pytest.raises(ValueError, match="related_control"):
        _apply_related_control(contexts, mask, "future")


def test_related_checkpoint_round_trip():
    torch = pytest.importorskip("torch")
    from torch import nn
    from futures_foundation.finetune.pretext._torch.related_series import (
        RelatedMantisEncoder, is_related_checkpoint, load_related_checkpoint,
        plain_encoder_state, related_checkpoint_state)

    class TinyMantis(nn.Module):
        hidden_dim, seq_len = 4, 6

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, self.hidden_dim)

        def forward(self, values):
            return self.proj(values.mean(-1, keepdim=True).squeeze(1))

    source = RelatedMantisEncoder(TinyMantis(), channels=2, num_roles=3, num_heads=2)
    with torch.no_grad():
        source.fusion.gate.fill_(0.5)
    state = related_checkpoint_state(source)
    assert is_related_checkpoint(state)
    assert set(plain_encoder_state(state)) == set(source.mantis.state_dict())

    target = RelatedMantisEncoder(TinyMantis(), channels=2, num_roles=3, num_heads=2)
    load_related_checkpoint(target, state)
    windows = torch.randn(3, 3, 2, 6)
    mask = torch.ones(3, 3, dtype=torch.bool)
    assert torch.allclose(source(windows, mask), target(windows, mask))
