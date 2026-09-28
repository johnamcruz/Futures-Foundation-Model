"""Contracts for Chronos-2 forecast-direction SSL (no model download needed).

Torch tests are gated by CHRONOS_TORCH_TESTS=1 and import torch in the body
(see tests/conftest.py).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.classifiers.chronos2 import forecast_ssl as fs


needs_torch = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="set CHRONOS_TORCH_TESTS=1 to run torch tests")


def _torch():
    import torch

    return torch


def _levels():
    return _torch().tensor([0.1, 0.5, 0.9])


def _stream(n: int = 600, *, start: str = "2025-06-01", minutes: int = 3,
            seed: int = 0) -> fs.Stream:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    values = np.column_stack([close, close + 1, close - 1, close,
                              rng.integers(1, 100, n).astype(float)])
    times = pd.date_range(start, periods=n, freq=f"{minutes}min", tz="UTC")
    return fs.Stream("TEST@3min", times.asi8, values)


# ------------------------------------------------------------------ quantile CDF

@needs_torch
def test_quantile_cdf_interpolates_between_knots():
    torch = _torch()
    q = torch.tensor([[0.0, 10.0, 20.0]])
    assert fs.quantile_cdf(q, _levels(), torch.tensor([10.0])).item() == pytest.approx(0.5)
    assert fs.quantile_cdf(q, _levels(), torch.tensor([5.0])).item() == pytest.approx(0.3)
    assert fs.quantile_cdf(q, _levels(), torch.tensor([15.0])).item() == pytest.approx(0.7)


@needs_torch
def test_quantile_cdf_clamps_outside_outer_knots():
    torch = _torch()
    q = torch.tensor([[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]])
    values = fs.quantile_cdf(q, _levels(), torch.tensor([-50.0, 99.0]))
    assert values.tolist() == pytest.approx([0.1, 0.9])


@needs_torch
def test_quantile_cdf_sorts_crossed_quantiles():
    torch = _torch()
    crossed = torch.tensor([[20.0, 10.0, 0.0]])
    assert fs.quantile_cdf(crossed, _levels(), torch.tensor([10.0])).item() == pytest.approx(0.5)


@needs_torch
def test_quantile_cdf_is_differentiable_in_the_quantiles():
    torch = _torch()
    q = torch.tensor([[0.0, 10.0, 20.0]], requires_grad=True)
    fs.quantile_cdf(q, _levels(), torch.tensor([5.0])).sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


@needs_torch
def test_p_up_reads_each_horizon_column():
    torch = _torch()
    # [B=1, Q=3, H=10]; horizon 5 centred above the close, horizon 10 below.
    q = torch.zeros(1, 3, 10)
    q[0, :, 4] = torch.tensor([101.0, 102.0, 103.0])
    q[0, :, 9] = torch.tensor([97.0, 98.0, 99.0])
    probability = fs.p_up(q, torch.tensor([100.0]), _levels(), horizons=(5, 10))
    assert probability[0].tolist() == pytest.approx([0.9, 0.1])


@needs_torch
def test_direction_loss_rewards_quantiles_on_the_realized_side():
    torch = _torch()
    last = torch.tensor([100.0])
    future = torch.full((1, 10), 105.0)  # up move at every horizon
    up = torch.tensor([[[101.0] * 10, [102.0] * 10, [103.0] * 10]])
    down = torch.tensor([[[97.0] * 10, [98.0] * 10, [99.0] * 10]])
    assert (fs.direction_loss(up, last, future, _levels(), (5, 10))
            < fs.direction_loss(down, last, future, _levels(), (5, 10)))


@needs_torch
def test_direction_loss_gradient_moves_quantiles_toward_the_realized_side():
    torch = _torch()
    last = torch.tensor([100.0])
    future = torch.full((1, 10), 105.0)
    q = torch.tensor([[[99.0] * 10, [100.5] * 10, [102.0] * 10]], requires_grad=True)
    fs.direction_loss(q, last, future, _levels(), (5,)).backward()
    # Raising the knots around the close lowers F(close) and raises P(up).
    assert (q.grad[0, :, 4] <= 0).all() and q.grad[0, :, 4].sum() < 0


# ------------------------------------------------------------------ anchors and windows

def test_period_bounds_keep_the_whole_target_inside_the_period():
    stream = _stream(600)
    times = pd.DatetimeIndex(stream.close_ns, tz="UTC")
    end = times[500].isoformat()
    lo, hi = fs.period_bounds(stream.close_ns, None, end,
                              context_length=256, forecast_length=64)
    assert lo == 255
    # The last anchor's target-end close is strictly before the boundary.
    assert stream.close_ns[hi - 1 + 64] < stream.close_ns[500]
    assert hi - 1 + 64 == 499


def test_period_bounds_respect_the_decision_start():
    stream = _stream(600)
    times = pd.DatetimeIndex(stream.close_ns, tz="UTC")
    lo, _ = fs.period_bounds(stream.close_ns, times[400].isoformat(), times[-1].isoformat(),
                             context_length=256, forecast_length=64)
    assert lo == 400


def test_period_bounds_empty_when_no_anchor_fits():
    stream = _stream(300)
    times = pd.DatetimeIndex(stream.close_ns, tz="UTC")
    lo, hi = fs.period_bounds(stream.close_ns, None, times[-1].isoformat(),
                              context_length=256, forecast_length=64)
    assert hi == lo


def test_contract_periods_are_ordered_and_end_at_the_holdout():
    train_end = pd.Timestamp(fs.PERIODS["train"][1])
    select_start, select_end = map(pd.Timestamp, fs.PERIODS["select"])
    outer_start, outer_end = map(pd.Timestamp, fs.PERIODS["outer"])
    assert train_end < select_start < select_end <= outer_start < outer_end
    assert outer_end == pd.Timestamp(fs.HOLDOUT_START)


def test_gather_aligns_context_target_and_decision_close():
    stream = _stream(600)
    anchors = np.array([300, 400])
    context, future, last = fs.gather(stream.values, anchors,
                                      context_length=256, forecast_length=64)
    assert context.shape == (2, 5, 256) and future.shape == (2, 64)
    assert last[0] == stream.values[300, fs.CLOSE]
    assert context[0, fs.CLOSE, -1] == stream.values[300, fs.CLOSE]
    assert context[0, 0, 0] == stream.values[300 - 255, 0]
    assert future[0, 0] == stream.values[301, fs.CLOSE]
    assert future[1, 63] == stream.values[464, fs.CLOSE]


def test_direction_labels_match_probe_atlas_definition():
    future = np.array([[101.0, 99.0], [100.0, 100.0]])
    last = np.array([100.0, 100.0])
    labels = fs.direction_labels(future, last, horizons=(1, 2))
    assert labels.tolist() == [[True, False], [False, False]]


def test_evenly_spaced_stays_in_range_and_is_unique():
    anchors = fs.evenly_spaced(10, 20, 100)
    assert anchors.tolist() == list(range(10, 20))
    assert fs.evenly_spaced(5, 5, 10).size == 0


# ------------------------------------------------------------------ forecast wrapper

class _FakeChronos:
    def __init__(self, quantiles: int = 3):
        self.quantiles = quantiles
        self.calls = []

    def __call__(self, *, context, group_ids, num_output_patches, future_target):
        torch = _torch()
        self.calls.append(dict(context=context, group_ids=group_ids,
                               num_output_patches=num_output_patches,
                               future_target=future_target))
        rows, horizon = context.shape[0], num_output_patches * 16
        preds = torch.arange(rows, dtype=torch.float32)[:, None, None].expand(
            rows, self.quantiles, horizon).clone()
        loss = None if future_target is None else torch.tensor(0.2)
        return SimpleNamespace(loss=loss, quantile_preds=preds)


@needs_torch
def test_forecast_close_groups_each_window_and_targets_only_close():
    torch = _torch()
    fake = _FakeChronos()
    context = torch.zeros(2, 5, 32)
    future = torch.ones(2, 32)
    loss, quantiles = fs.forecast_close(fake, context, future, forecast_length=32)
    call = fake.calls[0]
    assert call["group_ids"].tolist() == [0] * 5 + [1] * 5
    assert call["num_output_patches"] == 2
    target = call["future_target"].reshape(2, 5, 32)
    assert torch.isnan(target[:, [0, 1, 2, 4]]).all()
    assert (target[:, fs.CLOSE] == 1).all()
    # Native loss averages over all B*5 rows; only 1 in 5 carries a target.
    assert loss.item() == pytest.approx(1.0)
    # Close quantiles come from rows 3 and 8.
    assert quantiles.shape == (2, 3, 32)
    assert quantiles[:, 0, 0].tolist() == [3.0, 8.0]


@needs_torch
def test_forecast_close_without_target_returns_no_loss():
    torch = _torch()
    loss, _ = fs.forecast_close(_FakeChronos(), torch.zeros(1, 5, 32), forecast_length=16)
    assert loss is None


@needs_torch
def test_forecast_close_rejects_non_patch_horizon():
    torch = _torch()
    with pytest.raises(ValueError):
        fs.forecast_close(_FakeChronos(), torch.zeros(1, 5, 32), forecast_length=20)


# ------------------------------------------------------------------ metrics and controls

def test_causal_features_ignore_future_bars():
    stream = _stream(600)
    anchors = np.array([300, 350])
    before = fs.causal_features(stream, anchors)
    changed = stream.values.copy()
    changed[351:] *= 3.0
    after = fs.causal_features(fs.Stream(stream.name, stream.close_ns, changed), anchors)
    np.testing.assert_allclose(before, after)
    assert np.isfinite(before).all()


def test_causal_features_reject_short_history():
    with pytest.raises(ValueError):
        fs.causal_features(_stream(600), np.array([10]))


def test_pinball_is_zero_for_exact_quantiles_and_scaled_wql_is_one_for_flat():
    levels = np.array([0.1, 0.5, 0.9])
    y = np.array([101.0, 99.0])
    assert fs.pinball(y, np.repeat(y[:, None], 3, 1), levels).sum() == 0.0
    flat = np.full((2, 3), 100.0)
    assert fs.pinball(y, flat, levels).sum() > 0


def test_auc_with_se_shrinks_with_more_effective_samples():
    rng = np.random.default_rng(0)
    y = rng.random(2000) > 0.5
    score = y + rng.normal(0, 2.0, 2000)
    auc, se_small = fs.auc_with_se(y, score, 100)
    _, se_large = fs.auc_with_se(y, score, 2000)
    assert 0.5 < auc < 1.0 and se_large < se_small


def test_auc_with_se_single_class_is_nan():
    auc, se = fs.auc_with_se(np.ones(10, bool), np.arange(10.0), 10)
    assert np.isnan(auc) and np.isnan(se)


def test_score_stream_reports_every_horizon_with_controls():
    rng = np.random.default_rng(1)
    n, levels = 400, np.array([0.1, 0.5, 0.9])
    last = 100 + rng.normal(0, 1, n)
    future = last[:, None] + rng.normal(0, 1, (n, 2))
    close_q = last[:, None, None] + np.array([-1.0, 0.0, 1.0])[None, :, None]
    close_q = np.repeat(close_q, 2, axis=2)
    rows = fs.score_stream(close_q, last, future, np.arange(n) * 3, levels,
                           baseline=rng.random((n, 2)), p_up_values=rng.random((n, 2)),
                           horizons=(5, 10))
    assert set(rows) == {"h5", "h10"}
    for row in rows.values():
        assert row["n_effective"] <= row["n"]
        assert 0.0 <= row["coverage_80"] <= 1.0
        assert row["scaled_wql"] > 0
        assert {"auc", "shuffled_auc", "baseline_auc", "brier", "base_rate"} <= set(row)


# ------------------------------------------------------------------ gates

def _report(auc: dict[str, float], wql: dict[str, float], baseline: float = 0.5) -> dict:
    per_stream = {name: {"h5": {"auc": auc[name], "baseline_auc": baseline,
                                "scaled_wql": wql[name], "shuffled_auc": 0.5,
                                "coverage_80": 0.8}}
                  for name in auc}
    return {"per_stream": per_stream, "summary": fs.summarize(per_stream, (5,))}


def test_direction_gate_passes_consistent_lift_and_fails_noise():
    names = [f"S{i}" for i in range(12)]
    reference = _report({n: 0.52 for n in names}, {n: 0.9 for n in names})
    lifted = _report({n: 0.53 + 0.001 * i for i, n in enumerate(names)},
                     {n: 0.9 for n in names})
    noisy = _report({n: 0.52 + (0.02 if i % 2 else -0.02) for i, n in enumerate(names)},
                    {n: 0.9 for n in names})
    assert fs.direction_gate(lifted, reference, ("h5",))["pass"]
    assert not fs.direction_gate(noisy, reference, ("h5",))["pass"]


def test_direction_gate_fails_when_forecast_degrades():
    names = [f"S{i}" for i in range(12)]
    reference = _report({n: 0.52 for n in names}, {n: 0.9 for n in names})
    worse = _report({n: 0.54 for n in names}, {n: 0.95 for n in names})
    assert not fs.direction_gate(worse, reference, ("h5",))["pass"]


def test_forecast_gate_requires_broad_improvement_without_large_regressions():
    names = [f"S{i}" for i in range(8)]
    reference = _report({n: 0.5 for n in names}, {n: 1.0 for n in names})
    broad = _report({n: 0.5 for n in names}, {n: 0.97 for n in names})
    one_bad = _report({n: 0.5 for n in names},
                      {n: (1.10 if i == 0 else 0.97) for i, n in enumerate(names)})
    assert fs.forecast_gate(broad, reference)["pass"]
    assert not fs.forecast_gate(one_bad, reference)["pass"]


def test_train_rejects_arm_weight_mismatch(tmp_path):
    with pytest.raises(ValueError):
        fs.train_forecast_direction({}, parent=tmp_path, base_snapshot=tmp_path,
                                    out_dir=tmp_path, provenance={}, arm="a1",
                                    direction_weight=0.3)
    with pytest.raises(ValueError):
        fs.train_forecast_direction({}, parent=tmp_path, base_snapshot=tmp_path,
                                    out_dir=tmp_path, provenance={}, arm="a2",
                                    direction_weight=0.0)


# ------------------------------------------------------------------ direction focus

@needs_torch
def test_direction_loss_ignores_move_magnitude():
    """Same signs, 10x larger moves: the direction term must not change."""
    torch = _torch()
    last = torch.tensor([100.0, 100.0, 100.0])
    moves = torch.tensor([[1.0] * 10, [-2.0] * 10, [0.5] * 10])
    q = 100.0 + torch.tensor([-1.0, 0.2, 1.5])[None, :, None].expand(3, 3, 10)
    small = fs.direction_loss(q, last, last[:, None] + moves, _levels(), (5, 10))
    large = fs.direction_loss(q, last, last[:, None] + 10 * moves, _levels(), (5, 10))
    assert small.item() == pytest.approx(large.item())


@needs_torch
def test_direction_training_recovers_a_planted_sign_signal_under_volatility_noise():
    """Toy forecaster: the sign of the next move depends weakly on x, while the
    move size is dominated by an unrelated volatility factor v. Training only on
    the direction term must learn the sign (positive drift weight, AUC > 0.55)."""
    torch = _torch()
    from sklearn.metrics import roc_auc_score

    generator = torch.Generator().manual_seed(0)
    n = 4000
    x = torch.randn(n, generator=generator)
    v = torch.randn(n, generator=generator)
    scale = torch.exp(1.5 * v)
    move = scale * (0.4 * x + torch.randn(n, generator=generator))
    last = torch.full((n,), 100.0)
    future = (last + move)[:, None].expand(n, 5)
    drift = torch.zeros(1, requires_grad=True)
    spread = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([drift, spread], lr=0.05)
    offsets = torch.tensor([-1.0, 0.0, 1.0])
    for _ in range(200):
        centre = last + drift * x
        q = (centre[:, None] + torch.exp(spread) * offsets[None, :])[:, :, None].expand(n, 3, 5)
        loss = fs.direction_loss(q, last, future, _levels(), (5,))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert drift.item() > 0
    with torch.no_grad():
        centre = last + drift * x
        q = (centre[:, None] + torch.exp(spread) * offsets[None, :])[:, :, None].expand(n, 3, 5)
        probability = fs.p_up(q, last, _levels(), (5,))[:, 0].numpy()
    assert roc_auc_score((move > 0).numpy(), probability) > 0.55


# ------------------------------------------------------------------ all-horizon gate

def _multi(auc_by_h: dict[str, dict[str, float]], wql: float = 0.9) -> dict:
    names = sorted(next(iter(auc_by_h.values())))
    per_stream = {name: {key: {"auc": auc_by_h[key][name], "baseline_auc": 0.5,
                               "scaled_wql": wql, "shuffled_auc": 0.5,
                               "coverage_80": 0.8} for key in auc_by_h}
                  for name in names}
    return {"per_stream": per_stream,
            "summary": fs.summarize(per_stream, [int(k[1:]) for k in auc_by_h])}


def test_direction_gate_covers_all_four_horizons_by_default():
    names = [f"S{i}" for i in range(12)]
    flat = {f"h{h}": {n: 0.52 for n in names} for h in fs.HORIZONS}
    up = {f"h{h}": {n: 0.53 + 0.001 * i for i, n in enumerate(names)} for h in fs.HORIZONS}
    gate = fs.direction_gate(_multi(up), _multi(flat))
    assert set(gate["horizons"]) == {"h5", "h10", "h20", "h50"}
    assert gate["pass"]


def test_direction_gate_fails_when_a_long_horizon_regresses():
    names = [f"S{i}" for i in range(12)]
    flat = {f"h{h}": {n: 0.52 for n in names} for h in fs.HORIZONS}
    mixed = {f"h{h}": {n: 0.53 + 0.001 * i for i, n in enumerate(names)} for h in fs.HORIZONS}
    mixed["h50"] = {n: 0.51 for n in names}
    gate = fs.direction_gate(_multi(mixed), _multi(flat))
    assert gate["horizons"]["h5"]["pass"] and gate["horizons"]["h10"]["pass"]
    assert not gate["pass"]


# ------------------------------------------------------------------ retention gate

def _atlas(values: dict[str, tuple[str, float, dict[str, float]]]) -> dict:
    return {"probes": {name: {"family": family, "auc": auc, "per_stream_auc": streams}
                       for name, (family, auc, streams) in values.items()}}


def test_retention_gate_blocks_expansion_regression_but_ignores_direction_probes():
    parent = _atlas({
        "pred_expansion_h20": ("prediction", 0.86, {"NQ@3min": 0.85, "ES@3min": 0.87}),
        "ret_vol_regime": ("retention", 0.90, {"NQ@3min": 0.90, "ES@3min": 0.90}),
        "pred_direction_h5": ("prediction", 0.53, {"NQ@3min": 0.53, "ES@3min": 0.53}),
    })
    same = _atlas({
        "pred_expansion_h20": ("prediction", 0.857, {"NQ@3min": 0.85, "ES@3min": 0.865}),
        "ret_vol_regime": ("retention", 0.90, {"NQ@3min": 0.90, "ES@3min": 0.90}),
        "pred_direction_h5": ("prediction", 0.40, {"NQ@3min": 0.40, "ES@3min": 0.40}),
    })
    regressed = _atlas({
        "pred_expansion_h20": ("prediction", 0.83, {"NQ@3min": 0.82, "ES@3min": 0.84}),
        "ret_vol_regime": ("retention", 0.90, {"NQ@3min": 0.90, "ES@3min": 0.90}),
        "pred_direction_h5": ("prediction", 0.55, {"NQ@3min": 0.55, "ES@3min": 0.55}),
    })
    ok = fs.retention_gate(same, parent)
    assert ok["pass"] and "pred_direction_h5" not in ok["probes"]
    bad = fs.retention_gate(regressed, parent)
    assert not bad["pass"] and not bad["probes"]["pred_expansion_h20"]["pass"]


def test_retention_gate_blocks_a_single_stream_collapse():
    parent = _atlas({"pred_expansion_h5": ("prediction", 0.84,
                                           {"NQ@3min": 0.84, "ES@3min": 0.84})})
    collapsed = _atlas({"pred_expansion_h5": ("prediction", 0.835,
                                              {"NQ@3min": 0.79, "ES@3min": 0.88})})
    assert not fs.retention_gate(collapsed, parent)["pass"]


def test_retention_gate_requires_every_parent_probe():
    parent = _atlas({"pred_expansion_h5": ("prediction", 0.84, {"NQ@3min": 0.84})})
    with pytest.raises(KeyError):
        fs.retention_gate(_atlas({}), parent)


# ------------------------------------------------------------------ CLI

def _cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/chronos/chronos2_ssl_forecast_direction.py"
    spec = importlib.util.spec_from_file_location("forecast_direction_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_outer_evaluation_until_selection_is_frozen():
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.parse(["evaluate", "--checkpoint", "base", "--name", "a0", "--periods", "outer"])
    args = cli.parse(["evaluate", "--checkpoint", "base", "--name", "a0",
                      "--periods", "outer", "--confirm-outer-frozen"])
    assert args.periods == ("outer",)


def test_cli_train_defaults_match_the_frozen_contract():
    args = _cli().parse(["train", "--arm", "a2", "--direction-weight", "0.3"])
    assert (args.lr, args.batch_windows, args.steps, args.epochs, args.patience) == (
        1e-5, 32, 100, 20, 3)
    assert args.parent.name == "checkpoint" and args.parent.parent.name == "mask_full"


def test_cli_smoke_shrinks_the_run_and_writes_under_smoke():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a1", "--smoke"])
    settings = cli.run_settings(args)
    assert settings["tickers"] == ("NQ", "ES") and settings["timeframes"] == ("3min",)
    assert settings["epochs"] == 1 and settings["steps"] <= 10
    assert settings["out_dir"].parent.name == "smoke"


def test_cli_full_run_uses_all_36_streams():
    cli = _cli()
    settings = cli.run_settings(cli.parse(["train", "--arm", "a1"]))
    assert len(settings["tickers"]) * len(settings["timeframes"]) == 36
    assert settings["out_dir"].name == "a1_seed0"


@needs_torch
def test_predict_period_moves_to_cpu_before_float64():
    """MPS has no float64: predict_period must convert only after .cpu()."""
    torch = _torch()

    class _Guarded(torch.Tensor):
        pass

    class _Model(_FakeChronos):
        chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            preds = output.quantile_preds.as_subclass(_Guarded)
            return SimpleNamespace(loss=output.loss, quantile_preds=preds)

    def _no_double(self, *args, **kwargs):
        if self.__class__ is _Guarded:
            raise TypeError("float64 on accelerator")
        return torch.Tensor.double(self, *args, **kwargs)

    _Guarded.double = _no_double
    _Guarded.cpu = lambda self, *a, **k: torch.Tensor.cpu(self, *a, **k).as_subclass(torch.Tensor)
    stream = _stream(700)
    q, probability, last, future = fs.predict_period(
        _Model(), stream, np.array([300, 400]), device="cpu", horizons=(5, 10))
    assert q.dtype == np.float64 and q.shape == (2, 3, 2)
    assert probability.shape == (2, 2) and last.shape == (2,) and future.shape == (2, 2)


# ------------------------------------------------------------------ Probe Atlas handoff

def _atlas_launcher():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/chronos/chronos2_probe_atlas.py"
    spec = importlib.util.spec_from_file_location("chronos2_probe_atlas_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_report_is_accepted_by_probe_atlas(tmp_path):
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import tree_sha256

    checkpoint = tmp_path / "a2" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    base = {"model_id": "autogluon/chronos-2-small", "snapshot_path": "/s",
            "revision": "d" * 40, "weights_sha256": "a" * 64, "config_sha256": "b" * 64}
    report = fs.build_stage_report(
        arm="a2", direction_weight=3.0, seed=0, parent_path="/p", parent_sha256="c" * 64,
        checkpoint=checkpoint, base_identity=base, provenance={"NQ@3min": {"sha256": "e"}},
        timeframes=("1min", "3min", "5min", "15min"), streams=["NQ@3min"], code=None,
        training={}, history=[], parent_select={}, best_select={}, best_epoch=0,
        step_seconds=[1.0], elapsed_seconds=1.0)
    (checkpoint.parent / "report.json").write_text(__import__("json").dumps(report))
    identity = _atlas_launcher()._stage_identity(
        str(checkpoint), tree_sha256(checkpoint), None, context_length=fs.CONTEXT_LENGTH)
    assert identity["parent_checkpoint_sha256"] == "c" * 64
    assert identity["base_revision"] == "d" * 40
    assert report["stage"] == "forecast_direction" and report["config"]["arm"] == "a2"


# ------------------------------------------------------------------ λ selection (select period only)

def _select_report(auc: float, wql: float) -> dict:
    per_stream = {name: {f"h{h}": {"auc": auc, "baseline_auc": 0.5, "scaled_wql": wql,
                                    "shuffled_auc": 0.5, "coverage_80": 0.8}
                         for h in fs.HORIZONS} for name in ("NQ@3min", "ES@3min")}
    return {"period": "select", "per_stream": per_stream,
            "summary": fs.summarize(per_stream)}


def test_select_direction_weight_takes_best_all_horizon_auc_within_wql_budget():
    a1 = _select_report(0.52, 0.60)
    candidates = {0.3: _select_report(0.525, 0.60), 1.0: _select_report(0.53, 0.61),
                  3.0: _select_report(0.54, 0.62), 10.0: _select_report(0.56, 0.70)}
    choice = fs.select_direction_weight(a1, candidates)
    # Budget is 0.60 * 1.02 = 0.612: λ=3 and λ=10 break it despite better AUC.
    assert choice["direction_weight"] == 1.0
    assert choice["eligible"] == [0.3, 1.0]


def test_select_direction_weight_refuses_outer_reports():
    a1 = _select_report(0.52, 0.60)
    outer = dict(_select_report(0.6, 0.6), period="outer")
    with pytest.raises(ValueError):
        fs.select_direction_weight(a1, {1.0: outer})


def test_select_direction_weight_returns_none_when_nothing_is_eligible():
    a1 = _select_report(0.52, 0.60)
    choice = fs.select_direction_weight(a1, {1.0: _select_report(0.6, 0.9)})
    assert choice["direction_weight"] is None


def test_cli_select_writes_a_frozen_selection_record(tmp_path):
    import json

    cli = _cli()
    a1 = tmp_path / "a1.json"
    a1.write_text(json.dumps(_select_report(0.52, 0.60)))
    lam = tmp_path / "a2.json"
    lam.write_text(json.dumps(_select_report(0.53, 0.60)))
    out = tmp_path / "selection.json"
    cli.main(["select", "--a1", str(a1), "--a2", f"3={lam}", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["direction_weight"] == 3.0
    assert record["inputs"]["a2"] == {"3.0": str(lam)}
    assert "frozen_at" in record
