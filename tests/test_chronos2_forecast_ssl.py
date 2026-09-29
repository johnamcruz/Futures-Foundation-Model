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


# ------------------------------------------------------------------ A3 bar-structure inputs

def _bars(rows):
    """rows of (open, high, low, close, volume)."""
    return np.asarray(rows, dtype=float)


def test_bar_structure_describes_how_each_bar_closed():
    values = _bars([(10, 12, 8, 11, 100)] * 30 + [(10, 14, 10, 14, 300)])
    features = fs.bar_structure(values)
    assert features.shape == (31, len(fs.BAR_FEATURES))
    last = dict(zip(fs.BAR_FEATURES, features[-1]))
    assert last["close_location"] == pytest.approx(1.0)      # closed on the high
    assert last["body"] == pytest.approx(1.0)                # open 10 -> close 14 over range 4
    assert last["upper_wick"] == pytest.approx(0.0)
    assert last["lower_wick"] == pytest.approx(0.0)
    assert last["relative_volume"] == pytest.approx(np.log(3.0))
    first = dict(zip(fs.BAR_FEATURES, features[0]))
    assert first["close_location"] == pytest.approx(0.75)
    assert first["upper_wick"] == pytest.approx(0.25) and first["lower_wick"] == pytest.approx(0.5)


def test_bar_structure_handles_zero_range_bars():
    values = _bars([(10, 10, 10, 10, 50)] * 25)
    features = fs.bar_structure(values)
    row = dict(zip(fs.BAR_FEATURES, features[-1]))
    assert row["close_location"] == 0.5 and row["body"] == 0.0
    assert row["upper_wick"] == 0.0 and row["lower_wick"] == 0.0
    assert np.isfinite(features[-1]).all()


def test_bar_structure_never_reads_future_bars():
    stream = _stream(400)
    base = fs.bar_structure(stream.values)
    changed = stream.values.copy()
    changed[301:] = changed[301:] * 1.7 + 5.0
    after = fs.bar_structure(changed)
    np.testing.assert_allclose(base[:301], after[:301], equal_nan=True)


def test_bar_structure_marks_warmup_as_missing_not_invented():
    features = fs.bar_structure(_stream(100).values)
    scaled = [fs.BAR_FEATURES.index("scaled_return"), fs.BAR_FEATURES.index("relative_volume")]
    assert np.isnan(features[:20, scaled]).all()
    assert np.isfinite(features[25:]).all()


def test_gather_appends_bar_features_after_ohlcv():
    stream = _stream(600)
    features = fs.bar_structure(stream.values)
    context, future, last = fs.gather(stream.values, np.array([300]), features=features,
                                      context_length=256, forecast_length=64)
    assert context.shape == (1, 5 + len(fs.BAR_FEATURES), 256)
    np.testing.assert_allclose(context[0, 5:, -1], features[300])
    assert context[0, fs.CLOSE, -1] == stream.values[300, fs.CLOSE]
    assert future[0, 0] == stream.values[301, fs.CLOSE]


@needs_torch
def test_forecast_close_targets_only_close_with_extra_input_series():
    torch = _torch()
    fake = _FakeChronos()
    context = torch.zeros(2, 5 + len(fs.BAR_FEATURES), 32)
    loss, quantiles = fs.forecast_close(fake, context, torch.ones(2, 32), forecast_length=32)
    channels = 5 + len(fs.BAR_FEATURES)
    target = fake.calls[0]["future_target"].reshape(2, channels, 32)
    assert (target[:, fs.CLOSE] == 1).all()
    others = [i for i in range(channels) if i != fs.CLOSE]
    assert torch.isnan(target[:, others]).all()
    assert fake.calls[0]["group_ids"].tolist() == [0] * channels + [1] * channels
    assert quantiles[:, 0, 0].tolist() == [3.0, 3.0 + channels]
    assert loss.item() == pytest.approx(0.2 * channels)


def test_a3_requires_a_direction_term(tmp_path):
    with pytest.raises(ValueError):
        fs.train_forecast_direction({}, parent=tmp_path, base_snapshot=tmp_path,
                                    out_dir=tmp_path, provenance={}, arm="a3",
                                    direction_weight=0.0)


def test_only_a3_uses_bar_structure_inputs():
    assert fs.uses_bar_features("a3")
    assert not fs.uses_bar_features("a1") and not fs.uses_bar_features("a2")


def test_cli_a3_run_directory_and_eval_flag():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a3", "--direction-weight", "3"])
    assert cli.run_settings(args)["out_dir"].name == "a3_lam3_seed0"
    evaluate = cli.parse(["evaluate", "--checkpoint", "x", "--name", "a3", "--bar-features"])
    assert evaluate.bar_features


@needs_torch
def test_predict_period_feeds_bar_features_to_the_model():
    torch = _torch()

    class _Model(_FakeChronos):
        chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    stream = _stream(700)
    model = _Model()
    fs.predict_period(model, stream, np.array([300, 400]), device="cpu", horizons=(5,),
                      features=fs.bar_structure(stream.values))
    channels = 5 + len(fs.BAR_FEATURES)
    assert model.calls[0]["context"].shape == (2 * channels, 256)
    assert torch.isfinite(model.calls[0]["context"]).all()


# ------------------------------------------------------------------ direction where a big move is expected

def test_score_stream_reports_direction_on_the_widest_predicted_ranges():
    rng = np.random.default_rng(3)
    n, levels = 1000, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    width = rng.uniform(0.5, 5.0, n)                  # model's predicted range per row
    wide = width >= np.quantile(width, 0.8)
    sign = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    future = (last + sign * width)[:, None]           # one horizon
    # P(up) is informative only on the wide rows, random elsewhere.
    p_up = np.where(wide, (sign > 0) * 0.8 + 0.1, rng.random(n))[:, None]
    close_q = (last[:, None] + np.outer(width, [-1.0, 0.0, 1.0]))[:, :, None]
    rows = fs.score_stream(close_q, last, future, np.arange(n) * 5, levels,
                           baseline=rng.random((n, 1)), p_up_values=p_up, horizons=(5,))
    slice_ = rows["h5"]["expansion_slice"]
    assert slice_["n"] == int(wide.sum())
    assert slice_["auc"] > 0.95 > rows["h5"]["auc"]
    assert {"base_rate", "baseline_auc", "auc_se", "threshold_quantile"} <= set(slice_)


def test_summary_includes_expansion_slice_means():
    rng = np.random.default_rng(4)
    n, levels = 400, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    close_q = (last[:, None] + np.outer(rng.uniform(1, 3, n), [-1, 0, 1]))[:, :, None]
    rows = {name: fs.score_stream(close_q, last, (last + rng.normal(0, 2, n))[:, None],
                                  np.arange(n) * 5, levels, rng.random((n, 1)),
                                  rng.random((n, 1)), horizons=(5,))
            for name in ("A", "B")}
    summary = fs.summarize(rows, (5,))["h5"]
    assert {"mean_expansion_auc", "mean_expansion_baseline_auc"} <= set(summary)


def test_expansion_slice_ranks_predicted_range_relative_to_recent_volatility():
    """Same raw width, but half the rows come from a calm regime: relative to
    recent volatility those are the expected expansions."""
    rng = np.random.default_rng(5)
    n, levels = 1000, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    calm = np.arange(n) % 2 == 0
    recent_vol = np.where(calm, 0.5, 5.0)             # price units of recent 1-bar moves
    width = np.full(n, 2.0) + rng.uniform(0, 0.1, n)
    sign = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    close_q = (last[:, None] + np.outer(width, [-1.0, 0.0, 1.0]))[:, :, None]
    rows = fs.score_stream(close_q, last, (last + sign)[:, None], np.arange(n) * 5, levels,
                           baseline=rng.random((n, 1)), p_up_values=rng.random((n, 1)),
                           horizons=(5,), range_scale=recent_vol)
    assert rows["h5"]["expansion_slice"]["n"] == 200


def test_recent_volatility_scale_uses_only_bars_up_to_the_anchor():
    stream = _stream(400)
    anchors = np.array([200, 300])
    before = fs.recent_volatility(stream, anchors)
    changed = stream.values.copy()
    changed[301:] *= 2.0
    np.testing.assert_allclose(before, fs.recent_volatility(
        fs.Stream(stream.name, stream.close_ns, changed), anchors))
    assert (before > 0).all()


# ------------------------------------------------------------------ A4 separate direction head

class _HookedChronos:
    """Fake Chronos whose forecast tokens pass through ``output_patch_embedding``."""

    def __init__(self, d_model: int = 8, quantiles: int = 3):
        torch = _torch()
        self.d_model, self.quantiles = d_model, quantiles
        self.output_patch_embedding = torch.nn.Identity()
        self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9][:quantiles])

    def __call__(self, *, context, group_ids, num_output_patches, future_target):
        torch = _torch()
        rows = context.shape[0]
        # row r, patch p, dim j -> 100*r + 10*p + j
        tokens = (100 * torch.arange(rows, dtype=torch.float32)[:, None, None]
                  + 10 * torch.arange(num_output_patches, dtype=torch.float32)[None, :, None]
                  + torch.arange(self.d_model, dtype=torch.float32)[None, None, :])
        self.output_patch_embedding(tokens)
        preds = torch.zeros(rows, self.quantiles, num_output_patches * 16)
        loss = None if future_target is None else torch.tensor(0.1)
        return SimpleNamespace(loss=loss, quantile_preds=preds)


@needs_torch
def test_forecast_close_returns_the_close_rows_forecast_tokens():
    torch = _torch()
    model = _HookedChronos()
    _, _, hidden = fs.forecast_close(model, torch.zeros(2, 5, 32), torch.ones(2, 64),
                                     forecast_length=64, return_hidden=True)
    assert hidden.shape == (2, 4, 8)
    assert hidden[0, 0, 0].item() == 300.0          # window 0 close row = row 3
    assert hidden[1, 2, 1].item() == 800 + 20 + 1   # window 1 close row = row 8, patch 2


@needs_torch
def test_direction_head_maps_forecast_tokens_to_one_logit_per_horizon():
    torch = _torch()
    head = fs.make_direction_head(d_model=8, n_patches=4, horizons=(5, 10, 20, 50))
    tokens = torch.randn(3, 4, 8, requires_grad=True)
    logits = head(tokens)
    assert logits.shape == (3, 4)
    logits.sum().backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0


def test_teacher_heads_are_training_only_and_never_ship():
    """SSL-only: A7/A8 heads teach the encoder and are discarded. Evaluation
    reads direction from the model itself (native quantile P(up), Atlas REG)."""
    import inspect

    assert fs.DIRECTION_HEAD_POLICY == "training_only_teacher_discarded"
    assert "direction_head" not in inspect.signature(fs.predict_period).parameters
    assert "direction_head" not in inspect.signature(fs.evaluate).parameters


def test_teacher_stage_report_records_the_discarded_teacher(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"w")
    report = fs.build_stage_report(
        arm="a7", direction_weight=3.0, seed=0, parent_path="/p", parent_sha256="c" * 64,
        checkpoint=checkpoint, base_identity={}, provenance={}, timeframes=("3min",),
        streams=["NQ@3min"], code=None, training={}, history=[], parent_select={},
        best_select={}, best_epoch=0, step_seconds=[1.0], elapsed_seconds=1.0)
    assert report["config"]["direction_head"] == "training_only_teacher_discarded"
    assert sorted(p.name for p in checkpoint.iterdir()) == ["adapter_model.safetensors"]


# ------------------------------------------------------------------ A5 first-passage side (expansion-aligned SSL target)

def _path_stream(closes, n_before: int = 200, bar_range: float = 1.0) -> np.ndarray:
    """Flat history of 1-point bars, then the given close path."""
    base = [100.0] * n_before + list(closes)
    c = np.asarray(base, float)
    return np.column_stack([c, c + bar_range / 2, c - bar_range / 2, c, np.full(len(c), 10.0)])


def test_first_passage_barriers_follow_the_expansion_scaling():
    assert dict(fs.FIRST_PASSAGE_BARRIERS) == {5: 2.1, 10: 3.0, 20: 4.2, 50: 6.7}
    for horizon, k in fs.FIRST_PASSAGE_BARRIERS:
        assert k == pytest.approx(0.95 * np.sqrt(horizon), abs=0.05)


def test_causal_true_range_scale_is_a_trailing_median_including_the_bar():
    values = _path_stream([100.0] * 10)
    sigma = fs.true_range_scale(values, lookback=128)
    assert np.isnan(sigma[:127]).all()
    tr = np.log(100.5) - np.log(99.5)
    assert sigma[200] == pytest.approx(tr)
    changed = values.copy()
    changed[201:] *= 1.5
    assert fs.true_range_scale(changed, lookback=128)[200] == pytest.approx(sigma[200])


def test_first_passage_labels_up_down_and_none():
    tr = np.log(100.5) - np.log(99.5)
    step = np.exp(tr)                                   # one sigma per bar in log space
    up = [100.0 * step ** j for j in range(1, 11)]
    down = [100.0 / step ** j for j in range(1, 11)]
    flat = [100.0] * 10
    for path, hit, side in ((up, True, True), (down, True, False), (flat, False, None)):
        values = _path_stream(path)
        sigma = fs.true_range_scale(values, lookback=128)
        hits, ups, first = fs.first_passage(values, np.array([199]), sigma, horizon=5, k=2.1)
        assert bool(hits[0]) is hit
        if hit:
            assert bool(ups[0]) is side and first[0] == 3   # 3 sigma >= 2.1 at bar 3
        else:
            assert first[0] == 0


def test_first_passage_takes_whichever_barrier_is_reached_first():
    tr = np.log(100.5) - np.log(99.5)
    down_then_up = [100.0 * np.exp(-3 * tr), 100.0 * np.exp(5 * tr)]
    values = _path_stream(down_then_up + [100.0] * 8)
    sigma = fs.true_range_scale(values, lookback=128)
    hits, ups, first = fs.first_passage(values, np.array([199]), sigma, horizon=5, k=2.1)
    assert hits[0] and not ups[0] and first[0] == 1


def test_first_passage_reads_only_the_next_horizon_bars():
    tr = np.log(100.5) - np.log(99.5)
    late_jump = [100.0] * 5 + [100.0 * np.exp(10 * tr)]
    values = _path_stream(late_jump + [100.0] * 5)
    sigma = fs.true_range_scale(values, lookback=128)
    hits, _, _ = fs.first_passage(values, np.array([199]), sigma, horizon=5, k=2.1)
    assert not hits[0]


def test_score_stream_reports_side_given_first_passage():
    rng = np.random.default_rng(8)
    n, levels = 600, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    hit = rng.random((n, 1)) < 0.3
    up = rng.random((n, 1)) < 0.5
    p_up = np.where(up, 0.9, 0.1)
    close_q = (last[:, None] + np.outer(np.ones(n), [-1.0, 0.0, 1.0]))[:, :, None]
    rows = fs.score_stream(close_q, last, (last + 1)[:, None], np.arange(n) * 5, levels,
                           rng.random((n, 1)), p_up, horizons=(5,),
                           first_passage_labels=(hit, up))
    side = rows["h5"]["first_passage_side"]
    assert side["n"] == int(hit.sum()) and side["auc"] == pytest.approx(1.0)
    assert {"baseline_auc", "hit_rate", "up_rate"} <= set(side)


# ------------------------------------------------------------------ A6 candle-range (CRT) teacher

def _candles(rows):
    """rows of (high, low, close); open = previous close, volume constant."""
    rows = np.asarray(rows, float)
    opens = np.concatenate([[rows[0, 2]], rows[:-1, 2]])
    return np.column_stack([opens, rows[:, 0], rows[:, 1], rows[:, 2], np.full(len(rows), 10.0)])


def test_candle_range_classes_label_each_candle_against_the_previous_one():
    prior = (10.0, 8.0, 9.0)
    cases = {
        (11.0, 8.5, 10.5): "bull_breakout",   # took high, closed above it
        (11.0, 8.5, 9.5): "bear_sweep",       # took high, closed back inside
        (9.5, 7.0, 7.5): "bear_breakout",     # took low, closed below it
        (9.5, 7.0, 8.5): "bull_sweep",        # took low, closed back inside
        (9.5, 8.5, 9.0): "inside",
        (11.0, 7.0, 9.0): "outside",
    }
    for candle, expected in cases.items():
        classes = fs.candle_range_classes(_candles([prior, candle]))
        assert fs.CRT_CLASSES[classes[1]] == expected
    assert classes[0] == fs.CRT_CLASSES.index("inside")      # first candle has no prior


def test_candle_range_classes_are_causal():
    stream = _stream(300)
    base = fs.candle_range_classes(stream.values)
    changed = stream.values.copy()
    changed[201:] *= 1.3
    np.testing.assert_array_equal(base[:201], fs.candle_range_classes(changed)[:201])


# ------------------------------------------------------------------ ties are not direction

def test_score_stream_scores_direction_only_where_price_moved():
    """An unchanged close is not 'down'. Ties are excluded from the primary AUC
    and reported separately, so predicting ties cannot pass as direction skill."""
    rng = np.random.default_rng(10)
    n, levels = 800, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    move = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.35, 0.3, 0.35])
    future = (last + move)[:, None]
    p_up = np.where(move > 0, 0.9, np.where(move < 0, 0.1, 0.95))[:, None]  # ties look "up"
    close_q = (last[:, None] + np.outer(np.ones(n), [-1.0, 0.0, 1.0]))[:, :, None]
    row = fs.score_stream(close_q, last, future, np.arange(n) * 5, levels,
                          rng.random((n, 1)), p_up, horizons=(5,))["h5"]
    assert row["auc"] == pytest.approx(1.0)
    assert row["auc_with_ties"] < 0.95
    assert row["tie_rate"] == pytest.approx(np.mean(move == 0))
    assert row["n"] == int((move != 0).sum())


def test_causal_baseline_is_fit_on_moves_not_ties():
    """With symmetric moves and many ties, a tie-as-down fit would push P(up)
    toward ~0.3; fitting on moves only keeps it near 0.5."""
    rng = np.random.default_rng(11)
    n = 3000
    steps = rng.choice([-0.25, 0.0, 0.25], size=n, p=[0.2, 0.6, 0.2])
    close = 100.0 + np.cumsum(steps)
    values = np.column_stack([close, close + 0.25, close - 0.25, close, np.full(n, 10.0)])
    times = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    stream = fs.Stream("ZN@1min", times.asi8, values)
    scores = fs.causal_baseline_scores(stream, np.arange(100, 2000), np.arange(2100, 2800),
                                       horizons=(5,))
    assert abs(float(scores.mean()) - 0.5) < 0.06


@needs_torch
def test_predict_period_caps_series_per_forward_pass_with_extra_inputs():
    """11-series groups must not multiply GPU memory: cap total series per pass."""
    stream = _stream(900)

    class _Model(_FakeChronos):
        chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    model = _Model()
    fs.predict_period(model, stream, np.arange(300, 600), device="cpu", horizons=(5,),
                      batch_windows=256, features=fs.bar_structure(stream.values))
    rows = [call["context"].shape[0] for call in model.calls]
    assert max(rows) <= fs.MAX_SERIES_PER_PASS
    assert sum(rows) == 300 * (5 + len(fs.BAR_FEATURES))


# ------------------------------------------------------------------ A7 liquidity-break teacher (Osler: stops beyond levels)

def _session_stream(days: int = 4, minutes: int = 30) -> fs.Stream:
    """30-min bars over several ET days, flat at 100 with 1-point ranges."""
    times = pd.date_range("2025-03-03 00:00", periods=days * 48, freq=f"{minutes}min",
                          tz="America/New_York").tz_convert("UTC")
    close = np.full(len(times), 100.0)
    values = np.column_stack([close, close + 0.5, close - 0.5, close, np.full(len(times), 10.0)])
    return fs.Stream("NQ@30min", times.asi8, values)


def test_rolling_break_events_use_only_prior_bars():
    values = _bars([(10, 10.5, 9.5, 10, 1)] * 30 + [(10, 12, 10, 11.5, 1)] + [(10, 10.5, 9.5, 10, 1)] * 5)
    events = fs.liquidity_break_events(fs.Stream("X@1min", np.arange(len(values)) * 60_000_000_000,
                                                 values))
    names = list(fs.LIQUIDITY_EVENTS)
    assert events[30, names.index("break_20_high")] and not events[30, names.index("break_20_low")]
    assert not events[29].any()
    changed = values.copy()
    changed[31:] *= 3.0
    later = fs.liquidity_break_events(fs.Stream("X@1min", np.arange(len(values)) * 60_000_000_000,
                                                changed))
    np.testing.assert_array_equal(events[:31], later[:31])


def test_prior_day_levels_come_from_the_completed_previous_session():
    stream = _session_stream()
    values = stream.values.copy()
    et = pd.DatetimeIndex(stream.close_ns, tz="UTC").tz_convert("America/New_York")
    day2_rth = (et.date == pd.Timestamp("2025-03-04").date()) & (et.hour >= 10) & (et.hour < 16)
    values[day2_rth, 1] = 105.0                         # day-2 RTH high = 105
    probe = np.flatnonzero((et.date == pd.Timestamp("2025-03-05").date()) & (et.hour == 11))[0]
    values[probe, 1], values[probe, 3] = 106.5, 106.0   # day-3 close above day-2 high
    events = fs.liquidity_break_events(fs.Stream(stream.name, stream.close_ns, values))
    names = list(fs.LIQUIDITY_EVENTS)
    assert events[probe, names.index("break_prior_day_high")]
    # before day 3 the day-2 high is not known yet
    early = np.flatnonzero((et.date == pd.Timestamp("2025-03-04").date()) & (et.hour == 15))[0]
    assert not events[early, names.index("break_prior_day_high")]


@needs_torch
def test_liquidity_teacher_loss_learns_only_on_event_bars_that_moved():
    torch = _torch()
    logits = torch.tensor([[4.0, 4.0], [-4.0, -4.0], [4.0, -4.0]])
    up = torch.tensor([[True, True], [True, True], [False, False]])
    valid = torch.tensor([[True, True], [False, False], [True, False]])
    loss = fs.liquidity_teacher_loss(logits, up, valid)
    flipped = logits.clone()
    flipped[1] = 4.0                                    # non-event row: must not matter
    flipped[2, 1] = 4.0                                 # tie / masked cell: must not matter
    assert loss.item() == pytest.approx(fs.liquidity_teacher_loss(flipped, up, valid).item())
    wrong = logits.clone()
    wrong[0] = -4.0
    assert fs.liquidity_teacher_loss(wrong, up, valid) > loss


def test_a7_is_a1_plus_a_liquidity_teacher():
    assert fs.uses_liquidity_teacher("a7") and fs.uses_direction_head("a7")
    assert not fs.uses_bar_features("a7")
    assert not any(fs.uses_liquidity_teacher(arm) for arm in ("a1", "a2", "a3", "a8"))


def test_score_stream_reports_direction_on_liquidity_event_bars():
    rng = np.random.default_rng(12)
    n, levels = 600, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    move = rng.choice([-1.0, 1.0], size=n)
    event = rng.random(n) < 0.25
    p_up = np.where(event, (move > 0) * 0.8 + 0.1, rng.random(n))[:, None]
    close_q = (last[:, None] + np.outer(np.ones(n), [-1.0, 0.0, 1.0]))[:, :, None]
    row = fs.score_stream(close_q, last, (last + move)[:, None], np.arange(n) * 5, levels,
                          rng.random((n, 1)), p_up, horizons=(5,), event_mask=event)["h5"]
    assert row["event_slice"]["n"] == int(event.sum())
    assert row["event_slice"]["auc"] == pytest.approx(1.0)


# ------------------------------------------------------------------ REG-probe side check (what the expansion head reads)

class _RegChronos:
    """Fake encode(): hidden state of series r at token j is 1000*r + j."""

    def __init__(self, d_model=4):
        self.d_model = d_model

    def encode(self, *, context, group_ids, num_output_patches):
        torch = _torch()
        rows, length = context.shape
        patches = length // 16
        tokens = patches + 1 + num_output_patches
        hidden = (1000 * torch.arange(rows, dtype=torch.float32)[:, None, None]
                  + torch.arange(tokens, dtype=torch.float32)[None, :, None]).expand(rows, tokens, self.d_model)
        return (hidden,), None, None, patches


@needs_torch
def test_reg_embeddings_concatenate_the_ohlcv_reg_tokens_only():
    torch = _torch()
    context = torch.zeros(2, 11, 32)                 # 5 OHLCV + 6 extra series
    emb = fs.reg_embeddings(_RegChronos(), context)
    assert emb.shape == (2, 5 * 4)
    # window 1, OHLCV series 0..4 are rows 11..15; REG sits after 2 context patches
    assert emb[1, 0].item() == 11 * 1000 + 2 and emb[1, -1].item() == 15 * 1000 + 2


def test_probe_side_reports_signal_and_random_control():
    rng = np.random.default_rng(13)
    n_train, n_eval, dim = 3000, 1000, 16
    x_train = rng.normal(size=(n_train, dim)); x_eval = rng.normal(size=(n_eval, dim))
    y_train = (x_train[:, 0] + rng.normal(0, 1, n_train)) > 0
    y_eval = (x_eval[:, 0] + rng.normal(0, 1, n_eval)) > 0
    streams = np.array(["A", "B"] * (n_eval // 2))
    result = fs.probe_side(x_train, y_train, x_eval, y_eval, streams, seed=0)
    assert result["mean_auc"] > 0.65
    assert abs(result["mean_shuffled_auc"] - 0.5) < 0.05
    assert set(result["per_stream_auc"]) == {"A", "B"}


def test_cli_probe_command_parses():
    args = _cli().parse(["probe", "--checkpoint", "base", "--name", "a0", "--bar-features"])
    assert args.command == "probe" and args.bar_features and args.eval_per_stream == 1500


# ------------------------------------------------------------------ arm ranking (pre-registered checks)

def _arm_eval(auc: float, wql: float = 0.63, event: float = 0.52, side: float = 0.52,
              big: float = 0.52, baseline: float = 0.525, names=("A", "B", "C")) -> dict:
    per_stream = {n: {f"h{h}": {"auc": auc + 0.001 * i, "baseline_auc": baseline,
                                "scaled_wql": wql, "shuffled_auc": 0.5, "coverage_80": 0.8,
                                "expansion_slice": {"auc": big, "baseline_auc": baseline},
                                "event_slice": {"auc": event, "baseline_auc": baseline},
                                "first_passage_side": {"auc": side, "baseline_auc": baseline}}
                      for h in fs.HORIZONS} for i, n in enumerate(names)}
    return {"period": "select", "per_stream": per_stream, "summary": fs.summarize(per_stream)}


def _arm_probe(reg: float, causal: float = 0.52, shuffled: float = 0.5) -> dict:
    row = lambda auc: {"mean_auc": auc, "mean_shuffled_auc": shuffled, "streams_above_half": 3,
                       "streams": 3}
    return {"targets": {name: {"reg": row(reg), "causal": row(causal)}
                        for name in ("all_bars_h5", "all_bars_h10", "liquidity_break_h5",
                                     "breakout_side_h10")}}


def test_rank_arms_applies_the_preregistered_checks_against_a1():
    evals = {"a1": _arm_eval(0.514), "good": _arm_eval(0.535, event=0.55, side=0.55, big=0.54),
             "weak": _arm_eval(0.517), "broken": _arm_eval(0.54, wql=0.70)}
    probes = {"a1": _arm_probe(0.50), "good": _arm_probe(0.56), "weak": _arm_probe(0.51),
              "broken": _arm_probe(0.56)}
    table = fs.rank_arms(evals, probes, reference="a1")
    by_arm = {row["arm"]: row for row in table}
    assert table[0]["arm"] == "good"
    assert all(by_arm["good"]["checks"].values())
    assert by_arm["good"]["diagnostics"]["slices_above_baseline"]
    assert not by_arm["weak"]["checks"]["direction_vs_reference"]
    assert not by_arm["broken"]["checks"]["forecast_not_degraded"]
    assert "a1" not in by_arm


def test_rank_arms_tolerates_missing_probe():
    table = fs.rank_arms({"a1": _arm_eval(0.514), "x": _arm_eval(0.53)}, {}, reference="a1")
    assert table[0]["checks"]["embedding_probe"] is False


def test_cli_rank_reads_rescore_and_probe_files(tmp_path):
    import json

    cli = _cli()
    (tmp_path / "rescore_a1_select.json").write_text(json.dumps(_arm_eval(0.514)))
    (tmp_path / "rescore_good_select.json").write_text(json.dumps(
        _arm_eval(0.535, event=0.55, side=0.55, big=0.54)))
    (tmp_path / "good_regprobe_select.json").write_text(json.dumps(_arm_probe(0.56)))
    out = tmp_path / "ranking.json"
    cli.main(["rank", "--eval-dir", str(tmp_path), "--out", str(out)])
    ranking = json.loads(out.read_text())["ranking"]
    assert ranking[0]["arm"] == "good" and ranking[0]["passes"] == 4


# ------------------------------------------------------------------ stack test: does the model add beyond the simple rule?

def test_stack_gain_is_positive_only_when_the_model_adds_information():
    rng = np.random.default_rng(14)
    n = 4000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (0.4 * a + 0.4 * b + rng.normal(size=n)) > 0
    base = 1 / (1 + np.exp(-a))
    adds = 1 / (1 + np.exp(-b))                      # independent information
    copies = 1 / (1 + np.exp(-(a + rng.normal(0, 0.05, n))))  # same information as base
    gain_adds = fs.stack_gain(y, adds, base)
    gain_copies = fs.stack_gain(y, copies, base)
    assert gain_adds["gain"] > 0.02
    assert abs(gain_copies["gain"]) < 0.01
    assert {"stack_auc", "baseline_auc", "model_auc"} <= set(gain_adds)


def test_score_stream_can_return_rows_for_the_stack_test():
    rng = np.random.default_rng(15)
    n, levels = 300, np.array([0.1, 0.5, 0.9])
    last = np.full(n, 100.0)
    move = rng.choice([-1.0, 0.0, 1.0], size=(n, 1))
    close_q = (last[:, None] + np.outer(np.ones(n), [-1.0, 0.0, 1.0]))[:, :, None]
    rows = fs.stream_prediction_rows(last, last[:, None] + move, rng.random((n, 1)),
                                     rng.random((n, 1)), np.arange(n), horizons=(5,))
    assert rows["p_up"].shape == (n, 1) and rows["baseline"].shape == (n, 1)
    assert rows["moved"].dtype == bool and rows["up"].dtype == bool
    assert rows["moved"][:, 0].sum() == int((move != 0).sum())


# ------------------------------------------------------------------ tie-masked direction loss (next-stage option)

@needs_torch
def test_direction_loss_can_ignore_unchanged_closes():
    torch = _torch()
    last = torch.tensor([100.0, 100.0])
    future = torch.stack([torch.full((10,), 101.0), torch.full((10,), 100.0)])  # row 2 is a tie
    q = torch.tensor([[[101.0] * 10, [102.0] * 10, [103.0] * 10],
                      [[97.0] * 10, [98.0] * 10, [99.0] * 10]])
    q_tie_up = q.clone()
    q_tie_up[1] = torch.tensor([[101.0] * 10, [102.0] * 10, [103.0] * 10])
    masked = fs.direction_loss(q, last, future, _levels(), (5,), mask_ties=True)
    masked_changed = fs.direction_loss(q_tie_up, last, future, _levels(), (5,), mask_ties=True)
    assert masked.item() == pytest.approx(masked_changed.item())       # tie row ignored
    unmasked = fs.direction_loss(q, last, future, _levels(), (5,))
    unmasked_changed = fs.direction_loss(q_tie_up, last, future, _levels(), (5,))
    assert unmasked.item() != pytest.approx(unmasked_changed.item())   # default unchanged


def test_cli_train_mask_ties_flag_names_the_run():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a2", "--direction-weight", "3", "--mask-ties"])
    assert args.mask_ties
    assert cli.run_settings(args)["out_dir"].name == "a2_lam3_ties-masked_seed0"


def test_rank_arms_counts_only_generic_checks_and_reports_slices_as_diagnostics():
    """FFM is a generic market-context model: expansion-specific slices are
    reported but do not count toward the ranking."""
    evals = {"a1": _arm_eval(0.514),
             "generic": _arm_eval(0.535, event=0.49, side=0.49, big=0.49)}
    probes = {"generic": _arm_probe(0.56)}
    row = fs.rank_arms(evals, probes, reference="a1")[0]
    assert set(row["checks"]) == {"direction_vs_reference", "beats_causal_baseline",
                                  "embedding_probe", "forecast_not_degraded"}
    assert row["passes"] == 4
    assert row["diagnostics"]["slices_above_baseline"] is False


# ------------------------------------------------------------------ context length is a parameter (A8: longer context)

@needs_torch
def test_predict_period_uses_the_requested_context_length():
    class _Model(_FakeChronos):
        chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    stream = _stream(2000)
    model = _Model()
    fs.predict_period(model, stream, np.array([1500, 1600]), device="cpu", horizons=(5,),
                      context_length=1024)
    assert model.calls[0]["context"].shape[-1] == 1024


def test_stage_report_records_the_context_length(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"w")
    report = fs.build_stage_report(
        arm="a2", direction_weight=3.0, seed=0, parent_path="/p", parent_sha256="c" * 64,
        checkpoint=checkpoint, base_identity={}, provenance={}, timeframes=("3min",),
        streams=["NQ@3min"], code=None, training={}, history=[], parent_select={},
        best_select={}, best_epoch=0, step_seconds=[1.0], elapsed_seconds=1.0,
        context_length=1024)
    assert report["config"]["context_length"] == 1024


def test_train_rejects_context_not_a_multiple_of_the_patch(tmp_path):
    with pytest.raises(ValueError):
        fs.train_forecast_direction({}, parent=tmp_path, base_snapshot=tmp_path,
                                    out_dir=tmp_path, provenance={}, arm="a2",
                                    direction_weight=3.0, context_length=1000)


def test_cli_context_length_flag_names_the_run_and_reaches_evaluate():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a2", "--direction-weight", "3", "--context-length", "1024"])
    assert cli.run_settings(args)["out_dir"].name == "a2_lam3_ctx1024_seed0"
    default = cli.parse(["train", "--arm", "a2", "--direction-weight", "3"])
    assert cli.run_settings(default)["out_dir"].name == "a2_lam3_seed0"
    assert cli.parse(["evaluate", "--checkpoint", "x", "--name", "y",
                      "--context-length", "1024"]).context_length == 1024
    assert cli.parse(["probe", "--checkpoint", "x", "--name", "y",
                      "--context-length", "1024"]).context_length == 1024


# ------------------------------------------------------------------ R3' mirrored-future contrastive direction (pure SSL)

@needs_torch
def test_future_path_is_scale_free_and_mirror_flips_only_the_sign():
    torch = _torch()
    future = torch.tensor([[101.0, 102.0, 101.5, 103.0]])
    last = torch.tensor([100.0])
    path = fs.future_path(future, last)
    doubled = fs.future_path(100.0 + 2.0 * (future - 100.0), last)
    assert torch.allclose(path.abs().sum(1), torch.ones(1), atol=1e-6)      # unit scale
    assert torch.allclose(path, doubled, atol=1e-2)                           # size-invariant (approx)
    assert torch.allclose(fs.mirror_path(path), -path)


@needs_torch
def test_mirror_contrastive_loss_only_rewards_the_right_direction():
    torch = _torch()
    real = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    context = torch.tensor([[1.0, 0.0], [0.0, 1.0]])            # matches its real future
    flipped_context = -context                                  # matches the mirrored future
    good = fs.mirror_contrastive_loss(context, real, -real, temperature=0.1)
    bad = fs.mirror_contrastive_loss(flipped_context, real, -real, temperature=0.1)
    assert good < bad
    # the loss is invariant to scaling the future (both paths scale together)
    assert fs.mirror_contrastive_loss(context, 3 * real, -3 * real, temperature=0.1) == pytest.approx(
        good.item(), abs=1e-5)


def test_r3_mirror_arm_flags():
    assert fs.uses_mirror_contrastive("a8") and fs.uses_direction_head("a8")
    assert not fs.uses_bar_features("a8")
    assert not any(fs.uses_mirror_contrastive(a) for a in ("a1", "a2", "a3", "a7"))


def test_cli_a8_run_directory():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a8", "--direction-weight", "1"])
    assert cli.run_settings(args)["out_dir"].name == "a8_lam1_seed0"


@needs_torch
def test_mirror_teacher_maps_forecast_tokens_and_paths_to_the_same_space():
    torch = _torch()
    teacher = fs.make_mirror_teacher(d_model=8, n_patches=4)
    tokens, path = torch.randn(3, 4, 8), torch.randn(3, fs.MIRROR_LENGTH)
    assert teacher["context"](tokens).shape == (3, fs.MIRROR_DIM)
    assert teacher["future"](path).shape == (3, fs.MIRROR_DIM)


# ------------------------------------------------------------------ CLM lessons folded into A8

@needs_torch
def test_mirror_loss_masks_false_negatives_from_overlapping_windows():
    torch = _torch()
    context = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
    real = torch.tensor([[1.0, 0.0], [0.98, 0.02]])            # nearly identical futures
    exclude = torch.tensor([[False, True], [True, False]])      # same stream, overlapping
    masked = fs.mirror_contrastive_loss(context, real, -real, false_negatives=exclude,
                                        use_mirror=False, symmetric=False)
    unmasked = fs.mirror_contrastive_loss(context, real, -real, use_mirror=False,
                                          symmetric=False)
    assert masked < unmasked


@needs_torch
def test_mirror_loss_curriculum_can_disable_the_hard_negative():
    torch = _torch()
    context = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    real = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    confusable = context.clone()                                  # a mirror that competes
    with_mirror = fs.mirror_contrastive_loss(context, real, confusable, use_mirror=True)
    without = fs.mirror_contrastive_loss(context, real, confusable, use_mirror=False)
    assert with_mirror > without + 0.1                            # extra negative adds loss


@needs_torch
def test_mirror_loss_uses_a_learned_capped_temperature():
    torch = _torch()
    context, real = torch.eye(2), torch.eye(2)
    hot = fs.mirror_contrastive_loss(context, real, -real, logit_scale=torch.tensor(0.0))
    sharp = fs.mirror_contrastive_loss(context, real, -real, logit_scale=torch.tensor(10.0))
    capped = fs.mirror_contrastive_loss(context, real, -real,
                                        logit_scale=torch.tensor(float(np.log(100.0))))
    assert sharp < hot
    assert sharp.item() == pytest.approx(capped.item(), abs=1e-6)  # exp(10) capped at 100


def test_same_stream_overlap_mask():
    streams = np.array([0, 0, 1, 0])
    anchors = np.array([100, 110, 105, 400])
    mask = fs.overlap_false_negatives(streams, anchors, window=20)
    assert mask.tolist() == [[False, True, False, False], [True, False, False, False],
                             [False, False, False, False], [False, False, False, False]]


# ------------------------------------------------------------------ A9: direction BCE directly on the consumer REG embedding

class _EncoderChronos:
    """Fake Chronos whose forward runs ``encoder`` over [context | REG | forecast] tokens.
    Hidden value of series r at token j is 1000*r + j."""

    def __init__(self, d_model=4):
        torch = _torch()
        self.d_model = d_model
        self.encoder = torch.nn.Identity()
        self.output_patch_embedding = torch.nn.Identity()
        self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    def __call__(self, *, context, group_ids, num_output_patches, future_target):
        torch = _torch()
        rows, length = context.shape
        tokens = length // 16 + 1 + num_output_patches
        hidden = (1000 * torch.arange(rows, dtype=torch.float32)[:, None, None]
                  + torch.arange(tokens, dtype=torch.float32)[None, :, None]).expand(
                      rows, tokens, self.d_model).clone()
        hidden = self.encoder(SimpleNamespace(last_hidden_state=hidden)).last_hidden_state
        self.output_patch_embedding(hidden[:, -num_output_patches:])
        preds = torch.zeros(rows, 3, num_output_patches * 16)
        loss = None if future_target is None else torch.tensor(0.1)
        return SimpleNamespace(loss=loss, quantile_preds=preds)


@needs_torch
def test_forecast_close_returns_the_consumer_reg_embedding_from_the_same_pass():
    torch = _torch()
    model = _EncoderChronos()
    _, _, reg = fs.forecast_close(model, torch.zeros(2, 5, 32), torch.ones(2, 64),
                                  forecast_length=64, return_reg=True)
    assert reg.shape == (2, 5 * 4)
    # 32-bar context = 2 patches, so REG is token 2; window 1's series are rows 5..9
    assert reg[0, 0].item() == 0 * 1000 + 2
    assert reg[1, 0].item() == 5 * 1000 + 2 and reg[1, -1].item() == 9 * 1000 + 2


@needs_torch
def test_reg_teacher_maps_the_consumer_embedding_to_one_logit_per_horizon():
    torch = _torch()
    head = fs.make_reg_teacher(d_model=8, horizons=fs.REG_HORIZONS)
    logits = head(torch.randn(4, 5 * 8))
    assert logits.shape == (4, len(fs.REG_HORIZONS))


def test_a9_targets_the_consumer_embedding():
    assert fs.uses_reg_teacher("a9") and fs.uses_direction_head("a9")
    assert not fs.uses_bar_features("a9")
    assert fs.REG_HORIZONS == (5, 10, 20)
    assert not any(fs.uses_reg_teacher(a) for a in ("a1", "a2", "a3", "a7", "a8"))


def test_a9_requires_a_direction_term(tmp_path):
    with pytest.raises(ValueError):
        fs.train_forecast_direction({}, parent=tmp_path, base_snapshot=tmp_path,
                                    out_dir=tmp_path, provenance={}, arm="a9",
                                    direction_weight=0.0)


def test_cli_a9_run_directory():
    cli = _cli()
    args = cli.parse(["train", "--arm", "a9", "--direction-weight", "1"])
    assert cli.run_settings(args)["out_dir"].name == "a9_lam1_seed0"
