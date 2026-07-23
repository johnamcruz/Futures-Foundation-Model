import inspect
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CHRONOS_TORCH_TESTS") != "1",
    reason="torch test — set CHRONOS_TORCH_TESTS=1")


def _bars(batch=3, context=64, future=25):
    import torch
    rows = context + future
    close = torch.linspace(100.0, 112.0, rows).repeat(batch, 1)
    high, low = close + 1.0, close - 1.0
    open_ = close - .1
    volume = torch.full_like(close, 1000.0)
    bars = torch.stack((open_, high, low, close, volume), dim=1)
    return bars[:, :, :context], bars[:, :, context:]


def test_torch_targets_match_public_numpy_reference():
    import torch
    from futures_foundation.finetune.pretext.momentum_volatility import (
        momentum_volatility_targets,
    )
    from futures_foundation.finetune.pretext._torch.momentum_volatility import (
        _momentum_volatility_targets_torch,
    )

    context, future = _bars(batch=1)
    context[:, 1] += torch.linspace(0.0, 1.0, context.shape[-1])
    future[:, 1] += torch.linspace(0.0, 1.0, future.shape[-1])
    offsets = torch.tensor([4, 9, 19, 24])
    momentum, volatility, coupling = _momentum_volatility_targets_torch(
        context, future, offsets)
    joined = torch.cat((context, future), dim=2)[0].numpy()
    expected = momentum_volatility_targets(
        joined[1], joined[2], joined[3], 63)
    np.testing.assert_allclose(momentum[0], expected.momentum, rtol=1e-6)
    np.testing.assert_allclose(volatility[0], expected.volatility, rtol=1e-6)
    np.testing.assert_array_equal(coupling[0], expected.coupling)


def test_future_mutation_does_not_change_causal_scale_or_context():
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility import (
        _momentum_volatility_targets_torch,
    )

    context, future = _bars(batch=1)
    original_context = context.clone()
    first = _momentum_volatility_targets_torch(
        context, future, torch.tensor([4, 9, 19, 24]))
    future[:, 1] += 100.0
    second = _momentum_volatility_targets_torch(
        context, future, torch.tensor([4, 9, 19, 24]))
    assert torch.equal(context, original_context)
    assert not torch.equal(first[1], second[1])


def test_network_exposes_one_shared_embedding_and_expected_heads(monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import momentum_volatility as module

    class FakeEncoder(torch.nn.Module):
        hidden_dim = 8
        seq_len = 16

        def forward(self, value):
            return value.mean(2).repeat(1, 8)

    monkeypatch.setattr(module, "MultiHorizonForecastNet", module.MultiHorizonForecastNet)
    monkeypatch.setattr(
        "futures_foundation.finetune.pretext._torch.common.load_mantis",
        lambda *_args, **_kwargs: FakeEncoder())
    net = module.MomentumVolatilityNet(
        C=5, new_channels=3, horizons=(5, 10, 20, 25))
    calls = 0
    original = net.embed

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    net.embed = counted
    candle, momentum, volatility, coupling = net.forward_mv(
        torch.randn(2, 5, 64))
    assert calls == 1
    assert candle.shape == (2, 5, 4)
    assert momentum.shape == volatility.shape == (2, 4)
    assert coupling.shape == (2, 4, 4)


def test_public_trainer_requires_warm_lora_defaults():
    from futures_foundation.finetune.pretext._torch.momentum_volatility import (
        train_ssl_momentum_volatility,
    )

    signature = inspect.signature(train_ssl_momentum_volatility)
    assert signature.parameters["lora_r"].default == 8
    assert signature.parameters["freeze_encoder_layers"].default == 2
    assert signature.parameters["backbone_ckpt"].default is None
    assert signature.parameters["scale_lookback"].default == 64
