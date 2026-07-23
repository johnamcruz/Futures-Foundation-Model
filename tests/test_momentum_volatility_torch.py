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
    np.testing.assert_allclose(
        momentum[0], expected.momentum_strength, rtol=1e-6)
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
    assert not hasattr(net, "adapter")
    assert isinstance(net.mv_head, torch.nn.Linear)


def test_transition_loss_backpropagates_into_exported_encoder(monkeypatch):
    import torch
    from futures_foundation.finetune.pretext._torch import momentum_volatility as module

    class FakeEncoder(torch.nn.Module):
        hidden_dim = 4
        seq_len = 16

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, self.hidden_dim, bias=False)

        def forward(self, value):
            pooled = value.mean(dim=2)
            return self.projection(pooled)

    monkeypatch.setattr(module, "load_mantis", lambda *_args, **_kwargs: FakeEncoder())
    net = module.MomentumVolatilityNet(C=5, horizons=(5, 10))
    *_, embedding = net.forward_mv(
        torch.randn(8, 5, 64), return_embedding=True)
    states = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = module._transition_contrastive_loss(embedding, states)
    loss.backward()

    gradient = net.encoder.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_transition_classification_is_balanced_and_handles_absent_state():
    import torch
    from futures_foundation.finetune.pretext._torch.momentum_volatility import (
        _balanced_transition_cross_entropy,
    )

    logits = torch.randn(7, 4, requires_grad=True)
    truth = torch.tensor([0, 0, 0, 0, 1, 2, 2])
    loss = _balanced_transition_cross_entropy(logits, truth)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_tiny_mv_v3_training_returns_encoder_only_state(monkeypatch, tmp_path):
    import torch
    from futures_foundation.finetune.pretext._torch import momentum_volatility as module

    class FakeEncoder(torch.nn.Module):
        hidden_dim = 4
        seq_len = 16

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, self.hidden_dim)

        def forward(self, value):
            return self.projection(value.mean(dim=2))

    monkeypatch.setattr(module, "load_mantis", lambda *_args, **_kwargs: FakeEncoder())
    parent = tmp_path / "parent.pt"
    torch.save(FakeEncoder().state_dict(), parent)
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, .2, 420)).astype(np.float32)
    width = rng.uniform(.1, .8, len(close)).astype(np.float32)
    bars = np.column_stack((
        close, close + width, close - width, close,
        rng.uniform(100, 1000, len(close)))).astype(np.float32)
    state, history = module.train_ssl_momentum_volatility(
        bars, np.arange(0, 200), np.arange(250, 340),
        horizons=(5,), context_lengths=(64,), epochs=1, steps_per_epoch=1,
        batch=4, device="cpu", lora_r=0, freeze_encoder_layers=0,
        scale_lookback=64, momentum_lookback=20, verbose=False,
        backbone_ckpt=str(parent))

    assert state
    assert all(not key.startswith(("mv_head.", "decoder."))
               for key in state)
    assert "mv_transition_auc" in history[-1]
    assert "mv_parent_cosine" in history[-1]
    assert "mv_parent_relative_rmse" in history[-1]


def test_parent_retention_compares_student_to_frozen_same_input_teacher(
    monkeypatch, tmp_path,
):
    import torch
    from futures_foundation.finetune.pretext._torch import momentum_volatility as module

    class FakeEncoder(torch.nn.Module):
        hidden_dim = 4
        seq_len = 16

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, self.hidden_dim)

        def forward(self, value):
            return self.projection(value.mean(dim=2))

    monkeypatch.setattr(module, "load_mantis", lambda *_args, **_kwargs: FakeEncoder())
    parent = tmp_path / "parent.pt"
    torch.save(FakeEncoder().state_dict(), parent)
    context, future = _bars(batch=8, context=64, future=25)
    bars = torch.cat((context, future), dim=2).permute(0, 2, 1)
    big = bars.reshape(-1, 5).numpy()
    starts = np.arange(0, len(big) - 90)
    trainer = module._MomentumVolatilityTrainer(
        big, starts, starts, horizons=(5,), context_lengths=(64,),
        batch=4, epochs=1, steps_per_epoch=1, device="cpu",
        backbone_ckpt=str(parent), lora_r=0, verbose=False)
    trainer.build_net()
    _, _, parts = trainer._losses(trainer.make_batch(trainer.tr))

    assert float(parts["retention"].detach()) == pytest.approx(0.0, abs=1e-8)
    assert float(parts["parent_cosine"].detach()) == pytest.approx(1.0, abs=1e-6)
    assert all(not parameter.requires_grad
               for parameter in trainer.parent_encoder.parameters())
    with torch.no_grad():
        trainer.net.encoder.projection.weight.add_(.1)
    _, _, shifted = trainer._losses(trainer.make_batch(trainer.tr))
    assert float(shifted["retention"].detach()) > 0


def test_public_trainer_requires_warm_lora_defaults():
    from futures_foundation.finetune.pretext._torch.momentum_volatility import (
        train_ssl_momentum_volatility,
    )

    signature = inspect.signature(train_ssl_momentum_volatility)
    assert signature.parameters["lora_r"].default == 8
    assert signature.parameters["freeze_encoder_layers"].default == 2
    assert signature.parameters["backbone_ckpt"].default is None
    assert signature.parameters["scale_lookback"].default == 64
    assert signature.parameters["lr"].default == 1e-5
    assert signature.parameters["head_lr"].default == 1e-4
    assert signature.parameters["transition_contrastive_weight"].default == .1
    assert signature.parameters["parent_retention_weight"].default == .5
