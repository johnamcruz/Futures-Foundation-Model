"""Pure contracts for causal Chronos-2 Volume-Structure SSL targets."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from futures_foundation.finetune.classifiers.chronos2 import ssl_stages
from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
    VOLUME_STRUCTURE_REPORT_SCHEMA,
    VOLUME_STRUCTURE_STAGED_CHECKPOINT,
    VOLUME_STRUCTURE_STAGED_REPORT,
    VOLUME_STRUCTURE_TRAINER_SCHEMA,
    _evenly_spaced_starts,
    _fit_volume_structure_thresholds,
    _native_reg_volume_structure_losses,
    _recover_volume_structure_finalization,
    _runtime_pinned_adapter,
    _validate_native_volume_lift,
    _validate_volume_structure_resume,
    _volume_structure_encoder_input,
    _volume_structure_patch_mask,
    _volume_structure_run_identity,
    _volume_structure_standardize,
    _volume_structure_summary,
    _volume_structure_thresholds_from_arrays,
)


def _window(
        close: np.ndarray,
        volume: np.ndarray,
        *,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)
    if high is None:
        high = close + 1.0
    if low is None:
        low = close - 1.0
    return np.stack((close, high, low, close, volume)).astype(np.float32)


def test_volume_structure_summary_is_price_affine_and_volume_scale_invariant():
    length = 64
    close = 100.0 + np.sin(np.arange(length) / 5.0) * 4.0
    volume = np.exp(np.sin(np.arange(length) / 7.0)) * 1_000.0
    raw = _window(close, volume)

    transformed = raw.copy()
    transformed[:4] = transformed[:4] * 7.5 + 123.0
    transformed[4] *= 37.0

    np.testing.assert_allclose(
        _volume_structure_summary(raw),
        _volume_structure_summary(transformed),
        rtol=1e-6,
        atol=1e-6,
    )


def test_volume_structure_summary_responds_to_concentration_and_participation():
    length = 64
    high = np.full(length, 111.0, dtype=np.float32)
    low = np.full(length, 89.0, dtype=np.float32)
    volume = np.ones(length, dtype=np.float32)
    concentrated = _window(
        np.full(length, 100.0, dtype=np.float32),
        volume,
        high=high,
        low=low,
    )
    dispersed = _window(
        np.linspace(90.0, 110.0, length, dtype=np.float32),
        volume,
        high=high,
        low=low,
    )
    concentrated_summary = _volume_structure_summary(concentrated)
    dispersed_summary = _volume_structure_summary(dispersed)

    # Summary layout: four relative-volume patches, four weighted typical-price
    # locations, four relative-range patches, concentration, participation,
    # and displacement-volume alignment.
    assert concentrated_summary.shape == dispersed_summary.shape == (15,)
    assert concentrated_summary[12] > dispersed_summary[12]

    high_participation = dispersed.copy()
    low_participation = dispersed.copy()
    high_participation[4, -length // 4:] *= 8.0
    low_participation[4, -length // 4:] *= 0.125
    assert (
        _volume_structure_summary(high_participation)[13]
        > _volume_structure_summary(low_participation)[13]
    )


def test_volume_structure_summary_cannot_read_after_completed_context():
    context_length = 64
    close = np.linspace(100.0, 106.0, 96, dtype=np.float32)
    matrix = _window(close, np.linspace(500.0, 1_500.0, 96, dtype=np.float32))
    expected = _volume_structure_summary(matrix[:, :context_length])

    changed = matrix.copy()
    changed[:4, context_length:] = changed[:4, context_length:] * 100.0 + 1_000.0
    changed[4, context_length:] = 1_000_000.0
    actual = _volume_structure_summary(changed[:, :context_length])

    np.testing.assert_array_equal(actual, expected)


def test_evenly_spaced_starts_are_deterministic_and_cover_full_history():
    starts = np.arange(7, 10_007, 7, dtype=np.int64)
    selected = _evenly_spaced_starts(starts, max_samples=19)
    expected = starts[np.linspace(
        0, len(starts) - 1, num=19, dtype=np.int64)]

    np.testing.assert_array_equal(selected, expected)
    assert selected[0] == starts[0]
    assert selected[-1] == starts[-1]
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(
        _evenly_spaced_starts(starts[:8], max_samples=19), starts[:8])


def test_volume_structure_thresholds_are_deterministic_and_representative():
    rows = 320
    time = np.arange(rows, dtype=np.float32)
    close = 100.0 + np.sin(time / 11.0) * (1.0 + time / rows * 5.0)
    width = 0.5 + 0.45 * (1.0 + np.sin(time / 9.0))
    volume = 1_000.0 * np.exp(0.8 * np.sin(time / 13.0))
    matrix = _window(close, volume, high=close + width, low=close - width)
    context_length = 32
    starts = np.arange(rows - context_length + 1, dtype=np.int64)

    first = _fit_volume_structure_thresholds(
        matrix,
        starts,
        context_length=context_length,
        max_samples=23,
    )
    second = _fit_volume_structure_thresholds(
        matrix,
        starts,
        context_length=context_length,
        max_samples=23,
    )

    assert first == second
    assert first["fit_samples"] == 23
    assert first["participation_low"] < first["participation_high"]
    assert first["concentration_low"] < first["concentration_high"]
    assert first["displacement_low"] < first["displacement_high"]
    assert np.isfinite(first["temporal_mean"])
    assert np.isfinite(first["temporal_std"])


def test_participation_thresholds_preserve_falling_rising_semantics_when_skewed():
    rows = 100
    summaries = np.zeros((rows, 15), dtype=np.float32)
    summaries[:, 12] = np.linspace(0.1, 0.9, rows)
    summaries[:, 13] = np.concatenate((
        np.linspace(-1.0, -0.05, 80),
        np.linspace(0.05, 0.5, 20),
    ))
    summaries[:, 14] = np.linspace(0.1, 2.0, rows)

    thresholds = _volume_structure_thresholds_from_arrays(
        np.arange(rows, dtype=np.int64), summaries)

    assert thresholds["participation_low"] < 0.0
    assert thresholds["participation_high"] > 0.0
    assert thresholds["participation_high"] == pytest.approx(1e-6)
    states = np.stack([
        ssl_stages._volume_structure_states(summary, thresholds)
        for summary in summaries
    ])
    assert np.any(states[:, 0] == 0)
    assert np.any(states[:, 0] == 2)


def test_volume_structure_identity_is_stable_and_binds_parent_data_and_config():
    parent = "a" * 64
    data = "b" * 64
    config = {
        "objective_code_sha256": "1" * 64,
        "device": "cpu",
        "context_length": 64,
        "timeframes": ["3min"],
        "objective_weights": {
            "reconstruction": 1.0,
            "participation": 0.5,
        },
    }
    identity = _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256=data,
        config=config,
    )
    reordered = _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256=data,
        config={
            "device": "cpu",
            "objective_code_sha256": "1" * 64,
            "objective_weights": {
                "participation": 0.5,
                "reconstruction": 1.0,
            },
            "timeframes": ["3min"],
            "context_length": 64,
        },
    )

    assert identity == reordered
    assert len(identity) == 64
    assert identity != _volume_structure_run_identity(
        parent_sha256="c" * 64,
        data_identity_sha256=data,
        config=config,
    )
    assert identity != _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256="d" * 64,
        config=config,
    )
    assert identity != _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256=data,
        config={**config, "context_length": 128},
    )
    assert identity != _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256=data,
        config={**config, "device": "mps"},
    )
    assert identity != _volume_structure_run_identity(
        parent_sha256=parent,
        data_identity_sha256=data,
        config={**config, "objective_code_sha256": "2" * 64},
    )


def test_volume_structure_resume_requires_exact_run_identity():
    identity = "e" * 64
    _validate_volume_structure_resume(
        {
            "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
            "run_identity_sha256": identity,
        },
        run_identity_sha256=identity,
    )

    with pytest.raises(RuntimeError, match="identity"):
        _validate_volume_structure_resume(
            {
                "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
                "run_identity_sha256": "f" * 64,
            },
            run_identity_sha256=identity,
        )
    with pytest.raises(RuntimeError, match="identity"):
        _validate_volume_structure_resume({}, run_identity_sha256=identity)


def test_runtime_pinned_adapter_does_not_mutate_parent(tmp_path):
    parent = tmp_path / "adapter"
    parent.mkdir()
    original = {
        "base_model_name_or_path": "autogluon/chronos-2-small",
        "peft_type": "LORA",
        "revision": None,
    }
    (parent / "adapter_config.json").write_text(json.dumps(original))
    weights = parent / "adapter_model.safetensors"
    weights.write_bytes(b"adapter")
    base_snapshot = tmp_path / ("b" * 40)
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}\n")
    (base_snapshot / "model.safetensors").write_bytes(b"base")

    with _runtime_pinned_adapter(
        parent,
        "b" * 40,
        base_snapshot,
    ) as runtime:
        assert runtime != parent
        runtime_config = json.loads(
            (runtime / "adapter_config.json").read_text())
        assert runtime_config["base_model_name_or_path"] == str(
            base_snapshot.resolve())
        assert runtime_config["revision"] is None
        assert (runtime / "adapter_model.safetensors").read_bytes() == b"adapter"

    assert json.loads((parent / "adapter_config.json").read_text()) == original
    assert weights.read_bytes() == b"adapter"


def test_trainable_loader_receives_exact_snapshot_and_restores_canonical_metadata(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    parent = tmp_path / "adapter"
    parent.mkdir()
    original = {
        "base_model_name_or_path": "autogluon/chronos-2-small",
        "peft_type": "LORA",
        "revision": None,
    }
    (parent / "adapter_config.json").write_text(json.dumps(original))
    (parent / "adapter_model.safetensors").write_bytes(b"adapter")
    revision = "b" * 40
    base_snapshot = tmp_path / revision
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}\n")
    (base_snapshot / "model.safetensors").write_bytes(b"base")
    captured = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, adapter_config):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.ones(1))
            self.base_model = SimpleNamespace(model=object())
            self.peft_config = {
                "default": SimpleNamespace(**adapter_config),
            }

    class FakeAutoPeftModel:
        @staticmethod
        def from_pretrained(runtime, **kwargs):
            captured["config"] = json.loads(
                (Path(runtime) / "adapter_config.json").read_text())
            captured["kwargs"] = kwargs
            return FakeModel(captured["config"])

    fake_peft = ModuleType("peft")
    fake_peft.AutoPeftModel = FakeAutoPeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model, _ = ssl_stages._load_trainable_adapter(
        parent,
        "cpu",
        base_revision=revision,
        base_snapshot=base_snapshot,
    )

    assert captured["config"]["base_model_name_or_path"] == str(
        base_snapshot.resolve())
    assert captured["config"]["revision"] is None
    assert captured["kwargs"]["is_trainable"] is True
    assert model.peft_config["default"].base_model_name_or_path == (
        "autogluon/chronos-2-small")
    assert model.peft_config["default"].revision == revision
    assert json.loads((parent / "adapter_config.json").read_text()) == original


def test_native_reg_volume_losses_train_embeddings_without_temporary_heads():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(91)
    embeddings = torch.randn(
        (12, 8), generator=generator, requires_grad=True)
    instance = torch.arange(6).repeat(2)
    states = torch.tensor([
        [0, 0, 0],
        [0, 2, 2],
        [0, 0, 2],
        [2, 2, 0],
        [2, 0, 0],
        [2, 2, 2],
    ])

    losses, eligible = _native_reg_volume_structure_losses(
        embeddings,
        instance,
        states,
        temperature=0.1,
        require_all=True,
    )
    assert eligible == {
        "participation": 6,
        "concentration": 6,
        "displacement": 6,
    }
    assert all(torch.isfinite(value) for value in losses.values())

    changed_participation = states.clone()
    changed_participation[:, 0] = torch.tensor([0, 0, 2, 0, 2, 2])
    changed, _ = _native_reg_volume_structure_losses(
        embeddings,
        instance,
        changed_participation,
        temperature=0.1,
        require_all=True,
    )
    torch.testing.assert_close(
        losses["concentration"], changed["concentration"])

    sum(losses.values()).backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert float(embeddings.grad.abs().sum()) > 0.0


def test_native_volume_lift_requires_both_checkpoint_capabilities():
    assert _validate_native_volume_lift(
        {"participation": 1.5, "concentration": 1.2},
        {"participation": 1.4, "concentration": 1.0},
        margin=0.01,
    ) == {"participation": pytest.approx(0.1), "concentration": pytest.approx(0.2)}

    with pytest.raises(RuntimeError, match="concentration"):
        _validate_native_volume_lift(
            {"participation": 1.5, "concentration": 1.2},
            {"participation": 1.4, "concentration": 1.2},
            margin=0.0,
        )


def test_volume_structure_entrypoint_freezes_full_corpus_and_safe_defaults():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "chronos" / "chronos2_ssl_volume_structure.py"
    ).read_text()

    assert 'value.add_argument("--timeframes", default=",".join(TIMEFRAMES))' in source
    assert '"--base-snapshot"' in source
    assert 'value.add_argument("--epochs", type=int, default=60)' in source
    assert 'value.add_argument("--lr", type=float, default=1e-5)' in source
    assert 'value.add_argument("--patience", type=int, default=8)' in source
    assert "preflight_volume_structure_ssl" in source
    assert "participation_weight=args.participation_weight" in source
    assert "concentration_weight=args.concentration_weight" in source
    assert "head_auxiliary_weight=args.head_auxiliary_weight" in source
    assert "native_promotion_margin=args.native_promotion_margin" in source
    assert "base_snapshot=args.base_snapshot" in source


def test_visible_normalization_is_independent_of_hidden_patch_values():
    torch = pytest.importorskip("torch")
    length = 32
    close = np.linspace(100.0, 108.0, length, dtype=np.float32)
    raw = torch.from_numpy(_window(
        close,
        np.linspace(500.0, 1_500.0, length, dtype=np.float32),
    )[None])
    visible_time = torch.ones((1, length), dtype=torch.bool)
    visible_time[:, 8:12] = False

    first, first_finite, first_visible = _volume_structure_standardize(
        raw, visible_time=visible_time)
    changed = raw.clone()
    changed[:, :4, 8:12] = changed[:, :4, 8:12] * 100.0 + 10_000.0
    changed[:, 4, 8:12] = 1_000_000.0
    second, second_finite, second_visible = _volume_structure_standardize(
        changed, visible_time=visible_time)

    assert torch.equal(first_finite, second_finite)
    assert torch.equal(first_visible, second_visible)
    assert not first_visible[:, :, 8:12].any()
    torch.testing.assert_close(first[first_visible], second[second_visible])
    assert not torch.equal(first[:, :, 8:12], second[:, :, 8:12])

    first_context, first_mask = _volume_structure_encoder_input(
        first, first_visible)
    second_context, second_mask = _volume_structure_encoder_input(
        second, second_visible)
    assert torch.equal(first_mask, second_mask)
    assert torch.isnan(first_context[~first_mask]).all()
    assert torch.isnan(second_context[~second_mask]).all()

    # Exercise the same pre-mask normalization used by real Chronos-2. Hidden
    # target perturbations must not change any normalized visible value.
    from chronos.chronos_bolt import InstanceNorm
    normalizer = InstanceNorm()
    first_normalized, _ = normalizer(first_context)
    second_normalized, _ = normalizer(second_context)
    torch.testing.assert_close(
        first_normalized[first_mask], second_normalized[second_mask])


def test_volume_structure_input_preserves_scale_free_participation_direction():
    torch = pytest.importorskip("torch")
    length = 32
    close = np.linspace(100.0, 104.0, length, dtype=np.float32)
    volume = np.full(length, 1_000.0, dtype=np.float32)
    volume[-8:] *= 8.0
    raw = torch.from_numpy(_window(close, volume)[None])
    transformed = raw.clone()
    transformed[:, :4] = transformed[:, :4] * 9.0 + 77.0
    transformed[:, 4] *= 31.0

    first, _, _ = _volume_structure_standardize(raw)
    second, _, _ = _volume_structure_standardize(transformed)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
    assert float(first[:, 4, -8:].mean()) > float(first[:, 4, :-8].mean())


def test_volume_structure_patch_mask_hides_exact_complete_patches():
    torch = pytest.importorskip("torch")
    first_generator = torch.Generator(device="cpu").manual_seed(37)
    second_generator = torch.Generator(device="cpu").manual_seed(37)
    patch_mask = _volume_structure_patch_mask(
        batch=5,
        n_patches=8,
        mask_ratio=0.25,
        generator=first_generator,
        device="cpu",
    )
    repeated = _volume_structure_patch_mask(
        batch=5,
        n_patches=8,
        mask_ratio=0.25,
        generator=second_generator,
        device="cpu",
    )

    assert torch.equal(patch_mask, repeated)
    assert torch.equal(
        patch_mask.sum(dim=1), torch.full((5,), 2, dtype=torch.long))
    time_mask = patch_mask.repeat_interleave(4, dim=1)
    assert torch.equal(
        time_mask.sum(dim=1), torch.full((5,), 8, dtype=torch.long))
    assert torch.equal(time_mask.reshape(5, 8, 4).all(dim=2), patch_mask)
    assert torch.equal(time_mask.reshape(5, 8, 4).any(dim=2), patch_mask)

    low = _volume_structure_patch_mask(
        2, 8, 0.01, torch.Generator().manual_seed(1), "cpu")
    high = _volume_structure_patch_mask(
        2, 8, 0.99, torch.Generator().manual_seed(1), "cpu")
    assert low.sum(dim=1).tolist() == [1, 1]
    assert high.sum(dim=1).tolist() == [7, 7]


def _prepared_volume_structure_fixture() -> SimpleNamespace:
    tickers = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
    channels = ("open", "high", "low", "close", "volume")

    def stream(rows: int, *, phase: float, offset: float) -> np.ndarray:
        time = np.arange(rows, dtype=np.float32) + phase
        close = (
            100.0
            + offset
            + np.sin(time / 11.0) * (1.0 + time / rows * 5.0)
            + 0.2 * np.sin(time / 2.7)
        )
        width = 0.2 + 1.8 * (0.5 + 0.5 * np.sin(time / 9.0)) ** 2
        volume = 1_000.0 * np.exp(
            1.1 * np.sin(time / 13.0) + 0.35 * np.sin(time / 3.1))
        return _window(
            close,
            volume,
            high=close + width,
            low=close - width,
        )

    return SimpleNamespace(
        train=np.concatenate([
            stream(320, phase=0.0, offset=index * 10.0)
            for index in range(len(tickers))
        ]),
        validation_matrix=np.concatenate([
            stream(160, phase=2.0, offset=index * 10.0)
            for index in range(len(tickers))
        ]),
        channel_names=tuple(
            f"{ticker}.{channel}"
            for ticker in tickers
            for channel in channels
        ),
        report={"identity_sha256": "d" * 64, "timeframe": "3min"},
    )


def test_volume_structure_mocked_cpu_trainer_smoke(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeBase:
        model_dim = 8
        chronos_config = SimpleNamespace(
            input_patch_size=4,
            input_patch_stride=4,
            use_reg_token=True,
            use_arcsinh=False,
        )

        def __init__(self, owner):
            self.owner = owner

        def encode(
                self,
                *,
                context,
                context_mask,
                group_ids,
                num_output_patches,
        ):
            del group_ids, num_output_patches
            patch_size = self.chronos_config.input_patch_size
            n_patches = context.shape[-1] // patch_size
            loc = torch.nan_to_num(
                torch.nanmean(context.float(), dim=-1, keepdim=True), nan=0.0)
            normalization_scale = torch.nan_to_num(
                (context.float() - loc).square().nanmean(
                    dim=-1, keepdim=True).sqrt(),
                nan=1.0,
            ).clamp_min(1e-5)
            normalized = (context - loc) / normalization_scale
            clean = torch.where(
                context_mask, normalized, torch.zeros_like(normalized))
            patch_values = clean.reshape(-1, n_patches, patch_size)
            patch_visible = context_mask.reshape(-1, n_patches, patch_size)
            counts = patch_visible.sum(dim=2).clamp_min(1)
            patch_mean = patch_values.sum(dim=2) / counts
            basis = torch.arange(
                1,
                self.model_dim + 1,
                dtype=context.dtype,
                device=context.device,
            )
            tokens = (
                patch_mean[:, :, None] * self.owner.lora_weight[None, None, :]
                + patch_mean.square()[:, :, None] * basis[None, None, :] * 0.01
            )
            register = tokens.mean(dim=1, keepdim=True)
            return (
                [torch.cat((tokens, register), dim=1)],
                (loc, normalization_scale),
                None,
                n_patches,
            )

    class FakeModel(torch.nn.Module):
        def __init__(self, base_revision):
            super().__init__()
            self.base_revision = base_revision
            self.lora_weight = torch.nn.Parameter(
                torch.linspace(0.1, 0.8, 8, dtype=torch.float32))
            self.base = FakeBase(self)

        def save_pretrained(self, destination):
            destination = Path(destination)
            destination.mkdir(parents=True)
            (destination / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "autogluon/chronos-2-small",
                "peft_type": "LORA",
                "revision": self.base_revision,
            }) + "\n")
            (destination / "adapter_model.safetensors").write_bytes(
                self.lora_weight.detach().cpu().numpy().tobytes())

    base_revision = "b" * 40
    model = FakeModel(base_revision)
    initial_lora = model.lora_weight.detach().clone()
    loaded_models = []

    def load_adapter(
            parent,
            device,
            *,
            base_revision=None,
            base_snapshot=None,
    ):
        del device
        assert base_revision == "b" * 40
        assert Path(base_snapshot).name == base_revision
        if Path(parent).name == VOLUME_STRUCTURE_STAGED_CHECKPOINT:
            loaded = FakeModel(base_revision)
            values = np.frombuffer(
                (Path(parent) / "adapter_model.safetensors").read_bytes(),
                dtype=np.float32,
            ).copy()
            with torch.no_grad():
                loaded.lora_weight.copy_(torch.from_numpy(values))
            loaded_models.append(loaded)
            return loaded, loaded.base
        loaded_models.append(model)
        return model, model.base

    monkeypatch.setattr(
        ssl_stages,
        "_load_trainable_adapter",
        load_adapter,
    )
    monkeypatch.setattr(
        ssl_stages,
        "_adapter_state",
        lambda value: {
            "lora_weight": value.lora_weight.detach().cpu().clone(),
        },
    )

    def restore(value, state):
        with torch.no_grad():
            value.lora_weight.copy_(state["lora_weight"])

    monkeypatch.setattr(ssl_stages, "_restore_adapter", restore)
    # The deliberately tiny fake encoder has nearly scale-only REG geometry,
    # so its one-step contrastive loss is not a meaningful promotion signal.
    # The pure gate is tested separately above; this smoke covers publication.
    monkeypatch.setattr(
        ssl_stages,
        "_validate_native_volume_lift",
        lambda parent, child, *, margin: {
            name: float(parent[name] - child[name])
            for name in ("participation", "concentration")
        },
    )

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "autogluon/chronos-2-small",
        "peft_type": "LORA",
        "revision": None,
    }) + "\n")
    base_snapshot = tmp_path / base_revision
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}\n")
    (base_snapshot / "model.safetensors").write_bytes(b"pinned base")
    parent_sha256 = ssl_stages.tree_sha256(parent)
    output = tmp_path / "volume"
    report = ssl_stages.train_volume_structure_ssl(
        {"3min": _prepared_volume_structure_fixture()},
        parent=parent,
        base_snapshot=base_snapshot,
        out_dir=output,
        device="cpu",
        context_length=32,
        epochs=1,
        steps_per_epoch=1,
        batch_windows=6,
        gradient_accumulation=1,
        learning_rate=1e-3,
        patience=1,
        projection_dim=8,
        noise=0.0,
        scale=0.0,
        mask_ratio=0.25,
        threshold_samples=16,
        validation_windows_per_stream=4,
        price_bins=16,
        log_every_steps=1,
        seed=7,
    )

    saved = json.loads((output / "report.json").read_text())
    assert report["schema"] == VOLUME_STRUCTURE_REPORT_SCHEMA
    assert report["stage"] == "volume_structure_ssl"
    assert report["status"] == "complete"
    assert report["checkpoint"]["sha256"] == ssl_stages.tree_sha256(
        output / "checkpoint")
    assert report["parent"]["sha256"] == parent_sha256
    assert ssl_stages.tree_sha256(parent) == parent_sha256
    assert saved["run_identity_sha256"] == report["run_identity_sha256"]
    assert saved["checkpoint"] == report["checkpoint"]
    assert len(report["history"]) == 1
    assert len(report["history"][0]["val_per_stream"]) == 9
    assert {
        "participation",
        "participation_native",
        "participation_head",
        "concentration",
        "concentration_native",
        "concentration_head",
        "displacement",
        "displacement_native",
        "displacement_head",
    } <= report["history"][0]["train_components"].keys()
    assert {
        "participation",
        "participation_native",
        "participation_head",
        "concentration",
        "concentration_native",
        "concentration_head",
        "displacement",
        "displacement_native",
        "displacement_head",
    } <= report["history"][0]["val_components"].keys()
    native = report["checkpoint_only_validation"]
    assert native["status"] == "pass"
    assert native["contract"] == (
        "freshly_reloaded_lora_native_reg_without_temporary_heads")
    assert set(native["parent"]["aggregate"]) == {
        "participation", "concentration", "displacement"}
    assert set(native["checkpoint"]["aggregate"]) == {
        "participation", "concentration", "displacement"}
    assert set(native["loss_lift_parent_minus_checkpoint"]) == {
        "participation", "concentration"}
    assert len(loaded_models) == 2
    assert loaded_models[0] is model
    assert loaded_models[1] is not model
    assert not torch.equal(model.lora_weight.detach(), initial_lora)
    assert not (output / "trainer.pt").exists()
    assert not (output / VOLUME_STRUCTURE_STAGED_CHECKPOINT).exists()
    assert not (output / VOLUME_STRUCTURE_STAGED_REPORT).exists()
    assert (output / "checkpoint" / "adapter_config.json").is_file()
    checkpoint_files = [
        path.name.lower()
        for path in (output / "checkpoint").rglob("*")
        if path.is_file()
    ]
    assert checkpoint_files
    assert not any(
        token in name
        for name in checkpoint_files
        for token in ("head", "decoder", "projection", "trainer")
    )
    artifact = report["final_artifact_contract"]
    assert artifact["ssl_heads_required_for_inference"] is False
    assert artifact["temporary_heads_in_checkpoint"] is False
    assert artifact["trainer_state"] == (
        "discarded_after_successful_checkpoint")
    assert artifact["inference_requires"] == [
        "chronos_base_model", "lora_checkpoint"]


def _completion_reports(output, checkpoint_sha256, run_identity):
    finalizing = {
        "schema": VOLUME_STRUCTURE_REPORT_SCHEMA,
        "stage": "volume_structure_ssl",
        "status": "finalizing",
        "run_identity_sha256": run_identity,
        "checkpoint": {
            "path": str(output / "checkpoint"),
            "sha256": checkpoint_sha256,
        },
        "config": {
            "checkpoint_selection": {
                "contract": (
                    "gate_feasible_weighted_native_reg_"
                    "participation_concentration_v1"),
                "temporary_head_metrics_used": False,
            },
            "base_model": {
                "model_id": "autogluon/chronos-2-small",
                "revision": "b" * 40,
                "weights_sha256": "c" * 64,
                "config_sha256": "d" * 64,
            },
        },
        "checkpoint_only_validation": {
            "status": "pass",
            "contract": (
                "freshly_reloaded_lora_native_reg_without_temporary_heads"),
            "parent": {
                "aggregate": {"participation": 1.2, "concentration": 1.1},
            },
            "checkpoint": {
                "aggregate": {"participation": 1.0, "concentration": 0.8},
            },
            "loss_lift_parent_minus_checkpoint": {
                "participation": 0.2,
                "concentration": 0.3,
            },
            "required_margin": 1e-4,
        },
        "final_artifact_contract": {
            "checkpoint_files": [
                "adapter_config.json", "adapter_model.safetensors"],
            "temporary_heads_in_checkpoint": False,
            "ssl_heads_required_for_inference": False,
            "trainer_state": "discarded_after_successful_checkpoint",
            "inference_requires": ["chronos_base_model", "lora_checkpoint"],
        },
    }
    return finalizing, {**finalizing, "status": "complete"}


def test_volume_structure_finalization_recovery_discards_only_trainer(tmp_path):
    torch = pytest.importorskip("torch")
    output = tmp_path / "volume"
    staged_checkpoint = output / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    staged_checkpoint.mkdir(parents=True)
    (staged_checkpoint / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "autogluon/chronos-2-small",
    }))
    (staged_checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    checkpoint_sha256 = ssl_stages.tree_sha256(staged_checkpoint)
    run_identity = "a" * 64
    finalizing, complete = _completion_reports(
        output, checkpoint_sha256, run_identity)
    (output / "report.json").write_text(json.dumps(finalizing))
    (output / VOLUME_STRUCTURE_STAGED_REPORT).write_text(json.dumps(complete))
    torch.save({
        "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
        "run_identity_sha256": run_identity,
    }, output / "trainer.pt")
    preserved = output / "preflight.json"
    preserved.write_text("{}")

    recovered = _recover_volume_structure_finalization(
        output, run_identity_sha256=run_identity)

    assert recovered == complete
    assert (output / "checkpoint").is_dir()
    assert not staged_checkpoint.exists()
    assert not (output / "trainer.pt").exists()
    assert not (output / VOLUME_STRUCTURE_STAGED_REPORT).exists()
    assert json.loads((output / "report.json").read_text())["status"] == "complete"
    assert preserved.is_file()


def test_volume_structure_recovery_rejects_mismatched_run_without_mutation(
        tmp_path):
    torch = pytest.importorskip("torch")
    output = tmp_path / "volume"
    staged_checkpoint = output / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    staged_checkpoint.mkdir(parents=True)
    (staged_checkpoint / "adapter_config.json").write_text("{}")
    (staged_checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    checkpoint_sha256 = ssl_stages.tree_sha256(staged_checkpoint)
    old_identity = "a" * 64
    current_identity = "b" * 64
    finalizing, complete = _completion_reports(
        output, checkpoint_sha256, old_identity)
    (output / "report.json").write_text(json.dumps(finalizing))
    (output / VOLUME_STRUCTURE_STAGED_REPORT).write_text(json.dumps(complete))
    torch.save({
        "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
        "run_identity_sha256": old_identity,
    }, output / "trainer.pt")

    with pytest.raises(RuntimeError, match="identity"):
        _recover_volume_structure_finalization(
            output, run_identity_sha256=current_identity)

    assert staged_checkpoint.is_dir()
    assert (output / "trainer.pt").is_file()
    assert (output / VOLUME_STRUCTURE_STAGED_REPORT).is_file()
    assert json.loads((output / "report.json").read_text()) == finalizing

    (output / "trainer.pt").write_bytes(b"corrupt trainer")
    with pytest.raises(RuntimeError, match="trainer state is unreadable"):
        _recover_volume_structure_finalization(
            output, run_identity_sha256=old_identity)
    assert staged_checkpoint.is_dir()
    assert (output / "trainer.pt").read_bytes() == b"corrupt trainer"
    assert (output / VOLUME_STRUCTURE_STAGED_REPORT).is_file()


def test_volume_structure_train_never_recovers_before_parent_authentication(
        tmp_path):
    output = tmp_path / "volume"
    output.mkdir()
    trainer = output / "trainer.pt"
    trainer.write_bytes(b"must not be deleted")

    with pytest.raises(RuntimeError, match="parent adapter is missing"):
        ssl_stages.train_volume_structure_ssl(
            {},
            parent=tmp_path / "missing-parent",
            base_snapshot=tmp_path / ("b" * 40),
            out_dir=output,
            device="cpu",
            context_length=32,
            epochs=1,
            steps_per_epoch=1,
            batch_windows=6,
            threshold_samples=16,
            validation_windows_per_stream=4,
            resume=True,
        )

    assert trainer.read_bytes() == b"must not be deleted"


def test_volume_structure_recovery_rejects_staged_checkpoint_without_trainer(
        tmp_path):
    output = tmp_path / "volume"
    staged = output / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    staged.mkdir(parents=True)
    (staged / "adapter_config.json").write_text("{}")
    (staged / "adapter_model.safetensors").write_bytes(b"adapter")
    identity = "a" * 64
    finalizing, complete = _completion_reports(
        output, ssl_stages.tree_sha256(staged), identity)
    (output / "report.json").write_text(json.dumps(finalizing))
    (output / VOLUME_STRUCTURE_STAGED_REPORT).write_text(json.dumps(complete))

    with pytest.raises(RuntimeError, match="without trainer state"):
        _recover_volume_structure_finalization(
            output, run_identity_sha256=identity)

    assert staged.is_dir()
    assert (output / VOLUME_STRUCTURE_STAGED_REPORT).is_file()
    assert json.loads((output / "report.json").read_text()) == finalizing


def test_volume_structure_recovery_rejects_committed_run_with_extra_staging(
        tmp_path):
    output = tmp_path / "volume"
    checkpoint = output / "checkpoint"
    staged = output / VOLUME_STRUCTURE_STAGED_CHECKPOINT
    for directory in (checkpoint, staged):
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}")
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    identity = "a" * 64
    _, complete = _completion_reports(
        output, ssl_stages.tree_sha256(checkpoint), identity)
    (output / "report.json").write_text(json.dumps(complete))

    with pytest.raises(RuntimeError, match="extra staged checkpoint"):
        _recover_volume_structure_finalization(
            output, run_identity_sha256=identity)

    assert checkpoint.is_dir()
    assert staged.is_dir()
    assert json.loads((output / "report.json").read_text()) == complete
