"""Torch-free contracts for the Chronos-2 multivariate MVP and classifier seam."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.classifiers.chronos2.multivariate import (
    OHLCV_COLS,
    prepare_3min_multivariate,
    prepare_multivariate,
)


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_streams(directory: Path, *, rows: int = 620) -> None:
    timestamps = pd.date_range("2024-12-30T00:00:00Z", periods=rows, freq="3min")
    for number, ticker in enumerate(TICKERS):
        stream_ts = timestamps.delete(75) if ticker == "NQ" else timestamps
        base = 100.0 + number + np.arange(len(stream_ts), dtype=float) * 0.01
        frame = pd.DataFrame({
            "datetime": stream_ts,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": 1_000.0 + number,
        })
        path = directory / f"{ticker}_3min.csv"
        frame.to_csv(path, index=False)
        manifest = {
            "ticker": ticker,
            "timeframe": "3min",
            "schema": "ffm_continuous_contract_v1",
            "selection": "cme_session_total_volume",
            "back_adjusted": True,
            "output_sha256": _sha256(path),
            "rows": len(frame),
            "start": str(stream_ts[0]),
            "end": str(stream_ts[-1]),
            "source_sha256": f"source-{ticker}",
        }
        path.with_suffix(".csv.manifest.json").write_text(json.dumps(manifest))


def test_multivariate_alignment_is_joint_close_time_and_never_fills(tmp_path):
    _write_streams(tmp_path)
    result = prepare_3min_multivariate(
        tmp_path,
        tickers=TICKERS,
        holdout_start="2024-12-31T06:00:00Z",
        val_frac=0.2,
        context_length=16,
        prediction_length=4,
        validation_windows=4,
    )
    assert result.train.shape[0] == 45
    assert result.channel_names[:5] == tuple(f"ES.{column}" for column in OHLCV_COLS)
    assert result.channel_names[5:10] == tuple(f"NQ.{column}" for column in OHLCV_COLS)
    # The missing NQ bar stays absent on the union timestamp; ES remains observed.
    missing_close = pd.Timestamp("2024-12-30T03:48:00Z")
    row = int(np.where(pd.DatetimeIndex(result.train_close_times) == missing_close)[0][0])
    assert np.isfinite(result.train[0:5, row]).all()
    assert np.isnan(result.train[5:10, row]).all()
    assert result.report["alignment"] == "union_of_bar_close_times_no_fill"
    assert result.report["n_variates"] == 45


def test_multivariate_targets_stay_before_holdout_and_identity_is_stable(tmp_path):
    _write_streams(tmp_path)
    kwargs = dict(
        tickers=TICKERS,
        holdout_start="2024-12-31T06:00:00Z",
        val_frac=0.2,
        context_length=16,
        prediction_length=4,
        validation_windows=4,
    )
    first = prepare_3min_multivariate(tmp_path, **kwargs)
    second = prepare_3min_multivariate(tmp_path, **kwargs)
    assert first.report["identity_sha256"] == second.report["identity_sha256"]
    holdout = pd.Timestamp(kwargs["holdout_start"])
    assert pd.Timestamp(first.report["aligned_close_end"]) < holdout
    for start, end in first.validation_target_ranges:
        assert pd.Timestamp(start) < holdout
        assert pd.Timestamp(end) < holdout


def test_multivariate_fails_closed_on_bad_source_ohlc(tmp_path):
    _write_streams(tmp_path)
    path = tmp_path / "ES_3min.csv"
    frame = pd.read_csv(path)
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1
    frame.to_csv(path, index=False)
    manifest_path = path.with_suffix(".csv.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="OHLC"):
        prepare_3min_multivariate(
            tmp_path, tickers=TICKERS, holdout_start="2024-12-31T06:00:00Z",
            context_length=16, prediction_length=4)


def test_chronos2_classifier_is_lazy_and_matches_seam(monkeypatch):
    from futures_foundation.finetune.classifier import Classifier, get_classifier
    from futures_foundation.finetune.classifiers.chronos2 import frozen

    source = ast.parse(Path(frozen.__file__).read_text())
    imported = {
        name.name.split(".")[0]
        for node in ast.walk(source) if isinstance(node, ast.Import)
        for name in node.names
    }
    imported |= {
        node.module.split(".")[0]
        for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" not in imported

    monkeypatch.setenv("EMBED_CACHE", "0")
    classifier = get_classifier(
        "chronos2_frozen", device="cpu", with_features=False)
    assert isinstance(classifier, Classifier)
    windows = np.ones((3, 5, 32), np.float32)

    class Labeler:
        MV_SEQ = 32
        MV_MODE = "ohlcv"

        def mv_contexts(self, keys):
            return windows[np.asarray(keys)]

    monkeypatch.setattr(
        classifier, "_embed",
        lambda labeler, keys: np.arange(len(keys) * 12, dtype=np.float32).reshape(len(keys), 12))
    features = classifier.featurize(Labeler(), [0, 1, 2])
    assert features.shape == (3, 12, 1)


def test_chronos2_alias_resolves_same_classifier():
    from futures_foundation.finetune.classifier import get_classifier
    assert type(get_classifier("chronos2")) is type(get_classifier("chronos2_frozen"))


def test_chronos2_serving_rejects_multi_ticker_tensor(monkeypatch):
    from futures_foundation.finetune.classifier import get_classifier

    monkeypatch.setenv("EMBED_CACHE", "0")
    classifier = get_classifier(
        "chronos2_frozen", device="cpu", with_features=False)

    class Labeler:
        MV_SEQ = 32
        MV_MODE = "ohlcv"

        def mv_contexts(self, keys):
            return np.ones((len(keys), 45, self.MV_SEQ), np.float32)

    with pytest.raises(ValueError, match="exactly one ticker"):
        classifier.featurize(Labeler(), [0])


def test_generic_multivariate_rejects_unknown_timeframe(tmp_path):
    with pytest.raises(ValueError, match="unsupported timeframe"):
        prepare_multivariate(tmp_path, timeframe="2min")


def test_chronos2_kaufman_regime_matches_mantis_threshold_contract():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _kaufman_regime,
    )

    raw = torch.zeros((4, 5, 5), dtype=torch.float32)
    raw[0, 3] = torch.tensor([0, 1, 0, 1, 0])  # ER=0: chop
    raw[1, 3] = torch.tensor([0, 1, 2, 3, 4])  # ER=1: uptrend
    raw[2, 3] = torch.tensor([4, 3, 2, 1, 0])  # ER=1: downtrend
    raw[3, 3] = torch.tensor([0, 2, 0, 2, 2])  # ER=1/3: transition
    regime, efficiency = _kaufman_regime(
        raw, chop=0.25, trend=0.50)

    assert regime.tolist() == [0, 1, 2, -1]
    assert efficiency.tolist() == pytest.approx([0.0, 1.0, 1.0, 1 / 3])


def test_balanced_kaufman_er_is_causal_scale_free_and_direction_agnostic():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _kaufman_binary_states,
        _kaufman_efficiency_scores,
    )

    close = np.asarray([
        0, 1, 2, 3, 4,       # efficient up move
        4, 3, 2, 1, 0,       # efficient down move
        0, 1, 0, 1, 0,       # chop
    ], dtype=np.float32)
    matrix = np.zeros((5, len(close)), dtype=np.float32)
    matrix[3] = close
    starts = np.asarray([0, 5, 10], dtype=np.int64)
    efficiency = _kaufman_efficiency_scores(
        matrix, starts, context_length=5, kaufman_length=5)
    transformed = matrix.copy()
    transformed[3] = transformed[3] * 37.0 + 123.0
    transformed_efficiency = _kaufman_efficiency_scores(
        transformed, starts, context_length=5, kaufman_length=5)

    np.testing.assert_allclose(efficiency, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(transformed_efficiency, efficiency)
    assert _kaufman_binary_states(
        efficiency, chop=0.25, trend=0.50).tolist() == [2, 2, 0]

    future_changed = matrix.copy()
    future_changed[:, 5:] = 1_000_000.0
    first_before = _kaufman_efficiency_scores(
        matrix, np.asarray([0]), context_length=5, kaufman_length=5)
    first_after = _kaufman_efficiency_scores(
        future_changed, np.asarray([0]), context_length=5, kaufman_length=5)
    np.testing.assert_array_equal(first_before, first_after)


def test_balanced_kaufman_states_preserve_fixed_threshold_boundaries():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _kaufman_binary_states,
    )

    states = _kaufman_binary_states(
        np.asarray([0.0, 0.25, 0.251, 0.499, 0.50, 1.0]),
        chop=0.25,
        trend=0.50,
    )
    assert states.tolist() == [0, 0, -1, -1, 2, 2]


def test_balanced_kaufman_native_reg_concat_matches_serving_channel_order():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _reg_embeddings_concat,
    )

    class FakeBase:
        def encode(
                self, *, context, context_mask, group_ids,
                num_output_patches):
            del context_mask, group_ids, num_output_patches
            mean = context.mean(dim=1)
            basis = torch.asarray(
                [1.0, 2.0, 3.0], dtype=context.dtype,
                device=context.device)
            register = mean[:, None] * basis[None, :]
            return [register[:, None, :]], None, None, 0

    windows = torch.stack([
        torch.full((4,), float(channel + 1)) for channel in range(5)
    ])[None].requires_grad_(True)
    finite = torch.ones_like(windows, dtype=torch.bool)
    embeddings = _reg_embeddings_concat(FakeBase(), windows, finite)

    expected = torch.cat([
        torch.asarray([value, 2 * value, 3 * value])
        for value in range(1, 6)
    ])[None]
    assert embeddings.shape == (1, 15)
    assert torch.equal(embeddings, expected)
    embeddings.sum().backward()
    assert windows.grad is not None
    assert (windows.grad.abs().sum(dim=2) > 0).all()

    with pytest.raises(ValueError, match=r"\[B,5,L\]"):
        _reg_embeddings_concat(
            FakeBase(), windows[:, :4], finite[:, :4])


def test_balanced_kaufman_native_objective_rewards_state_separation():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _native_balanced_kaufman_metrics,
    )

    states = torch.asarray([0, 0, 2, 2], dtype=torch.long)
    instance = torch.arange(4).repeat(2)
    separated_one_view = torch.asarray([
        [1.0, 0.05], [1.0, -0.05], [-1.0, 0.05], [-1.0, -0.05],
    ])
    mixed_one_view = torch.asarray([
        [1.0, 0.05], [-1.0, -0.05], [1.0, -0.05], [-1.0, 0.05],
    ])
    separated = _native_balanced_kaufman_metrics(
        torch.cat((separated_one_view, separated_one_view), dim=0),
        instance,
        states,
        temperature=0.10,
        require_both=True,
    )
    mixed = _native_balanced_kaufman_metrics(
        torch.cat((mixed_one_view, mixed_one_view), dim=0),
        instance,
        states,
        temperature=0.10,
        require_both=True,
    )

    assert separated["loss"] < mixed["loss"]
    assert separated["margin"] > mixed["margin"]
    assert separated["class_counts"] == {"0": 2, "2": 2}


def test_balanced_kaufman_native_gate_cannot_be_passed_by_projection_only():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _validate_native_balanced_kaufman_lift,
    )

    parent = {"loss": 1.0, "margin": 0.2, "embedding_std": 0.1}
    child = {"loss": 1.01, "margin": 0.19, "embedding_std": 0.1}
    with pytest.raises(RuntimeError, match="did not improve"):
        _validate_native_balanced_kaufman_lift(
            parent, child, margin=1e-4)


def test_balanced_kaufman_pools_are_per_stream_fixed_and_exactly_balanced():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _prepare_balanced_kaufman_data,
    )

    def matrix(close):
        close = np.asarray(close, dtype=np.float32)
        return np.stack((
            close,
            close + 1.0,
            close - 1.0,
            close,
            np.ones_like(close) * 1_000.0,
        ))

    chop = np.tile(np.asarray([100.0, 101.0]), 40)
    trend = np.arange(80, dtype=np.float32) + 100.0
    prepared = types.SimpleNamespace(
        train=matrix(np.concatenate((chop, trend))),
        validation_matrix=matrix(np.concatenate((chop, trend))),
        channel_names=tuple(
            f"ES.{column}" for column in OHLCV_COLS),
        report={"identity_sha256": "a" * 64, "timeframe": "3min"},
    )
    data = _prepare_balanced_kaufman_data(
        {"3min": prepared},
        context_length=32,
        kaufman_length=16,
        kaufman_chop=0.25,
        kaufman_trend=0.50,
        validation_windows_per_state=4,
    )
    stream = data["streams"][("3min", "ES")]
    assert len(stream["train_state_starts"][0]) >= 2
    assert len(stream["train_state_starts"][2]) >= 2
    assert (stream["validation_states"] == 0).sum() == 4
    assert (stream["validation_states"] == 2).sum() == 4
    assert data["preflight_streams"]["3min"]["ES"][
        "validation_selected"] == {"chop": 4, "trend": 4}

    no_validation_trend = types.SimpleNamespace(
        **{
            **prepared.__dict__,
            "validation_matrix": matrix(np.tile(
                np.asarray([100.0, 101.0]), 80)),
        }
    )
    with pytest.raises(RuntimeError, match="validation support"):
        _prepare_balanced_kaufman_data(
            {"3min": no_validation_trend},
            context_length=32,
            kaufman_length=16,
            kaufman_chop=0.25,
            kaufman_trend=0.50,
            validation_windows_per_state=4,
        )


def test_balanced_kaufman_sampler_is_exactly_half_chop_half_trend():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _sample_balanced_kaufman_batch,
    )

    matrix = np.stack([
        np.arange(128, dtype=np.float32) + channel * 1_000
        for channel in range(5)
    ])
    streams = {
        ("3min", "ES"): {
            "train_matrix": matrix,
            "train_state_starts": {
                0: np.arange(0, 32, dtype=np.int64),
                2: np.arange(64, 96, dtype=np.int64),
            },
        },
    }
    windows, states = _sample_balanced_kaufman_batch(
        streams,
        batch_windows=8,
        context_length=16,
        rng=np.random.default_rng(7),
    )

    assert windows.shape == (8, 5, 16)
    assert (states == 0).sum() == 4
    assert (states == 2).sum() == 4
    starts = windows[:, 0, 0]
    assert len(np.unique(starts)) == len(starts)


def test_balanced_kaufman_identity_binds_volume_report_data_and_config():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _balanced_kaufman_run_identity,
    )

    kwargs = {
        "parent_sha256": "a" * 64,
        "parent_report_sha256": "b" * 64,
        "parent_report_run_identity_sha256": "c" * 64,
        "data_identity_sha256": "d" * 64,
        "config": {"context_length": 256, "timeframes": ["3min"]},
    }
    identity = _balanced_kaufman_run_identity(**kwargs)
    assert len(identity) == 64
    for key, replacement in (
        ("parent_sha256", "e" * 64),
        ("parent_report_sha256", "f" * 64),
        ("parent_report_run_identity_sha256", "1" * 64),
        ("data_identity_sha256", "2" * 64),
    ):
        assert identity != _balanced_kaufman_run_identity(
            **{**kwargs, key: replacement})
    assert identity != _balanced_kaufman_run_identity(
        **{**kwargs, "config": {"context_length": 128, "timeframes": ["3min"]}})


def test_balanced_kaufman_resume_rejects_identity_drift():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        BALANCED_KAUFMAN_TRAINER_SCHEMA,
        _validate_balanced_kaufman_resume,
    )

    saved = {
        "schema": BALANCED_KAUFMAN_TRAINER_SCHEMA,
        "run_identity_sha256": "a" * 64,
    }
    _validate_balanced_kaufman_resume(
        saved, run_identity_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="identity"):
        _validate_balanced_kaufman_resume(
            saved, run_identity_sha256="b" * 64)


def test_balanced_kaufman_public_training_contract_is_fixed():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        train_balanced_kaufman_ssl,
    )

    signature = inspect.signature(train_balanced_kaufman_ssl).parameters
    assert signature["context_length"].default == 256
    assert signature["kaufman_length"].default == 64
    assert signature["kaufman_chop"].default == 0.25
    assert signature["kaufman_trend"].default == 0.50
    assert signature["epochs"].default == 60
    assert signature["steps_per_epoch"].default == 100
    assert signature["batch_windows"].default == 32
    assert signature["learning_rate"].default == 5e-6
    assert signature["patience"].default == 8
    assert signature["head_auxiliary_weight"].default == 0.25
    assert signature["adapter_retention_weight"].default == 0.1
    assert signature["native_promotion_margin"].default == 1e-4
    assert signature["validation_windows_per_state"].default == 16
    assert signature["log_every_steps"].default == 10

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "chronos" / "chronos2_ssl_contrastive.py"
    )
    namespace = runpy.run_path(str(script))
    args = namespace["parser"]().parse_args([])
    assert args.regime_key == "kaufman"
    assert args.base_snapshot is None
    assert args.timeframes == "1min,3min,5min,15min"
    assert args.parent.name == "checkpoint"
    assert args.parent.parent.name == "volume_structure_ssl_v3_seed0"
    assert args.parent_report == args.parent.parent / "report.json"
    assert args.kaufman_chop == 0.25
    assert args.kaufman_trend == 0.50
    assert args.kaufman_length == 64
    assert args.batch_windows == 32
    assert args.gradient_accumulation == 1
    assert args.lr == 5e-6
    assert args.epochs == 60
    assert args.steps == 100
    assert args.patience == 8
    assert args.head_auxiliary_weight == 0.25
    assert args.adapter_retention_weight == 0.1
    assert args.native_promotion_margin == 1e-4
    assert args.validation_windows_per_state == 16
    assert args.log_every_steps == 10


def _run_mocked_balanced_kaufman_trainer(
        tmp_path, monkeypatch, *, final_native_gate_passes):
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2 import ssl_stages

    revision = "b" * 40
    parent = tmp_path / "volume-parent" / "checkpoint"
    parent.mkdir(parents=True)
    (parent / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "autogluon/chronos-2-small",
        "peft_type": "LORA",
        "revision": revision,
    }) + "\n")
    (parent / "adapter_model.safetensors").write_bytes(b"immutable-parent")
    base_snapshot = tmp_path / revision
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}\n")
    (base_snapshot / "model.safetensors").write_bytes(b"pinned-base")
    parent_report = parent.parent / "report.json"
    parent_checkpoint_files = sorted(
        str(path.relative_to(parent))
        for path in parent.rglob("*") if path.is_file()
    )
    parent_report.write_text(json.dumps({
        "schema": ssl_stages.VOLUME_STRUCTURE_REPORT_SCHEMA,
        "stage": "volume_structure_ssl",
        "status": "complete",
        "run_identity_sha256": "c" * 64,
        "data_identity_sha256": "e" * 64,
        "checkpoint": {
            "path": str(parent.resolve()),
            "sha256": ssl_stages.tree_sha256(parent),
        },
        "checkpoint_only_validation": {
            "status": "pass",
            "contract": (
                "freshly_reloaded_lora_native_reg_without_temporary_heads"),
            "parent": {"aggregate": {
                "participation": 1.0, "concentration": 1.0}},
            "checkpoint": {"aggregate": {
                "participation": 0.5, "concentration": 0.5}},
            "loss_lift_parent_minus_checkpoint": {
                "participation": 0.5, "concentration": 0.5},
            "required_margin": 1e-4,
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
                "snapshot_path": str(base_snapshot.resolve()),
                "revision": revision,
                "weights_sha256": "f" * 64,
                "config_sha256": "1" * 64,
            },
        },
        "final_artifact_contract": {
            "checkpoint_files": parent_checkpoint_files,
            "temporary_heads_in_checkpoint": False,
            "ssl_heads_required_for_inference": False,
            "trainer_state": "discarded_after_successful_checkpoint",
            "inference_requires": ["chronos_base_model", "lora_checkpoint"],
        },
    }) + "\n")
    parent_before = {
        path.relative_to(parent): path.read_bytes()
        for path in parent.rglob("*") if path.is_file()
    }
    parent_report_before = parent_report.read_bytes()

    prepared_item = types.SimpleNamespace(
        train=np.zeros((5, 300), dtype=np.float32),
        validation_matrix=np.zeros((5, 300), dtype=np.float32),
        channel_names=tuple(f"ES.{column}" for column in OHLCV_COLS),
        report={"identity_sha256": "d" * 64, "timeframe": "3min"},
    )
    prepared = {"3min": prepared_item}
    data_identity = ssl_stages._corpus_identity(prepared)
    parent_identity = {
        "path": str(parent.resolve()),
        "sha256": ssl_stages.tree_sha256(parent),
    }
    parent_report_identity = {
        "path": str(parent_report.resolve()),
        "sha256": _sha256(parent_report),
        "schema": ssl_stages.VOLUME_STRUCTURE_REPORT_SCHEMA,
        "stage": "volume_structure_ssl",
        "run_identity_sha256": "c" * 64,
        "data_identity_sha256": "e" * 64,
    }
    base_identity = {
        "model_id": "autogluon/chronos-2-small",
        "snapshot_path": str(base_snapshot.resolve()),
        "revision": revision,
        "weights_sha256": "f" * 64,
        "config_sha256": "1" * 64,
    }
    authentication_calls = []

    def authenticate(**kwargs):
        authentication_calls.append(kwargs)
        assert kwargs["parent"] == parent.resolve()
        assert kwargs["parent_report"] == parent_report.resolve()
        assert kwargs["base_snapshot"] == base_snapshot.resolve()
        assert kwargs["data_identity_sha256"] == data_identity
        return parent_identity, parent_report_identity, base_identity

    monkeypatch.setattr(
        ssl_stages, "_authenticate_balanced_kaufman_parent", authenticate)
    data = {
        "corpus": prepared,
        "data_identity_sha256": data_identity,
        "aggregate_state_counts": {"chop": 8, "trend": 8},
        "streams": {("3min", "ES"): {}},
        "preflight_streams": {
            "3min": {
                "ES": {
                    "train_state_counts": {"chop": 8, "trend": 8},
                    "validation_selected": {"chop": 2, "trend": 2},
                },
            },
        },
    }
    monkeypatch.setattr(
        ssl_stages, "_prepare_balanced_kaufman_data", lambda *args, **kwargs: data)

    raw = np.empty((4, 5, 256), dtype=np.float32)
    index = np.arange(256, dtype=np.float32)
    for row, trend in enumerate((False, False, True, True)):
        close = (
            100.0 + index * (0.04 + row * 0.001)
            if trend else 100.0 + ((index + row) % 2)
        )
        raw[row] = np.stack((
            close,
            close + 1.0,
            close - 1.0,
            close + 0.1,
            1_000.0 + index * (0.1 + row * 0.01),
        ))
    states = np.asarray([0, 0, 2, 2], dtype=np.int64)
    monkeypatch.setattr(
        ssl_stages,
        "_sample_balanced_kaufman_batch",
        lambda *args, **kwargs: (raw.copy(), states.copy()),
    )

    class FakeBase:
        model_dim = 2
        chronos_config = types.SimpleNamespace(use_reg_token=True)

        def __init__(self, owner):
            self.owner = owner

        def encode(
                self, *, context, context_mask, group_ids,
                num_output_patches):
            del context_mask, group_ids, num_output_patches
            clean = torch.nan_to_num(context.float())
            slope = clean[:, -1] - clean[:, 0]
            energy = clean.square().mean(dim=1)
            register = torch.stack((
                slope * self.owner.lora_weight[0]
                + energy * self.owner.lora_weight[1] * 0.01,
                energy * self.owner.lora_weight[1]
                + slope.square() * self.owner.lora_weight[0] * 0.01,
            ), dim=1)
            return [register[:, None, :]], None, None, 0

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.asarray([0.4, 0.7]))
            self.base = FakeBase(self)

        def save_pretrained(self, destination):
            destination = Path(destination)
            destination.mkdir(parents=True)
            (destination / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "autogluon/chronos-2-small",
                "peft_type": "LORA",
                "revision": revision,
            }) + "\n")
            (destination / "adapter_model.safetensors").write_bytes(
                self.lora_weight.detach().cpu().numpy().tobytes())

    training_model = FakeModel()
    loaded_models = []

    def load_adapter(
            source, device, *, base_revision=None, base_snapshot=None):
        del device
        assert base_revision == revision
        assert Path(base_snapshot).resolve() == globals_base_snapshot.resolve()
        if Path(source).name == ssl_stages.BALANCED_KAUFMAN_STAGED_CHECKPOINT:
            loaded = FakeModel()
            saved = np.frombuffer(
                (Path(source) / "adapter_model.safetensors").read_bytes(),
                dtype=np.float32,
            ).copy()
            with torch.no_grad():
                loaded.lora_weight.copy_(torch.from_numpy(saved))
        else:
            assert Path(source).resolve() == parent.resolve()
            loaded = training_model
        loaded_models.append(loaded)
        return loaded, loaded.base

    globals_base_snapshot = base_snapshot
    monkeypatch.setattr(ssl_stages, "_load_trainable_adapter", load_adapter)
    monkeypatch.setattr(
        ssl_stages,
        "_adapter_state",
        lambda model: {
            "lora_weight": model.lora_weight.detach().cpu().clone(),
        },
    )

    def restore_adapter(model, adapter):
        with torch.no_grad():
            model.lora_weight.copy_(adapter["lora_weight"])

    monkeypatch.setattr(ssl_stages, "_restore_adapter", restore_adapter)

    parent_metrics = {"loss": 1.2, "margin": 0.1, "embedding_std": 0.2}
    epoch_metrics = {"loss": 1.0, "margin": 0.3, "embedding_std": 0.2}
    final_metrics = (
        {"loss": 0.9, "margin": 0.4, "embedding_std": 0.2}
        if final_native_gate_passes
        else {"loss": 1.3, "margin": 0.0, "embedding_std": 0.2}
    )
    metrics = [parent_metrics, epoch_metrics, final_metrics]

    def native_validation(*args, seed, **kwargs):
        del args, kwargs
        aggregate = metrics.pop(0)
        return {
            "contract": (
                "fixed_balanced_native_chronos_reg_5d_concat_without_ssl_heads"),
            "seed": seed,
            "aggregate": aggregate,
            "worst_streams": {},
            "per_stream": {"ES@3min": {**aggregate, "class_counts": {
                "0": 2, "2": 2}}},
        }

    monkeypatch.setattr(
        ssl_stages, "_balanced_kaufman_native_validation", native_validation)
    output = tmp_path / "balanced-kaufman"
    kwargs = {
        "parent": parent,
        "parent_report": parent_report,
        "base_snapshot": base_snapshot,
        "out_dir": output,
        "device": "cpu",
        "context_length": 256,
        "epochs": 1,
        "steps_per_epoch": 1,
        "batch_windows": 4,
        "gradient_accumulation": 1,
        "learning_rate": 1e-3,
        "patience": 1,
        "projection_dim": 4,
        "noise": 0.0,
        "scale": 0.0,
        "validation_windows_per_state": 2,
        "log_every_steps": 1,
        "seed": 7,
    }
    result = {
        "ssl_stages": ssl_stages,
        "output": output,
        "parent": parent,
        "parent_report": parent_report,
        "parent_before": parent_before,
        "parent_report_before": parent_report_before,
        "parent_identity": parent_identity,
        "parent_report_identity": parent_report_identity,
        "data_identity": data_identity,
        "authentication_calls": authentication_calls,
        "loaded_models": loaded_models,
        "remaining_metrics": metrics,
    }
    if final_native_gate_passes:
        result["report"] = ssl_stages.train_balanced_kaufman_ssl(
            prepared, **kwargs)
    else:
        with pytest.raises(RuntimeError, match="did not improve"):
            ssl_stages.train_balanced_kaufman_ssl(prepared, **kwargs)
    return result


def test_balanced_kaufman_trainer_publishes_fresh_head_free_child(
        tmp_path, monkeypatch):
    result = _run_mocked_balanced_kaufman_trainer(
        tmp_path, monkeypatch, final_native_gate_passes=True)
    ssl_stages = result["ssl_stages"]
    output = result["output"]
    report = result["report"]

    assert report["schema"] == ssl_stages.BALANCED_KAUFMAN_REPORT_SCHEMA
    assert report["stage"] == "balanced_kaufman_ssl"
    assert report["status"] == "complete"
    assert report["parent"] == result["parent_identity"]
    assert report["parent_report"] == result["parent_report_identity"]
    assert report["data_identity_sha256"] == result["data_identity"]
    assert report["config"]["direction_contract"] == (
        "direction_agnostic_chop_vs_trend")
    assert report["config"]["checkpoint_selection"] == {
        "contract": "gate_feasible_native_reg_balanced_kaufman_v1",
        "temporary_head_metrics_used": False,
        "metric": "macro_stream_native_loss",
    }
    native = report["checkpoint_only_validation"]
    assert native["status"] == "pass"
    assert native["contract"] == (
        "freshly_reloaded_lora_native_reg_without_temporary_heads")
    assert set(native["loss_lift_parent_minus_checkpoint"]) == {
        "loss", "margin"}
    assert report["checkpoint"]["sha256"] == ssl_stages.tree_sha256(
        output / "checkpoint")
    assert len(result["loaded_models"]) == 2
    assert result["loaded_models"][0] is not result["loaded_models"][1]
    assert len(result["authentication_calls"]) == 2
    assert not result["remaining_metrics"]
    assert not (output / "trainer.pt").exists()
    assert not (output / ssl_stages.BALANCED_KAUFMAN_STAGED_CHECKPOINT).exists()
    assert not (output / ssl_stages.BALANCED_KAUFMAN_STAGED_REPORT).exists()
    checkpoint_files = [
        path.name.lower()
        for path in (output / "checkpoint").rglob("*") if path.is_file()
    ]
    assert checkpoint_files
    assert not any(
        token in name
        for name in checkpoint_files
        for token in ("head", "decoder", "projection", "trainer")
    )
    artifact = report["final_artifact_contract"]
    assert artifact["temporary_heads_in_checkpoint"] is False
    assert artifact["ssl_heads_required_for_inference"] is False
    assert artifact["trainer_state"] == "discarded_after_successful_checkpoint"
    assert artifact["inference_requires"] == [
        "chronos_base_model", "lora_checkpoint"]
    assert {
        path.relative_to(result["parent"]): path.read_bytes()
        for path in result["parent"].rglob("*") if path.is_file()
    } == result["parent_before"]
    assert result["parent_report"].read_bytes() == result["parent_report_before"]


def test_balanced_kaufman_failed_fresh_native_gate_keeps_resume_state(
        tmp_path, monkeypatch):
    result = _run_mocked_balanced_kaufman_trainer(
        tmp_path, monkeypatch, final_native_gate_passes=False)
    ssl_stages = result["ssl_stages"]
    output = result["output"]

    assert len(result["loaded_models"]) == 2
    assert result["loaded_models"][0] is not result["loaded_models"][1]
    assert len(result["authentication_calls"]) == 2
    assert not result["remaining_metrics"]
    assert (output / "trainer.pt").is_file()
    assert not (output / "checkpoint").exists()
    assert (output / ssl_stages.BALANCED_KAUFMAN_STAGED_CHECKPOINT).is_dir()
    assert not (output / "report.json").exists()
    assert not (output / ssl_stages.BALANCED_KAUFMAN_STAGED_REPORT).exists()
    assert {
        path.relative_to(result["parent"]): path.read_bytes()
        for path in result["parent"].rglob("*") if path.is_file()
    } == result["parent_before"]
    assert result["parent_report"].read_bytes() == result["parent_report_before"]


def test_chronos2_contrastive_entrypoint_defaults_to_kaufman():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "chronos" / "chronos2_ssl_contrastive.py"
    ).read_text()
    assert '"--regime-key", choices=("kaufman", "temporal"), default="kaufman"' in source
    assert '"--kaufman-chop", type=float, default=0.25' in source
    assert '"--kaufman-trend", type=float, default=0.50' in source
    assert '"--kaufman-length", type=int, default=64' in source


def test_chronos2_range_dynamics_states_are_self_supervised_and_scale_free():
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _range_dynamics_regime,
    )

    raw = torch.zeros((3, 5, 8), dtype=torch.float32)
    # First/second half ranges: 4/1, 2/2, and 1/4.
    raw[0, 1] = torch.tensor([4, 3, 2, 1, 1, .75, .5, .25])
    raw[1, 1] = torch.tensor([2, 1.5, 1, .5, 2, 1.5, 1, .5])
    raw[2, 1] = torch.tensor([1, .75, .5, .25, 4, 3, 2, 1])
    regime, score = _range_dynamics_regime(
        raw,
        lower=np.log(0.5),
        upper=np.log(2.0),
        dynamics_length=8,
    )
    scaled_regime, scaled_score = _range_dynamics_regime(
        raw * 100,
        lower=np.log(0.5),
        upper=np.log(2.0),
        dynamics_length=8,
    )

    assert regime.tolist() == [0, 1, 2]
    assert score.tolist() == pytest.approx(
        [np.log(0.25), 0.0, np.log(4.0)])
    assert scaled_regime.tolist() == regime.tolist()
    assert scaled_score.tolist() == pytest.approx(score.tolist())


def test_chronos2_range_dynamics_never_reads_after_completed_context():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _range_dynamics_scores,
    )

    matrix = np.zeros((5, 20), dtype=np.float32)
    matrix[1] = np.arange(20, dtype=np.float32) + 1
    matrix[2] = matrix[1] - 0.5
    original = _range_dynamics_scores(
        matrix, np.asarray([0]), context_length=8, dynamics_length=8)
    changed = matrix.copy()
    changed[1:, 8:] = 1_000_000
    after_future_change = _range_dynamics_scores(
        changed, np.asarray([0]), context_length=8, dynamics_length=8)

    np.testing.assert_array_equal(original, after_future_change)


def test_chronos2_volatility_thresholds_are_fit_from_supplied_training_starts():
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import (
        _fit_range_dynamics_thresholds,
    )

    matrix = np.zeros((5, 80), dtype=np.float32)
    matrix[1] = 100 + np.sin(np.arange(80) / 3) * np.linspace(1, 8, 80)
    matrix[2] = matrix[1] - np.linspace(1, 4, 80)
    starts = np.arange(0, 49, dtype=np.int64)
    fitted = _fit_range_dynamics_thresholds(
        matrix,
        starts,
        context_length=32,
        dynamics_length=16,
        lower_quantile=0.25,
        upper_quantile=0.75,
        max_samples=17,
    )

    assert fitted["fit_samples"] == 17
    assert fitted["compression_max"] < fitted["expansion_min"]
    assert np.isfinite(fitted["fit_score_mean"])
    assert np.isfinite(fitted["fit_score_std"])


def test_chronos2_volatility_entrypoint_is_optional_kaufman_retaining_ssl():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "chronos"
        / "chronos2_ssl_volatility_contrastive.py"
    ).read_text()

    assert "train_volatility_contrastive" in source
    assert "contrastive_kaufman_full/checkpoint" in source
    assert '"--dynamics-length", type=int, default=64' in source
    assert '"--lower-quantile", type=float, default=0.25' in source
    assert '"--upper-quantile", type=float, default=0.75' in source
    assert '"--kaufman-retention-weight", type=float, default=1.0' in source


def test_chronos2_streaming_embedder_loads_pipeline_once(monkeypatch):
    torch = pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2._embed_worker import (
        embed_window_chunks,
    )

    calls = {"load": 0, "embed": 0}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls["load"] += 1
            assert source == "fake/model"
            return cls()

        def embed(self, windows, *, batch_size, context_length):
            calls["embed"] += 1
            assert windows.shape[1:] == (5, 8)
            assert batch_size == 5
            assert context_length == 8
            return [
                torch.full((5, 4, 3), float(index + 1))
                for index in range(len(windows))
            ], None

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(Chronos2Pipeline=FakePipeline),
    )
    chunks = (
        np.ones((2, 5, 8), np.float32),
        np.ones((1, 5, 8), np.float32),
    )
    output = list(embed_window_chunks(
        chunks,
        checkpoint="fake/model",
        device="cpu",
        batch=5,
        context_length=8,
    ))

    assert calls == {"load": 1, "embed": 3}
    assert [item.shape for item in output] == [(2, 15), (1, 15)]
