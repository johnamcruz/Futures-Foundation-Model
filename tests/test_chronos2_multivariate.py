"""Torch-free contracts for the Chronos-2 multivariate MVP and classifier seam."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
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
