"""Contracts for the standard MV3 market-context decoder."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from futures_foundation.market_context import (
    MARKET_CONTEXT_FIELDS,
    MARKET_CONTEXT_HORIZON,
    MARKET_CONTEXT_MODEL,
    MARKET_CONTEXT_SCHEMA,
    MarketContextDecoder,
    extract_market_context,
    load_market_context_decoder,
)
from futures_foundation.market_context_training import (
    fit_market_context_decoder,
)
from futures_foundation.finetune.pretext.momentum_volatility import (
    MOMENTUM_VOLATILITY_SCHEMA,
)


def _metadata(encoder_sha: str, embedding_dim: int = 6) -> dict:
    return {
        "schema": MARKET_CONTEXT_SCHEMA,
        "fields": list(MARKET_CONTEXT_FIELDS),
        "encoder_sha256": encoder_sha,
        "target_schema": MOMENTUM_VOLATILITY_SCHEMA,
        "horizon": MARKET_CONTEXT_HORIZON,
        "model": MARKET_CONTEXT_MODEL,
        "embedding_dim": embedding_dim,
        "fit_end": "2023-01-01",
        "calibration_start": "2023-01-01",
        "calibration_end": "2024-01-01",
        "evaluation_start": "2025-01-01",
        "evaluation_end": "2026-01-01",
        "scope": "9x4_strategy_agnostic",
    }


def _decoder(encoder_sha: str, embedding_dim: int = 6):
    coefficients = np.zeros((4, embedding_dim), np.float32)
    coefficients[:, :4] = np.eye(4, dtype=np.float32)
    return MarketContextDecoder(
        mean=np.zeros(embedding_dim, np.float32),
        scale=np.ones(embedding_dim, np.float32),
        coefficients=coefficients,
        intercept=np.zeros(4, np.float32),
        temperature=1.0,
        metadata=_metadata(encoder_sha, embedding_dim),
    )


def test_decoder_returns_named_probability_simplex():
    decoder = _decoder("a" * 64)
    embeddings = np.array([
        [4, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ], np.float32)

    result = decoder.transform(embeddings, include_embeddings=True)

    assert result.fields == (
        "p_trend_expansion",
        "p_trend_weakening",
        "p_noisy_expansion",
        "p_compression",
    )
    np.testing.assert_allclose(
        result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.probabilities.argmax(axis=1).tolist() == [0, 1, 2, 3]
    assert set(result.as_dict()) == set(MARKET_CONTEXT_FIELDS)
    np.testing.assert_array_equal(result.embeddings, embeddings)


def test_decoder_artifact_round_trips_without_pickle_and_binds_encoder(tmp_path):
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"exact encoder")
    encoder_sha = hashlib.sha256(encoder.read_bytes()).hexdigest()
    artifact = _decoder(encoder_sha).save(tmp_path / "context.npz")

    loaded = load_market_context_decoder(
        artifact, encoder_checkpoint=encoder)

    np.testing.assert_array_equal(
        loaded.coefficients, _decoder(encoder_sha).coefficients)
    assert loaded.metadata["encoder_sha256"] == encoder_sha

    other = tmp_path / "other.pt"
    other.write_bytes(b"different encoder")
    with pytest.raises(ValueError, match="does not match encoder"):
        load_market_context_decoder(
            artifact, encoder_checkpoint=other)


def test_decoder_fails_closed_on_schema_or_shape_drift():
    metadata = _metadata("b" * 64)
    metadata["fields"] = list(reversed(MARKET_CONTEXT_FIELDS))
    with pytest.raises(ValueError, match="fields drifted"):
        MarketContextDecoder(
            mean=np.zeros(6),
            scale=np.ones(6),
            coefficients=np.zeros((4, 6)),
            intercept=np.zeros(4),
            temperature=1.0,
            metadata=metadata,
        )
    with pytest.raises(ValueError, match=r"shape \[N,6\]"):
        _decoder("b" * 64).predict_proba(np.zeros((3, 5)))


def test_extract_market_context_runs_encoder_then_standard_decoder(
    tmp_path,
    monkeypatch,
):
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"encoder")
    encoder_sha = hashlib.sha256(encoder.read_bytes()).hexdigest()
    artifact = _decoder(encoder_sha).save(tmp_path / "context.npz")
    captured = {}

    def fake_embed_windows(windows, **kwargs):
        captured["windows"] = np.asarray(windows).copy()
        captured["kwargs"] = kwargs
        return np.tile(
            np.arange(6, dtype=np.float32),
            (len(windows), 1),
        )

    from futures_foundation.finetune.pretext._torch import common
    monkeypatch.setattr(common, "embed_windows", fake_embed_windows)
    windows = np.ones((3, 5, 128), np.float32)

    result = extract_market_context(
        windows,
        encoder_checkpoint=encoder,
        decoder_artifact=artifact,
        device="cpu",
        batch=16,
    )

    assert result.probabilities.shape == (3, 4)
    np.testing.assert_array_equal(captured["windows"], windows)
    assert captured["kwargs"]["ckpt"] == str(encoder)
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["batch"] == 16


def test_market_context_module_import_is_torch_free():
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "futures_foundation"
        / "market_context.py"
    ).read_text()
    assert "import torch" not in source


def test_context_decoder_fit_is_temporal_uniform_stream_and_beats_shuffle():
    generator = np.random.default_rng(12)
    parts = []
    states = []
    timestamps = []
    streams = []
    for date in ("2022-06-01", "2023-06-01", "2025-06-01"):
        for state in range(4):
            values = generator.normal(0, 0.2, (80, 8)).astype(np.float32)
            values[:, state] += 3.0
            parts.append(values)
            states.extend([state] * len(values))
            timestamps.extend([np.datetime64(date)] * len(values))
            streams.extend(
                "NQ@3min" if row % 2 == 0 else "GC@5min"
                for row in range(len(values))
            )
    embeddings = np.concatenate(parts)

    decoder, report = fit_market_context_decoder(
        embeddings,
        np.asarray(states),
        np.asarray(timestamps),
        np.asarray(streams),
        encoder_sha256="c" * 64,
        min_auc=0.80,
        min_shuffle_lift=0.20,
        max_iter=200,
    )

    assert report["pass"]
    assert report["splits"] == {
        "fit": 320,
        "calibration": 320,
        "evaluation": 320,
    }
    assert decoder.metadata["sampling"] == "uniform_stream_weighted"
    assert all(
        row["auc"] > 0.99
        and row["real_minus_shuffle_auc"] > 0.20
        for row in report["evaluation"]["classes"].values()
    )
