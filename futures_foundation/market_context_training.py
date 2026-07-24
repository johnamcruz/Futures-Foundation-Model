"""Leakage-safe fitting and evaluation for the standard FFM context decoder."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from .market_context import (
    MARKET_CONTEXT_FIELDS,
    MARKET_CONTEXT_HORIZON,
    MARKET_CONTEXT_MODEL,
    MARKET_CONTEXT_SCHEMA,
    MarketContextDecoder,
)
from .finetune.pretext.momentum_volatility import (
    COUPLING_CLASSES,
    MOMENTUM_VOLATILITY_SCHEMA,
)


def _timestamp_mask(
    timestamps: np.ndarray,
    start: str | None,
    end: str,
) -> np.ndarray:
    values = np.asarray(timestamps).astype("datetime64[ns]")
    mask = values < np.datetime64(end)
    if start is not None:
        mask &= values >= np.datetime64(start)
    return mask


def _require_classes(states: np.ndarray, rows: np.ndarray, split: str) -> None:
    observed = set(np.asarray(states[rows], np.int64).tolist())
    expected = set(range(len(COUPLING_CLASSES)))
    if observed != expected:
        raise ValueError(
            f"market-context {split} split lacks classes: "
            f"{sorted(expected - observed)}")


def _stream_weights(streams: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Give every ticker/timeframe equal total mass in the fit."""
    selected = np.asarray(streams, str)[rows]
    weight = np.zeros(len(rows), np.float64)
    unique, counts = np.unique(selected, return_counts=True)
    for stream, count in zip(unique, counts):
        weight[selected == stream] = len(rows) / (len(unique) * int(count))
    return weight


def _temperature(
    logits: np.ndarray,
    states: np.ndarray,
    *,
    candidates: Iterable[float] = tuple(np.linspace(0.5, 2.5, 81)),
) -> tuple[float, float]:
    """Fit one shared temperature on the chronological calibration block."""
    logits = np.asarray(logits, np.float64)
    states = np.asarray(states, np.int64)
    if logits.shape != (len(states), len(COUPLING_CLASSES)):
        raise ValueError("market-context calibration logits are misaligned")
    rows = np.arange(len(states))
    best = (float("inf"), None)
    for value in candidates:
        temperature = float(value)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("market-context temperature grid is invalid")
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probability = (
            scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        )
        loss = float(-log_probability[rows, states].mean())
        if loss < best[0]:
            best = (loss, temperature)
    if best[1] is None:
        raise RuntimeError("market-context calibration found no temperature")
    return float(best[1]), float(best[0])


def _class_metrics(
    decoder: MarketContextDecoder,
    embeddings: np.ndarray,
    states: np.ndarray,
    streams: np.ndarray,
    rows: np.ndarray,
    *,
    shuffle_seed: int,
) -> dict:
    from sklearn.metrics import roc_auc_score

    values = np.asarray(embeddings[rows], np.float32)
    truth = np.asarray(states[rows], np.int64)
    selected_streams = np.asarray(streams, str)[rows]
    probability = decoder.predict_proba(values)
    shuffled = decoder.predict_proba(
        values[np.random.default_rng(shuffle_seed).permutation(len(values))])
    metrics = {}
    for index, name in enumerate(COUPLING_CLASSES):
        binary = truth == index
        auc = float(roc_auc_score(binary, probability[:, index]))
        shuffle_auc = float(
            roc_auc_score(binary, shuffled[:, index]))
        per_stream = {}
        for stream in sorted(set(selected_streams)):
            use = selected_streams == stream
            if use.sum() >= 100 and binary[use].std() > 0:
                per_stream[stream] = float(
                    roc_auc_score(binary[use], probability[use, index]))
        metrics[name] = {
            "auc": auc,
            "shuffle_auc": shuffle_auc,
            "real_minus_shuffle_auc": auc - shuffle_auc,
            "positive_rate": float(binary.mean()),
            "n": int(len(binary)),
            "per_stream_auc": per_stream,
            "worst_stream_auc": (
                min(per_stream.values()) if per_stream else None),
        }
    rows_index = np.arange(len(truth))
    clipped = np.clip(probability[rows_index, truth], 1e-8, 1.0)
    one_hot = np.eye(len(COUPLING_CLASSES), dtype=np.float32)[truth]
    return {
        "classes": metrics,
        "accuracy": float((probability.argmax(axis=1) == truth).mean()),
        "nll": float(-np.log(clipped).mean()),
        "multiclass_brier": float(
            np.square(probability - one_hot).sum(axis=1).mean()),
        "n": int(len(truth)),
    }


def fit_market_context_decoder(
    embeddings,
    states,
    timestamps,
    streams,
    *,
    encoder_sha256: str,
    fit_end: str = "2023-01-01",
    calibration_start: str = "2023-01-01",
    calibration_end: str = "2024-01-01",
    evaluation_start: str = "2025-01-01",
    evaluation_end: str = "2026-01-01",
    scope: str = "9x4_strategy_agnostic",
    seed: int = 777,
    c: float = 0.1,
    max_iter: int = 1000,
    min_auc: float = 0.55,
    min_shuffle_lift: float = 0.02,
) -> tuple[MarketContextDecoder, dict]:
    """Fit on the past, calibrate later, and evaluate on untouched 2025 rows."""
    from sklearn.linear_model import LogisticRegression

    embeddings = np.asarray(embeddings, np.float32)
    states = np.asarray(states, np.int64)
    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    streams = np.asarray(streams, str)
    if (
        embeddings.ndim != 2
        or states.shape != (len(embeddings),)
        or timestamps.shape != (len(embeddings),)
        or streams.shape != (len(embeddings),)
        or not np.isfinite(embeddings).all()
        or ((states < 0) | (states >= len(COUPLING_CLASSES))).any()
    ):
        raise ValueError("market-context training arrays are invalid")
    if (
        not fit_end <= calibration_start < calibration_end
        or not calibration_end <= evaluation_start < evaluation_end
    ):
        raise ValueError("market-context temporal boundaries are invalid")
    fit = np.flatnonzero(_timestamp_mask(timestamps, None, fit_end))
    calibration = np.flatnonzero(_timestamp_mask(
        timestamps, calibration_start, calibration_end))
    evaluation = np.flatnonzero(_timestamp_mask(
        timestamps, evaluation_start, evaluation_end))
    for name, rows in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        if not len(rows):
            raise ValueError(f"market-context {name} split is empty")
        _require_classes(states, rows, name)

    mean = embeddings[fit].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = embeddings[fit].std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    train = (embeddings[fit] - mean) / scale
    classifier = LogisticRegression(
        C=float(c),
        max_iter=int(max_iter),
        random_state=int(seed),
        class_weight="balanced",
    )
    classifier.fit(
        train,
        states[fit],
        sample_weight=_stream_weights(streams, fit),
    )
    if not np.array_equal(
        classifier.classes_, np.arange(len(COUPLING_CLASSES))
    ):
        raise RuntimeError("market-context class ordering drifted")
    calibration_values = (embeddings[calibration] - mean) / scale
    logits = classifier.decision_function(calibration_values)
    temperature, calibration_nll = _temperature(
        logits, states[calibration])
    metadata = {
        "schema": MARKET_CONTEXT_SCHEMA,
        "fields": list(MARKET_CONTEXT_FIELDS),
        "encoder_sha256": str(encoder_sha256),
        "target_schema": MOMENTUM_VOLATILITY_SCHEMA,
        "horizon": MARKET_CONTEXT_HORIZON,
        "model": MARKET_CONTEXT_MODEL,
        "embedding_dim": int(embeddings.shape[1]),
        "fit_end": fit_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "scope": scope,
        "sampling": "uniform_stream_weighted",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    decoder = MarketContextDecoder(
        mean=mean,
        scale=scale,
        coefficients=np.asarray(classifier.coef_, np.float32),
        intercept=np.asarray(classifier.intercept_, np.float32),
        temperature=temperature,
        metadata=metadata,
    )
    evaluation_metrics = _class_metrics(
        decoder,
        embeddings,
        states,
        streams,
        evaluation,
        shuffle_seed=seed + 10_000,
    )
    gates = {}
    for name, metric in evaluation_metrics["classes"].items():
        gates[f"{name}_auc_at_least_{min_auc:g}"] = bool(
            metric["auc"] >= min_auc)
        gates[f"{name}_beats_shuffle_by_{min_shuffle_lift:g}"] = bool(
            metric["real_minus_shuffle_auc"] >= min_shuffle_lift)
    report = {
        "schema": "ffm_market_context_fit_report_v1",
        "metadata": dict(decoder.metadata),
        "splits": {
            "fit": int(len(fit)),
            "calibration": int(len(calibration)),
            "evaluation": int(len(evaluation)),
        },
        "temperature": temperature,
        "calibration_nll": calibration_nll,
        "evaluation": evaluation_metrics,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    return decoder, report


__all__ = ["fit_market_context_decoder"]
