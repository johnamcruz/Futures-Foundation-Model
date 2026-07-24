"""Versioned, strategy-agnostic market-context extraction from FFM embeddings.

The foundation encoder intentionally remains an encoder-only checkpoint.  This
module owns the small, auditable decoder that maps its frozen embedding to the
four momentum/volatility states learned by MV-v3.  The decoder never consumes
entries, pivots, stops, R multiples, or future bars at inference.

The artifact is bound to the exact encoder SHA-256 and target schema.  This
prevents a downstream consumer from silently combining decoder weights with a
different representation space.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .finetune.pretext.momentum_volatility import (
    COUPLING_CLASSES,
    MOMENTUM_VOLATILITY_SCHEMA,
)


MARKET_CONTEXT_SCHEMA = "ffm_market_context_v1"
MARKET_CONTEXT_FIELDS = tuple(f"p_{name}" for name in COUPLING_CLASSES)
MARKET_CONTEXT_HORIZON = 20
MARKET_CONTEXT_MODEL = "multinomial_linear_softmax"
_REQUIRED_METADATA = {
    "schema",
    "fields",
    "encoder_sha256",
    "target_schema",
    "horizon",
    "model",
    "embedding_dim",
    "fit_end",
    "calibration_start",
    "calibration_end",
    "evaluation_start",
    "evaluation_end",
    "scope",
}


def sha256(path: str | Path) -> str:
    """Return a stable identity for a checkpoint or decoder artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MarketContextBatch:
    """Named market-context probabilities aligned with foundation embeddings."""

    probabilities: np.ndarray
    fields: tuple[str, ...] = MARKET_CONTEXT_FIELDS
    embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.fields):
            raise ValueError(
                "market-context probabilities must have shape [N,4]")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError(
                "market-context probabilities must be a finite simplex")
        if tuple(self.fields) != MARKET_CONTEXT_FIELDS:
            raise ValueError("market-context field order does not match schema")
        if self.embeddings is not None:
            embeddings = np.asarray(self.embeddings)
            if embeddings.ndim != 2 or len(embeddings) != len(probabilities):
                raise ValueError(
                    "market-context embeddings must align with probabilities")

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return one stable named column per context field."""
        return {
            name: self.probabilities[:, index]
            for index, name in enumerate(self.fields)
        }


@dataclass(frozen=True)
class MarketContextDecoder:
    """Frozen linear-softmax decoder for one exact foundation representation."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    temperature: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, np.float32)
        scale = np.asarray(self.scale, np.float32)
        coefficients = np.asarray(self.coefficients, np.float32)
        intercept = np.asarray(self.intercept, np.float32)
        metadata = dict(self.metadata)
        missing = _REQUIRED_METADATA - set(metadata)
        if missing:
            raise ValueError(
                f"market-context metadata missing fields: {sorted(missing)}")
        embedding_dim = int(metadata["embedding_dim"])
        expected_classes = len(MARKET_CONTEXT_FIELDS)
        if (
            mean.shape != (embedding_dim,)
            or scale.shape != (embedding_dim,)
            or coefficients.shape != (expected_classes, embedding_dim)
            or intercept.shape != (expected_classes,)
        ):
            raise ValueError("market-context decoder arrays do not align")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept).all()
            or (scale <= 0).any()
            or not np.isfinite(self.temperature)
            or float(self.temperature) <= 0
        ):
            raise ValueError("market-context decoder contains invalid values")
        if metadata["schema"] != MARKET_CONTEXT_SCHEMA:
            raise ValueError("unsupported market-context artifact schema")
        if tuple(metadata["fields"]) != MARKET_CONTEXT_FIELDS:
            raise ValueError("market-context artifact fields drifted")
        if metadata["target_schema"] != MOMENTUM_VOLATILITY_SCHEMA:
            raise ValueError("market-context target schema drifted")
        if int(metadata["horizon"]) != MARKET_CONTEXT_HORIZON:
            raise ValueError("market-context horizon drifted")
        if metadata["model"] != MARKET_CONTEXT_MODEL:
            raise ValueError("unsupported market-context decoder model")
        encoder_sha = str(metadata["encoder_sha256"])
        if len(encoder_sha) != 64 or any(
            value not in "0123456789abcdef" for value in encoder_sha
        ):
            raise ValueError("market-context encoder identity is invalid")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "metadata", metadata)

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata["embedding_dim"])

    def predict_proba(self, embeddings) -> np.ndarray:
        """Decode causal frozen embeddings into the four named MV states."""
        values = np.asarray(embeddings, np.float32)
        if values.ndim != 2 or values.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings must have shape [N,{self.embedding_dim}]")
        if not np.isfinite(values).all():
            raise ValueError("market-context embeddings contain non-finite values")
        normalized = (values - self.mean) / self.scale
        logits = (
            normalized @ self.coefficients.T + self.intercept
        ) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(np.clip(logits, -80.0, 0.0))
        return (exponent / exponent.sum(axis=1, keepdims=True)).astype(
            np.float32)

    def transform(
        self,
        embeddings,
        *,
        include_embeddings: bool = False,
    ) -> MarketContextBatch:
        values = np.asarray(embeddings, np.float32)
        return MarketContextBatch(
            probabilities=self.predict_proba(values),
            embeddings=values if include_embeddings else None,
        )

    def save(self, path: str | Path) -> Path:
        """Atomically save the decoder without pickle or executable objects."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            mean=self.mean,
            scale=self.scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
            temperature=np.asarray(self.temperature, np.float32),
            metadata_json=np.asarray(
                json.dumps(dict(self.metadata), sort_keys=True)),
        )
        temporary.replace(destination)
        return destination


def load_market_context_decoder(
    path: str | Path,
    *,
    encoder_checkpoint: str | Path | None = None,
) -> MarketContextDecoder:
    """Load a safe decoder and optionally verify its exact encoder checkpoint."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "mean",
            "scale",
            "coefficients",
            "intercept",
            "temperature",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                f"market-context artifact missing arrays: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        decoder = MarketContextDecoder(
            mean=payload["mean"],
            scale=payload["scale"],
            coefficients=payload["coefficients"],
            intercept=payload["intercept"],
            temperature=float(payload["temperature"]),
            metadata=metadata,
        )
    if encoder_checkpoint is not None:
        actual = sha256(encoder_checkpoint)
        expected = str(decoder.metadata["encoder_sha256"])
        if actual != expected:
            raise ValueError(
                "market-context decoder does not match encoder checkpoint: "
                f"expected {expected}, got {actual}")
    return decoder


def extract_market_context(
    windows,
    *,
    encoder_checkpoint: str | Path,
    decoder_artifact: str | Path,
    device: str | None = None,
    batch: int = 512,
    include_embeddings: bool = False,
) -> MarketContextBatch:
    """Encode raw causal OHLCV windows and return standardized named context.

    ``windows`` must be ``[N,5,sequence]`` and end on the decision candle.
    Encoder loading is lazy, preserving FFM's torch-free import contract.
    """
    values = np.asarray(windows, np.float32)
    if values.ndim != 3 or values.shape[1] != 5:
        raise ValueError("market-context windows must have shape [N,5,sequence]")
    if values.shape[2] < 2 or not np.isfinite(values).all():
        raise ValueError("market-context windows are invalid")
    decoder = load_market_context_decoder(
        decoder_artifact,
        encoder_checkpoint=encoder_checkpoint,
    )
    from .finetune.pretext._torch.common import embed_windows

    embeddings = embed_windows(
        values,
        ckpt=str(encoder_checkpoint),
        device=device,
        batch=int(batch),
    )
    return decoder.transform(
        embeddings,
        include_embeddings=include_embeddings,
    )


__all__ = [
    "MARKET_CONTEXT_FIELDS",
    "MARKET_CONTEXT_HORIZON",
    "MARKET_CONTEXT_MODEL",
    "MARKET_CONTEXT_SCHEMA",
    "MarketContextBatch",
    "MarketContextDecoder",
    "extract_market_context",
    "load_market_context_decoder",
    "sha256",
]
