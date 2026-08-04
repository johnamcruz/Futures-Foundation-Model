"""Frozen Chronos-2 embeddings for causally aligned 3m/developing-15m bars.

The public boundary is deliberately explicit: every input window is one
Chronos multivariate task with ten ordered variates::

    [3m OHLCV, developing-15m OHLCV]

The two five-variate REG blocks remain named and separate.  Downstream callers
may combine them, but this module never anonymously pools or flattens the
higher-timeframe state into the primary state.

Heavy Chronos/torch imports live in ``_paired_embed_worker`` so importing the
public contract does not initialize the model runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .frozen import MODEL_ID, _checkpoint_fingerprint
from .paired_timeframes import CHANNEL_NAMES


SCHEMA = "ffm_chronos2_paired_grouped_reg_v1"
PRIMARY_VARIATES = 5
RELATED_VARIATES = 5
TOTAL_VARIATES = PRIMARY_VARIATES + RELATED_VARIATES
POOLING = "native_reg_token_minus_2"


@dataclass(frozen=True)
class Chronos2PairedEmbeddings:
    """Named native REG representations for one batch of paired windows.

    Both arrays have shape ``[B, 5, D]``.  Axis one remains in OHLCV order.
    Keeping the blocks separate is part of the consumer contract and preserves
    attribution between the local and developing higher-timeframe context.
    """

    three_minute: np.ndarray
    developing_fifteen_minute: np.ndarray

    def __post_init__(self) -> None:
        primary = np.asarray(self.three_minute, dtype=np.float32)
        related = np.asarray(self.developing_fifteen_minute, dtype=np.float32)
        if primary.ndim != 3 or primary.shape[1] != PRIMARY_VARIATES:
            raise ValueError(
                "three_minute embeddings must have shape [B,5,D], "
                f"got {primary.shape}"
            )
        if related.ndim != 3 or related.shape[1] != RELATED_VARIATES:
            raise ValueError(
                "developing_fifteen_minute embeddings must have shape [B,5,D], "
                f"got {related.shape}"
            )
        if primary.shape != related.shape:
            raise ValueError(
                "paired embedding halves must have identical [B,5,D] shapes"
            )
        if primary.shape[0] < 1 or primary.shape[2] < 1:
            raise ValueError("paired embeddings cannot have an empty batch or dimension")
        if not np.isfinite(primary).all() or not np.isfinite(related).all():
            raise ValueError("paired embeddings must be finite")
        object.__setattr__(self, "three_minute", np.ascontiguousarray(primary))
        object.__setattr__(
            self,
            "developing_fifteen_minute",
            np.ascontiguousarray(related),
        )

    @property
    def all_variates(self) -> np.ndarray:
        """Return the ordered ``[B,10,D]`` representation without pooling."""
        return np.concatenate(
            (self.three_minute, self.developing_fifteen_minute), axis=1
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.three_minute.shape[-1])


def validate_paired_windows(windows: np.ndarray) -> np.ndarray:
    """Validate and normalize the immutable ``[B,10,T]`` input boundary."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != TOTAL_VARIATES:
        raise ValueError(
            "paired Chronos-2 windows must have shape [B,10,T] ordered as "
            "[3m OHLCV, developing-15m OHLCV]; "
            f"got {values.shape}"
        )
    if values.shape[0] < 1 or values.shape[2] < 1:
        raise ValueError("paired Chronos-2 windows cannot have an empty batch/history")
    if not np.isfinite(values).all():
        raise ValueError("paired Chronos-2 windows contain non-finite OHLCV")
    return np.ascontiguousarray(values)


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def embed_paired_window_chunks(
    chunks: Iterable[np.ndarray],
    *,
    checkpoint: str | Path | None = None,
    model_id: str = MODEL_ID,
    device: str = "cpu",
    batch_windows: int = 1,
    context_length: int | None = None,
) -> Iterator[Chronos2PairedEmbeddings]:
    """Lazily yield frozen grouped embeddings for ``[B,10,T]`` chunks.

    Importing this public module remains free of Chronos and torch imports.  The
    isolated worker is imported only when the returned iterator is consumed,
    and it loads the model once for the complete chunk stream.
    """
    if isinstance(chunks, np.ndarray):
        raise TypeError(
            "chunks must be an iterable of [B,10,T] arrays; wrap a single "
            "array in a tuple"
        )
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of [B,10,T] arrays")
    try:
        stream = iter(chunks)
    except TypeError as error:
        raise TypeError("chunks must be an iterable of [B,10,T] arrays") from error

    resolved_batch = _positive_integer(batch_windows, name="batch_windows")
    resolved_context = (
        None
        if context_length is None
        else _positive_integer(context_length, name="context_length")
    )
    if checkpoint is not None and not isinstance(checkpoint, (str, Path)):
        raise TypeError("checkpoint must be a string, Path, or None")
    if isinstance(checkpoint, str) and not checkpoint.strip():
        raise ValueError("checkpoint cannot be empty")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")

    def _iterator() -> Iterator[Chronos2PairedEmbeddings]:
        from ._paired_embed_worker import (
            embed_paired_window_chunks as _worker_embed_paired_window_chunks,
        )

        outputs = _worker_embed_paired_window_chunks(
            stream,
            checkpoint=checkpoint,
            model_id=model_id,
            device=device,
            batch_windows=resolved_batch,
            context_length=resolved_context,
        )
        for output in outputs:
            if not isinstance(output, Chronos2PairedEmbeddings):
                raise RuntimeError(
                    "paired Chronos-2 worker returned an unnamed or invalid "
                    "embedding output"
                )
            yield output

    return _iterator()


def paired_encoder_identity(
    *,
    checkpoint: str | Path | None,
    model_id: str = MODEL_ID,
    context_length: int,
) -> dict:
    """Return the frozen encoder/schema identity required by cache manifests.

    Row keys, source hashes, and split identity remain the cache builder's
    responsibility.  This identity prevents a consumer from silently changing
    checkpoint bytes, channel order, context length, pooling, or output names.
    """
    if int(context_length) < 1:
        raise ValueError("context_length must be positive")
    checkpoint_value = None if checkpoint is None else str(checkpoint)
    payload = {
        "schema": SCHEMA,
        "model": _checkpoint_fingerprint(checkpoint_value, str(model_id)),
        "model_id": str(model_id),
        "input_shape": "[B,10,T]",
        "input_channels": list(CHANNEL_NAMES),
        "context_length": int(context_length),
        "grouping": "one_distinct_group_per_window_all_10_variates",
        "pooling": POOLING,
        "outputs": {
            "three_minute": "[B,5,D] OHLCV order",
            "developing_fifteen_minute": "[B,5,D] OHLCV order",
        },
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


__all__ = [
    "Chronos2PairedEmbeddings",
    "POOLING",
    "SCHEMA",
    "TOTAL_VARIATES",
    "embed_paired_window_chunks",
    "paired_encoder_identity",
    "validate_paired_windows",
]
