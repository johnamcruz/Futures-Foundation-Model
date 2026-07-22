"""Deterministic epoch sampling for pooled multi-stream downstream training.

Foundation representations are commonly trained on streams with very different row counts.
``uniform_stream`` prevents a dense stream (for example, one-minute bars) from owning the loss:
an epoch keeps its original length, chooses every stream equally often, and samples rows only
from the supplied train subset. Validation and test rows are never accepted separately or used
to determine the allocation.
"""
from __future__ import annotations

import numpy as np


SAMPLING_MODES = ("bar_proportional", "uniform_stream")


def resolve_epoch_sampling_mode(mode, epoch, total_epochs, *, switch_fraction=0.5):
    """Resolve a fixed or two-stage sampling curriculum for one epoch.

    ``bar_then_uniform`` first exposes the model to the naturally abundant corpus, then gives
    every stream equal influence during refinement. The schedule depends only on the declared
    epoch budget—never validation or OOS outcomes.
    """
    if mode in SAMPLING_MODES:
        return mode
    if mode != "bar_then_uniform":
        raise ValueError(f"unsupported sampling curriculum {mode!r}")
    total_epochs = int(total_epochs)
    epoch = int(epoch)
    if total_epochs <= 1 or not 0.0 < float(switch_fraction) < 1.0:
        raise ValueError("bar_then_uniform requires total_epochs > 1 and 0 < switch_fraction < 1")
    switch = max(1, min(total_epochs - 1, round(total_epochs * float(switch_fraction))))
    return "bar_proportional" if epoch < switch else "uniform_stream"


def sample_epoch_rows(rows, stream_ids, *, mode="bar_proportional", seed=0, epoch=0):
    """Return deterministic row indices for one training epoch.

    ``bar_proportional`` is an ordinary without-replacement shuffle. ``uniform_stream`` assigns
    equal counts (within one row) to each stream represented by ``rows``. Small streams are
    cycled with reshuffling when necessary, while large streams are downsampled for that epoch;
    subsequent epochs change the draw so the complete corpus remains reachable.
    """
    rows = np.asarray(rows, dtype=np.int64)
    stream_ids = np.asarray(stream_ids, dtype=object)
    if rows.ndim != 1 or stream_ids.ndim != 1:
        raise ValueError("rows and stream_ids must be one-dimensional")
    if len(rows) == 0:
        return rows.copy()
    if rows.min() < 0 or rows.max() >= len(stream_ids):
        raise ValueError("rows contain an index outside stream_ids")
    if mode not in SAMPLING_MODES:
        raise ValueError(f"unsupported sampling mode {mode!r}; expected one of {SAMPLING_MODES}")

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    if mode == "bar_proportional":
        return rows[rng.permutation(len(rows))]

    row_streams = stream_ids[rows]
    streams = np.asarray(sorted(set(str(value) for value in row_streams)), dtype=object)
    if not len(streams):
        raise ValueError("uniform_stream requires at least one represented stream")
    base, remainder = divmod(len(rows), len(streams))
    allocation = np.full(len(streams), base, dtype=np.int64)
    if remainder:
        allocation[rng.permutation(len(streams))[:remainder]] += 1

    sampled = []
    for stream, count in zip(streams, allocation):
        pool = rows[np.asarray([str(value) == stream for value in row_streams], dtype=bool)]
        if not len(pool):  # pragma: no cover - guarded by construction above
            raise RuntimeError(f"stream {stream!r} has no rows")
        need = int(count)
        while need:
            take = min(need, len(pool))
            sampled.append(pool[rng.permutation(len(pool))[:take]])
            need -= take
    result = np.concatenate(sampled).astype(np.int64, copy=False)
    return result[rng.permutation(len(result))]
