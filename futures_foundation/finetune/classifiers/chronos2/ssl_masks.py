"""Optional mask objectives for the Chronos-2 mask stage (the default shared time mask is unchanged and lives in ssl_stages).

FFM's shared mask hides the same time steps in every variate, so a hidden flow value can always be rebuilt from the
neighbouring steps and the cross-variate link is never forced.  Two optional modes use the variate roles taken from the
channel names (``<ticker>.flow_*`` = flow, everything else = price), never from channel positions:

* ``cross``: per window, either the shared random time mask, a contiguous block of the flow variates (price visible) or a
  block of the price variates (flow visible);
* ``tail``: the last 2-12 steps of every variate are hidden; the loss counts the price variates only.
"""
from __future__ import annotations

import numpy as np

MODES = ("time", "cross", "tail")
FLOW_MARK = ".flow_"


def flow_flags(channel_names) -> np.ndarray:
    """Boolean per channel: True for flow variates (``<ticker>.flow_*``)."""
    return np.asarray([FLOW_MARK in str(name) for name in channel_names], dtype=bool)


def block_mask(length: int, ratio: float, rng: np.random.Generator, *, min_len: int = 8, max_len: int = 48) -> np.ndarray:
    """Boolean [length] mask built from contiguous blocks covering about ``ratio`` of the steps; index 0 is always hidden."""
    mask = np.zeros(length, dtype=bool)
    target = max(1, int(round(length * ratio)))
    guard = 0
    while mask.sum() < target and guard < 1000:
        size = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(1, length - size + 1)))
        mask[start:start + size] = True
        guard += 1
    mask[0] = True
    return mask


def _shared_mask(length: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    shared = rng.random(length) < ratio
    shared[0] = True
    return shared


def modal_time_mask(batch: int, is_flow: np.ndarray, length: int, ratio: float, rng: np.random.Generator, *,
                    shared_share: float = 1.0 / 3.0) -> np.ndarray:
    """Boolean [batch, channels, length], True = hidden (see module docstring, ``cross``)."""
    is_flow = np.asarray(is_flow, dtype=bool)
    if not is_flow.any() or is_flow.all():
        raise ValueError("cross-modal masking needs both flow and price variates")
    out = np.zeros((batch, len(is_flow), length), dtype=bool)
    for b in range(batch):
        draw = rng.random()
        if draw < shared_share:
            out[b, :, :] = _shared_mask(length, ratio, rng)[None, :]
        else:
            blocks = block_mask(length, ratio, rng)
            hide_flow = draw < shared_share + (1.0 - shared_share) / 2.0
            out[b, is_flow if hide_flow else ~is_flow, :] = blocks[None, :]
    return out


def tail_mask(batch: int, channels: int, length: int, rng: np.random.Generator, *, lo: int = 2, hi: int = 12) -> np.ndarray:
    """Boolean [batch, channels, length]: the last L steps (L uniform in [lo, hi]) of EVERY variate are hidden."""
    out = np.zeros((batch, channels, length), dtype=bool)
    for b in range(batch):
        out[b, :, length - int(rng.integers(lo, hi + 1)):] = True
    return out


def selection_masks(finite: np.ndarray, is_flow: np.ndarray, mode: str, ratio: float, rng: np.random.Generator):
    """(selected, visible) boolean [B, C, T] for ``cross`` or ``tail``; ``None`` when ``cross`` has no flow variate.

    selected = entries the loss counts; visible = entries the encoder may see.  Any window in which a variate that has
    finite values would have no visible step is redrawn as the shared time mask.
    """
    finite = np.asarray(finite, dtype=bool)
    is_flow = np.asarray(is_flow, dtype=bool)
    batch, channels, length = finite.shape
    if mode == "cross":
        if not is_flow.any():
            return None
        hide = modal_time_mask(batch, is_flow, length, ratio, rng)
        selected = finite & hide
    elif mode == "tail":
        hide = tail_mask(batch, channels, length, rng)
        selected = finite & hide & ~is_flow[None, :, None]       # loss on price variates only
    else:
        raise ValueError("mode must be 'cross' or 'tail'")
    visible = finite & ~hide
    broken = finite.any(-1) & ~visible.any(-1)                    # a variate left with no visible step
    for b in np.flatnonzero(broken.any(-1)):
        shared = _shared_mask(length, ratio, rng)
        hide[b, :, :] = shared[None, :]
        selected[b] = finite[b] & hide[b] & (~is_flow[:, None] if mode == "tail" else True)
        visible[b] = finite[b] & ~hide[b]
        # the shared mask always hides step 0: give any variate still without a visible step its first finite one back
        for c in np.flatnonzero(finite[b].any(-1) & ~visible[b].any(-1)):
            first = int(np.flatnonzero(finite[b, c])[0])
            visible[b, c, first] = True
            selected[b, c, first] = False
    return selected, visible
