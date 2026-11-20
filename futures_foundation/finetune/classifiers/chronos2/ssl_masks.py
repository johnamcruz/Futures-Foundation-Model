"""Cross-modal block masks for the Chronos-2 mask stage (optional; the default shared time mask is unchanged).

FFM's shared mask hides the same time steps in every variate, so a hidden flow value can always be rebuilt from the
price variates' neighbouring steps *and the reverse is never forced*.  The ``cross`` mode hides a contiguous block of one
modality while the other stays fully visible, so the model must explain flow from price or price from flow.
"""
from __future__ import annotations

import numpy as np

MODES = ("time", "cross", "tail")


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


def modal_time_mask(batch: int, channels: int, length: int, ratio: float, rng: np.random.Generator, *,
                    n_price: int = 5, shared_share: float = 1.0 / 3.0) -> np.ndarray:
    """Boolean [batch, channels, length]: True = hidden.

    Per window: with probability ``shared_share`` the usual shared random time mask over all variates; otherwise, equally
    often, a block of the flow variates (price visible) or a block of the price variates (flow visible).
    """
    if channels <= n_price:
        raise ValueError("cross-modal masking needs at least one non-price variate")
    out = np.zeros((batch, channels, length), dtype=bool)
    for b in range(batch):
        draw = rng.random()
        if draw < shared_share:
            shared = rng.random(length) < ratio
            shared[0] = True
            out[b, :, :] = shared[None, :]
        else:
            blocks = block_mask(length, ratio, rng)
            if draw < shared_share + (1.0 - shared_share) / 2.0:
                out[b, n_price:, :] = blocks[None, :]          # flow hidden, price visible
            else:
                out[b, :n_price, :] = blocks[None, :]          # price hidden, flow visible
    return out


def tail_mask(batch: int, channels: int, length: int, rng: np.random.Generator, *, lo: int = 2, hi: int = 12) -> np.ndarray:
    """Boolean [batch, channels, length]: the last L steps (L uniform in [lo, hi]) of EVERY variate are hidden.

    A forecast-shaped mask: flow in the hidden span cannot be used as a same-bar shortcut; the loss is taken on price only
    (the caller restricts ``selected`` to the price variates), so flow can only help by predicting the next price bars.
    """
    out = np.zeros((batch, channels, length), dtype=bool)
    for b in range(batch):
        out[b, :, length - int(rng.integers(lo, hi + 1)):] = True
    return out
