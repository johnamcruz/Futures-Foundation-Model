"""Rolling window utilities."""

import numpy as np


def rolling_mean(arr, window):
    """
    Rolling mean without pandas. Expanding window for the first `window-1` bars.
    Returns float64 array (n,).
    """
    arr    = np.asarray(arr, dtype=np.float64)
    n      = len(arr)
    out    = np.full(n, np.nan, dtype=np.float64)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(window - 1):
        out[i] = np.mean(arr[:i + 1])
    return out


def rolling_atr_percentile(atr, window=200):
    """
    Rolling ATR percentile rank: fraction of prior `window` ATR values < current.
    Returns float32 array (n,) in [0, 1].
    """
    atr = np.asarray(atr, dtype=np.float64)
    n   = len(atr)
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        start   = max(0, i - window)
        window_ = atr[start:i]
        out[i]  = float(np.sum(window_ < atr[i])) / max(len(window_), 1)
    return out
