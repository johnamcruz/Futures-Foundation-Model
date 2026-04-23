"""
Candle psychology features for the FFM backbone.

Strategy-agnostic price action descriptors computed from raw OHLCV only.
No CISD, OTE, or strategy-specific logic here.

Public API:
    add_candle_features(df) -> pd.DataFrame
        Accepts df with columns: open, high, low, close, volume
        Returns df with 5 new columns appended:
            candle_type, engulf_count, momentum_speed_ratio,
            wick_rejection, dir_consistency

Config params (pass overrides to add_candle_features):
    ENGULF_LOOKBACK = 5
    MOMENTUM_WINDOW = 20
    DIRECTION_WINDOW = 5
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ENGULF_LOOKBACK = 5
MOMENTUM_WINDOW = 20
DIRECTION_WINDOW = 5


def add_candle_features(
    df: pd.DataFrame,
    engulf_lookback: int = ENGULF_LOOKBACK,
    momentum_window: int = MOMENTUM_WINDOW,
    direction_window: int = DIRECTION_WINDOW,
) -> pd.DataFrame:
    """
    Append 5 candle psychology features to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        engulf_lookback: Prior bars checked for body engulfing (default 5)
        momentum_window: Rolling window for impulse/retrace speed ratio (default 20)
        direction_window: Rolling window for directional consistency (default 5)

    Returns:
        Copy of df with 5 new float32 columns appended.
    """
    df = df.copy()

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bar_range = h - l + 1e-9

    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    upper_wick = h - body_high
    lower_wick = body_low - l

    body_ratio = ((c - o).abs() / bar_range).clip(0.0, 1.0)
    upper_wick_ratio = (upper_wick / bar_range).clip(0.0, 1.0)
    lower_wick_ratio = (lower_wick / bar_range).clip(0.0, 1.0)

    df["candle_type"] = _classify_candle(
        body_ratio, upper_wick_ratio, lower_wick_ratio, c >= o
    ).astype(np.float32)
    df["engulf_count"] = _compute_engulf_count(
        body_high.values, body_low.values, lookback=engulf_lookback, index=df.index
    )
    df["momentum_speed_ratio"] = _compute_momentum_speed_ratio(
        c.values, window=momentum_window, index=df.index
    )
    # Signed wick asymmetry: positive = lower wick dominates (bullish rejection),
    # negative = upper wick dominates (bearish rejection). Range [-1, 1].
    df["wick_rejection"] = ((lower_wick - upper_wick) / bar_range).clip(-1.0, 1.0).astype(np.float32)
    df["dir_consistency"] = _compute_dir_consistency(
        c.values, o.values, window=direction_window, index=df.index
    )

    return df


def _classify_candle(
    body_ratio: pd.Series,
    upper_wick_ratio: pd.Series,
    lower_wick_ratio: pd.Series,
    is_bullish: pd.Series,
) -> pd.Series:
    """
    Classify each bar into one of 6 candle types.

    Returns:
        0 = doji / indecision
        1 = bull strength candle
        2 = bear strength candle
        3 = bull pin bar (buyer rejection at lows)
        4 = bear pin bar (seller rejection at highs)
        5 = neutral / mixed
    """
    wick_asymmetry = lower_wick_ratio - upper_wick_ratio
    conditions = [
        body_ratio < 0.15,
        (body_ratio > 0.70) & is_bullish,
        (body_ratio > 0.70) & ~is_bullish,
        wick_asymmetry > 0.30,
        wick_asymmetry < -0.30,
    ]
    choices = [0, 1, 2, 3, 4]
    return pd.Series(
        np.select(conditions, choices, default=5), index=body_ratio.index
    )


def _compute_engulf_count(
    body_highs: np.ndarray,
    body_lows: np.ndarray,
    lookback: int = ENGULF_LOOKBACK,
    index=None,
) -> pd.Series:
    """
    Count how many of the prior `lookback` bars are fully contained within
    the current bar's body (body-to-body engulfing).

    Range: [0.0, lookback]. First bar is always 0.0.
    Vectorized: O(lookback) passes over the array, no Python per-bar loop.
    """
    result = np.zeros(len(body_highs), dtype=np.float32)
    for i in range(1, lookback + 1):
        # Prior bar at offset i is engulfed by the current bar when its body
        # fits entirely within the current bar's body boundaries.
        engulfed = (body_highs[:-i] <= body_highs[i:]) & (body_lows[:-i] >= body_lows[i:])
        result[i:] += engulfed
    return pd.Series(result, index=index)


def _compute_momentum_speed_ratio(
    closes: np.ndarray,
    window: int = MOMENTUM_WINDOW,
    index=None,
) -> pd.Series:
    """
    Ratio of impulse speed to retracement speed over a rolling window.

    Identifies the dominant leg (trough→peak or peak→trough) as the impulse,
    and the remaining bars to window end as the retrace.

    Range: [0.0, 10.0] (clipped). Default 1.0 for bars before `window`.
    Vectorized via sliding_window_view — no Python per-bar loop.
    """
    n = len(closes)
    result = np.ones(n, dtype=np.float32)

    if n <= window:
        return pd.Series(result, index=index)

    # wc[k] = closes[k : k+window] for k=0..n-window-1
    # result[window+k] gets the ratio for wc[k]
    wc = sliding_window_view(closes.astype(np.float64), window_shape=window)[:-1]

    peak_idxs = np.argmax(wc, axis=1)
    trough_idxs = np.argmin(wc, axis=1)

    rows = np.arange(len(wc))
    peak_vals = wc[rows, peak_idxs]
    trough_vals = wc[rows, trough_idxs]
    impulse_pts = peak_vals - trough_vals  # always >= 0

    is_up = peak_idxs > trough_idxs
    last_vals = wc[:, -1]

    impulse_bars = np.maximum(
        np.where(is_up, peak_idxs - trough_idxs, trough_idxs - peak_idxs), 1
    ).astype(np.float64)
    retrace_pts = np.where(
        is_up, np.abs(last_vals - peak_vals), np.abs(last_vals - trough_vals)
    )
    retrace_bars = np.maximum(
        np.where(is_up, (window - 1) - peak_idxs, (window - 1) - trough_idxs), 1
    ).astype(np.float64)

    ratio = np.clip(
        (impulse_pts / impulse_bars) / (retrace_pts / retrace_bars + 1e-9), 0.0, 10.0
    )
    # Flat windows (no meaningful move) stay at neutral 1.0
    ratio = np.where(impulse_pts == 0.0, 1.0, ratio)

    result[window:] = ratio.astype(np.float32)
    return pd.Series(result, index=index)


def _compute_dir_consistency(
    closes: np.ndarray,
    opens: np.ndarray,
    window: int = DIRECTION_WINDOW,
    index=None,
) -> pd.Series:
    """
    Fraction of the last `window` bars (including current) whose close-open
    direction matches the current bar's direction.

    Doji bars (close == open) return 0.5 (neutral). Range: [0.0, 1.0].
    """
    directions = np.sign(closes - opens)
    n = len(directions)
    result = np.full(n, 0.5, dtype=np.float32)

    for i in range(n):
        curr = directions[i]
        if curr == 0:
            continue
        start = max(0, i - window + 1)
        w = directions[start: i + 1]
        result[i] = float((w == curr).sum() / len(w))

    return pd.Series(result, index=index)
