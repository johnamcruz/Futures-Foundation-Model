"""
Candle psychology features for the FFM backbone.

Strategy-agnostic price action descriptors computed from raw OHLCV only.
No CISD, OTE, or strategy-specific logic here.

Public API:
    add_candle_features(df) -> pd.DataFrame
        Accepts df with columns: open, high, low, close, volume
        Returns df with 6 new columns appended:
            body_ratio, upper_wick_ratio, lower_wick_ratio,
            candle_type, engulf_count, momentum_speed_ratio

Config params (pass overrides to add_candle_features):
    ENGULF_LOOKBACK = 5
    MOMENTUM_WINDOW = 20
"""

import numpy as np
import pandas as pd


ENGULF_LOOKBACK = 5
MOMENTUM_WINDOW = 20


def add_candle_features(
    df: pd.DataFrame,
    engulf_lookback: int = ENGULF_LOOKBACK,
    momentum_window: int = MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """
    Append 6 candle psychology features to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        engulf_lookback: Prior bars checked for body engulfing (default 5)
        momentum_window: Rolling window for impulse/retrace speed ratio (default 20)

    Returns:
        Copy of df with 6 new float32 columns appended.
    """
    df = df.copy()

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bar_range = h - l + 1e-9

    body_abs = (c - o).abs()
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    upper_wick = h - body_high
    lower_wick = body_low - l

    body_ratio = (body_abs / bar_range).clip(0.0, 1.0)
    upper_wick_ratio = (upper_wick / bar_range).clip(0.0, 1.0)
    lower_wick_ratio = (lower_wick / bar_range).clip(0.0, 1.0)

    df["body_ratio"] = body_ratio.astype(np.float32)
    df["upper_wick_ratio"] = upper_wick_ratio.astype(np.float32)
    df["lower_wick_ratio"] = lower_wick_ratio.astype(np.float32)
    df["candle_type"] = _classify_candle(
        body_ratio, upper_wick_ratio, lower_wick_ratio, c >= o
    ).astype(np.float32)
    df["engulf_count"] = _compute_engulf_count(
        body_high.values, body_low.values, lookback=engulf_lookback, index=df.index
    )
    df["momentum_speed_ratio"] = _compute_momentum_speed_ratio(
        c.values, window=momentum_window, index=df.index
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
    """
    n = len(body_highs)
    result = np.zeros(n, dtype=np.float32)

    for idx in range(1, n):
        cbh = body_highs[idx]
        cbl = body_lows[idx]
        count = 0
        for i in range(1, lookback + 1):
            if idx - i < 0:
                break
            if body_highs[idx - i] <= cbh and body_lows[idx - i] >= cbl:
                count += 1
        result[idx] = float(count)

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
    """
    n = len(closes)
    result = np.ones(n, dtype=np.float32)

    for idx in range(window, n):
        wc = closes[idx - window: idx]
        peak_idx = int(np.argmax(wc))
        trough_idx = int(np.argmin(wc))

        impulse_pts = wc[peak_idx] - wc[trough_idx]  # always >= 0 (max - min)
        if impulse_pts == 0.0:
            # Flat window — no meaningful move; leave as neutral 1.0
            continue

        if peak_idx > trough_idx:
            # Trough → Peak: upward impulse
            impulse_bars = max(peak_idx - trough_idx, 1)
            retrace_bars = max((window - 1) - peak_idx, 1)
            retrace_pts = abs(wc[-1] - wc[peak_idx])
        else:
            # Peak → Trough: downward impulse
            impulse_bars = max(trough_idx - peak_idx, 1)
            retrace_bars = max((window - 1) - trough_idx, 1)
            retrace_pts = abs(wc[-1] - wc[trough_idx])

        impulse_speed = impulse_pts / impulse_bars
        retrace_speed = retrace_pts / retrace_bars
        result[idx] = float(np.clip(impulse_speed / (retrace_speed + 1e-9), 0.0, 10.0))

    return pd.Series(result, index=index)
