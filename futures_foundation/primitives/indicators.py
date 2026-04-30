"""Indicator computations — ATR, SuperTrend, EMA, RSI."""

import numpy as np
import pandas as pd


def compute_atr(h, l, c, period):
    """
    Wilder's ATR on arrays of equal length n.

    Returns float64 array (n,). Seeds with h[0]-l[0]; uses Wilder's smoothing:
        ATR[i] = ((period-1)*ATR[i-1] + TR[i]) / period
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]),
                        np.abs(l[1:] - c[:-1])))

    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period

    return np.maximum(atr, 1e-9)


def compute_supertrend(h, l, c, period, mult):
    """
    Wilder's ATR SuperTrend.

    Returns:
        direction : int8 array (n,)    — +1=bull, -1=bear
        st_line   : float64 array (n,) — support/resistance line
        atr       : float64 array (n,) — underlying Wilder ATR

    Initialises direction=+1 (bull) so the state machine starts correctly.
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    atr = compute_atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()

    direction = np.ones(n, dtype=np.int8)
    st_line   = lb.copy()

    for i in range(1, n):
        ub[i] = ub[i] if (ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = lb[i] if (lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
        if direction[i - 1] == -1:
            direction[i] = 1 if c[i] > ub[i - 1] else -1
        else:
            direction[i] = -1 if c[i] < lb[i - 1] else 1
        st_line[i] = lb[i] if direction[i] == 1 else ub[i]

    return direction, st_line, atr


def compute_supertrend_htf(df_5m, period, mult, tf='1h'):
    """
    Resample df_5m to tf, compute SuperTrend, forward-fill direction back to
    the 5min index.

    Returns int8 array (n,) aligned to df_5m.index.
    """
    df_htf = df_5m.resample(tf).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()

    d, _, _ = compute_supertrend(
        df_htf['high'].values,
        df_htf['low'].values,
        df_htf['close'].values,
        period, mult,
    )

    return (pd.Series(d, index=df_htf.index)
              .reindex(df_5m.index, method='ffill')
              .fillna(1)
              .values.astype(np.int8))


def compute_ema(arr, period):
    """
    Exponential Moving Average. Seeds with arr[0]; multiplier k = 2/(period+1).
    Returns float64 array (n,).
    """
    arr = np.asarray(arr, dtype=np.float64)
    n   = len(arr)
    k   = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_rsi(closes, period=14):
    """
    Wilder's RSI. Returns float64 array (n,) in [0, 100].
    First `period` values are NaN (insufficient history).
    """
    closes = np.asarray(closes, dtype=np.float64)
    n      = len(closes)
    out    = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return out

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs     = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out
