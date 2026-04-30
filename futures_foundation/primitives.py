"""
futures_foundation.primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared signal detection building blocks used by both:
  - Colab training scripts (labeling + feature generation)
  - algoTraderAI inference modules (real-time feature computation)

All functions operate on numpy arrays for speed and portability.
Pandas is only used where resampling to a higher timeframe is required.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ──────────────────────────────────────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────────────────────────────────────

def compute_atr(h, l, c, period):
    """
    Wilder's ATR on arrays of equal length n.

    Returns float64 array (n,). The first element uses the first true range
    as the seed; subsequent elements use Wilder's smoothing:
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


# ──────────────────────────────────────────────────────────────────────────────
# SuperTrend
# ──────────────────────────────────────────────────────────────────────────────

def compute_supertrend(h, l, c, period, mult):
    """
    Wilder's ATR SuperTrend.

    Returns:
        direction : int8 array (n,)   — +1=bull, -1=bear
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


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

def compute_ema(arr, period):
    """
    Exponential Moving Average.

    Returns float64 array (n,). Seeds with the first value; uses standard
    multiplier k = 2/(period+1).
    """
    arr = np.asarray(arr, dtype=np.float64)
    n   = len(arr)
    k   = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Rolling utilities
# ──────────────────────────────────────────────────────────────────────────────

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
        start  = max(0, i - window)
        window_ = atr[start:i]
        out[i] = float(np.sum(window_ < atr[i])) / max(len(window_), 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pivot detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_pivots(highs, lows, period):
    """
    Vectorised pivot high / pivot low detection.

    A bar at index b is a pivot high if highs[b] is the unique maximum of
    the (2*period+1) window centred on b. Same for pivot lows.

    Returns:
        pivot_high_bars : int array of bar indices (confirmed at bar+period)
        pivot_low_bars  : int array of bar indices (confirmed at bar+period)
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows  = np.asarray(lows,  dtype=np.float64)
    n     = len(highs)

    if n < 2 * period + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    win_h   = sliding_window_view(highs, 2 * period + 1)
    win_l   = sliding_window_view(lows,  2 * period + 1)
    win_max = win_h.max(axis=1)
    win_min = win_l.min(axis=1)
    ch      = win_h[:, period]
    cl      = win_l[:, period]

    is_ph = (ch == win_max) & (np.sum(win_h == ch[:, None], axis=1) == 1)
    is_pl = (cl == win_min) & (np.sum(win_l == cl[:, None], axis=1) == 1)

    return np.where(is_ph)[0] + period, np.where(is_pl)[0] + period


# ──────────────────────────────────────────────────────────────────────────────
# RR barrier evaluation
# ──────────────────────────────────────────────────────────────────────────────

def apply_rr_barriers(h, l, c, entry_idx, is_long, entry_price, sl_price,
                      rr_targets, lookahead=None, is_session_end=None):
    """
    Walk forward from entry_idx measuring R:R outcomes for each target in rr_targets.

    Stops early on:
      - Stop hit (lows[j] <= sl_price for long; highs[j] >= sl_price for short)
      - Session end (if is_session_end[j] is True — uses close as exit price)
      - lookahead exhausted (if provided)
      - Data end

    Args:
        h, l, c         : float arrays aligned to the bar series
        entry_idx       : int — bar index of entry
        is_long         : bool
        entry_price     : float
        sl_price        : float
        rr_targets      : list[float] — e.g. [1.0, 1.5, 2.0, 3.0]
        lookahead       : int or None — max bars to look forward
        is_session_end  : bool array or None

    Returns:
        dict[float → dict] with keys: hit (bool), outcome (str), realized_rr (float)
        outcomes: 'target_hit', 'stopped', 'session_end', 'data_end'
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0}
                for rr in rr_targets}

    targets = {
        rr: entry_price + stop_dist * rr * (1 if is_long else -1)
        for rr in rr_targets
    }
    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0}
               for rr in rr_targets}

    end = n if lookahead is None else min(entry_idx + lookahead + 1, n)

    for j in range(entry_idx + 1, end):
        # Session end — exit at close
        if is_session_end is not None and is_session_end[j]:
            fr = ((c[j] - entry_price) / stop_dist if is_long
                  else (entry_price - c[j]) / stop_dist)
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome']      = 'session_end'
                    results[rr]['realized_rr']  = fr
            break

        # Stop hit
        if (is_long and l[j] <= sl_price) or (not is_long and h[j] >= sl_price):
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome']     = 'stopped'
                    results[rr]['realized_rr'] = -1.0
            break

        # Target hits (check in order so smaller targets resolve first)
        for rr in sorted(rr_targets):
            if results[rr]['outcome'] is not None:
                continue
            if ((is_long and h[j] >= targets[rr]) or
                    (not is_long and l[j] <= targets[rr])):
                results[rr]['hit']         = True
                results[rr]['outcome']     = 'target_hit'
                results[rr]['realized_rr'] = rr

        if all(r['outcome'] is not None for r in results.values()):
            break

    for rr in rr_targets:
        if results[rr]['outcome'] is None:
            results[rr]['outcome']     = 'data_end'
            results[rr]['realized_rr'] = 0.0

    return results


def best_rr_hit(rr_results, min_rr=1.0):
    """Return the highest R:R that was hit, or 0.0 if none reached min_rr."""
    return max(
        (rr for rr, r in rr_results.items() if r['hit'] and rr >= min_rr),
        default=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session utilities
# ──────────────────────────────────────────────────────────────────────────────

def session_mask(index, start_hour=7, start_min=0, end_hour=16, end_min=0):
    """
    Boolean mask for bars inside the given session window.
    index : DatetimeIndex with tz-aware NY timestamps.
    Returns bool array (n,).
    """
    t    = index.hour * 60 + index.minute
    s    = start_hour * 60 + start_min
    e    = end_hour   * 60 + end_min
    return (t >= s) & (t < e)


def session_end_mask(index, start_hour=7, start_min=0, end_hour=16, end_min=0):
    """
    Boolean mask marking the last bar inside each session window
    (i.e. the bar just before the session closes).
    """
    in_sess = session_mask(index, start_hour, start_min, end_hour, end_min)
    out     = np.zeros(len(index), dtype=bool)
    out[:-1] = in_sess[:-1] & ~in_sess[1:]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# VWAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_vwap(h, l, c, v, session_start_mask):
    """
    Session-reset VWAP. Resets at each True entry in session_start_mask.

    Returns float64 array (n,). Bars before the first session start are NaN.
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = len(c)

    tp  = (h + l + c) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)

    cum_tpv = 0.0
    cum_v   = 0.0
    in_session = False

    for i in range(n):
        if session_start_mask[i]:
            cum_tpv    = 0.0
            cum_v      = 0.0
            in_session = True
        if in_session:
            cum_tpv += tp[i] * v[i]
            cum_v   += v[i]
            out[i]   = cum_tpv / cum_v if cum_v > 0 else tp[i]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# CISD (Change In State of Delivery) detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_cisd_signals(o, h, l, c, tolerance=0.70, expiry_bars=50,
                        body_ratio_min=0.50, close_str_min=0.60):
    """
    Detect CISD (Change In State of Delivery) displacement candles.

    A bearish CISD forms when:
      - A prior bearish candle's open (pot_price) is breached to the upside
        and closes back below it on a strong bear bar (body_ratio + close_str).
    A bullish CISD forms symmetrically.

    All arrays must be equal-length numpy arrays of OHLC prices.

    Args:
        o, h, l, c       : float64 arrays of OHLC
        tolerance        : minimum sweep ratio for a valid CISD (default 0.70)
        expiry_bars      : potential zones expire after this many bars
        body_ratio_min   : minimum body/range for displacement candle
        close_str_min    : minimum close-strength for displacement candle
                           (bear: upper wick / range; bull: lower wick / range)

    Returns dict of float64/int8 arrays (all length n):
        cisd_signal       : int8  — 0=none, 1=bearish CISD, 2=bullish CISD
        displacement_str  : float32 — sweep ratio of displacement candle
        disp_body_ratio   : float32 — body/range of displacement candle
        disp_close_str    : float32 — close strength of displacement candle
        origin_level      : float64 — price level of the triggering potential
    """
    from collections import deque

    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    cisd_signal      = np.zeros(n, dtype=np.int8)
    disp_strength    = np.zeros(n, dtype=np.float32)
    disp_body_ratio  = np.zeros(n, dtype=np.float32)
    disp_close_str   = np.zeros(n, dtype=np.float32)
    origin_level     = np.full(n, np.nan, dtype=np.float64)

    bear_pots = deque()   # (open_price, bar_index) for bearish potential zones
    bull_pots = deque()   # (open_price, bar_index) for bullish potential zones

    for bar in range(1, n):
        # Add new potentials: direction change candles
        if c[bar - 1] < o[bar - 1] and c[bar] > o[bar]:   # bear→bull flip
            bear_pots.append((o[bar], bar))
        if c[bar - 1] > o[bar - 1] and c[bar] < o[bar]:   # bull→bear flip
            bull_pots.append((o[bar], bar))

        # Expire old potentials
        while bear_pots and bar - bear_pots[0][1] >= expiry_bars:
            bear_pots.popleft()
        while bull_pots and bar - bull_pots[0][1] >= expiry_bars:
            bull_pots.popleft()

        # ── Bearish CISD ──
        while bear_pots:
            pot_price, pot_bar = bear_pots[0]
            if c[bar] < pot_price:
                highest_c = c[pot_bar:bar + 1].max()
                top_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] < o[idx]:
                    top_level = o[idx]
                    idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > tolerance:
                        full_range = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (h[bar] - c[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 1
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            bear_pots.clear()
                            break
                        else:
                            bear_pots.popleft()
                            continue
                    else:
                        bear_pots.popleft()
                        continue
                else:
                    bear_pots.popleft()
                    continue
            else:
                break

        # ── Bullish CISD ──
        while bull_pots:
            pot_price, pot_bar = bull_pots[0]
            if c[bar] > pot_price:
                lowest_c     = c[pot_bar:bar + 1].min()
                bottom_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] > o[idx]:
                    bottom_level = o[idx]
                    idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > tolerance:
                        full_range = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (c[bar] - l[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 2
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            bull_pots.clear()
                            break
                        else:
                            bull_pots.popleft()
                            continue
                    else:
                        bull_pots.popleft()
                        continue
                else:
                    bull_pots.popleft()
                    continue
            else:
                break

    return {
        'cisd_signal':      cisd_signal,
        'displacement_str': disp_strength,
        'disp_body_ratio':  disp_body_ratio,
        'disp_close_str':   disp_close_str,
        'origin_level':     origin_level,
    }


def compute_ote_zones(cisd_result, h, l, c, fib_1=0.618, fib_2=0.786):
    """
    Compute OTE (Optimal Trade Entry) fibonacci zones from CISD signals.

    For a bearish CISD (signal=1): uses highest high from potential to signal bar.
    For a bullish CISD (signal=2): uses lowest low from potential to signal bar.

    Returns:
        fib_top : float64 array (n,) — upper boundary of OTE zone
        fib_bot : float64 array (n,) — lower boundary of OTE zone
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    cisd_signal = cisd_result['cisd_signal']
    fib_top = np.full(n, np.nan, dtype=np.float64)
    fib_bot = np.full(n, np.nan, dtype=np.float64)

    for bar in range(n):
        sig = cisd_signal[bar]
        if sig == 1:    # bearish
            h_max = h[:bar + 1].max()
            diff  = h_max - l[bar]
            if diff > 0:
                f1 = h_max - diff * fib_1
                f2 = h_max - diff * fib_2
                fib_top[bar] = max(f1, f2)
                fib_bot[bar] = min(f1, f2)
        elif sig == 2:  # bullish
            l_min = l[:bar + 1].min()
            diff  = h[bar] - l_min
            if diff > 0:
                f1 = l_min + diff * fib_1
                f2 = l_min + diff * fib_2
                fib_top[bar] = max(f1, f2)
                fib_bot[bar] = min(f1, f2)

    return fib_top, fib_bot
