"""
Label generation for pretraining tasks.

All labels derived automatically from OHLCV — no manual annotation needed.

IMPORTANT: Labels are FORWARD-LOOKING. Each label describes what happens
in the NEXT N bars, not the current state. This forces the backbone to
learn predictive representations rather than identity mappings.

Tasks:
    1. Regime (next 20 bars): trending up/down, rotational, volatile expansion
    2. Volatility State (next 10 bars): low, normal, elevated, extreme
    3. Market Structure (next 20 bars): bullish, bearish, mixed
    4. Range Position (next 10 bars): quintile within recent range

The model sees features at time T and must predict labels computed from
bars T+1 through T+N. This is the core of self-supervised pretraining:
compress current market context into representations that predict future
market behavior.
"""

import numpy as np
import pandas as pd


REGIME_LABELS = {0: "trending_up", 1: "trending_down", 2: "rotational", 3: "volatile_expansion"}
VOLATILITY_LABELS = {0: "low", 1: "normal", 2: "elevated", 3: "extreme"}
STRUCTURE_LABELS = {0: "bullish", 1: "bearish", 2: "mixed"}
RANGE_LABELS = {0: "q1_0_20", 1: "q2_20_40", 2: "q3_40_60", 3: "q4_60_80", 4: "q5_80_100"}


def generate_regime_labels(features, horizon=20, atr_expansion_threshold=1.5):
    """
    Predict the FUTURE regime over the next `horizon` bars.

    Uses forward-shifted returns and volatility to classify:
      0 = trending up:   strong positive forward returns
      1 = trending down: strong negative forward returns
      2 = rotational:    low forward returns, low volatility
      3 = volatile:      high forward volatility regardless of direction
    """
    close = features["_close"]

    # Forward return over next horizon bars
    fwd_return = close.shift(-horizon) / close - 1.0

    # Forward realized volatility (std of per-bar returns over next horizon)
    bar_returns = close.pct_change(1)
    fwd_volatility = bar_returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))

    # Thresholds based on rolling context
    vol_median = fwd_volatility.rolling(200, min_periods=50).median()
    vol_threshold = vol_median * atr_expansion_threshold

    ret_std = fwd_return.rolling(200, min_periods=50).std()
    ret_threshold = ret_std * 0.8  # ~0.8 sigma = meaningful move

    labels = pd.Series(2, index=features.index, dtype="Int64")  # default: rotational

    # Mask rows where forward data is unavailable
    valid = fwd_return.notna() & fwd_volatility.notna() & vol_threshold.notna() & ret_threshold.notna()

    # Volatile: high forward vol regardless of direction
    volatile_mask = (fwd_volatility > vol_threshold) & valid
    labels[volatile_mask] = 3

    # Trending (only if not volatile)
    labels[(fwd_return > ret_threshold) & (~volatile_mask) & valid] = 0
    labels[(fwd_return < -ret_threshold) & (~volatile_mask) & valid] = 1

    # NaN where we can't compute forward labels
    labels[~valid] = pd.NA

    return labels


def generate_volatility_labels(features, horizon=10):
    """
    Predict FUTURE volatility state over the next `horizon` bars.

    Uses forward realized vol ranked against recent history.
    """
    close = features["_close"]

    # Forward realized vol: std of returns over next horizon bars
    bar_returns = close.pct_change(1)
    fwd_vol = bar_returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))

    # Rank against trailing 100-bar distribution
    pct_rank = fwd_vol.rolling(100, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    labels = pd.Series(1, index=features.index, dtype="Int64")  # default: normal

    valid = pct_rank.notna()
    labels[valid & (pct_rank < 0.25)] = 0    # low
    labels[valid & (pct_rank >= 0.60) & (pct_rank < 0.85)] = 2  # elevated
    labels[valid & (pct_rank >= 0.85)] = 3   # extreme
    labels[~valid] = pd.NA

    return labels


def generate_structure_labels(features, horizon=20):
    """
    Predict FUTURE market structure over the next `horizon` bars.

    Bullish: upside expansion > downside from current close
    Bearish: downside expansion > upside
    Mixed:   neither dominates
    """
    close = features["_close"]

    # Forward price extremes
    fwd_max = close.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    fwd_min = close.shift(-1).rolling(horizon).min().shift(-(horizon - 1))

    # How far price goes up vs down from current close
    upside = (fwd_max - close) / close
    downside = (close - fwd_min) / close

    # Asymmetry ratio: >1 = bullish, <1 = bearish
    asymmetry = upside / downside.replace(0, np.nan)

    labels = pd.Series(2, index=features.index, dtype="Int64")  # default: mixed

    valid = asymmetry.notna()
    labels[valid & (asymmetry > 1.5)] = 0   # bullish
    labels[valid & (asymmetry < 0.67)] = 1  # bearish
    labels[~valid] = pd.NA

    return labels


def generate_range_labels(features, horizon=10, lookback=20):
    """
    Predict where the FUTURE close will sit within the current range.

    Uses the current lookback-bar range as reference, then checks where
    the close at T+horizon falls within that range.
    """
    close = features["_close"]

    # Current range context (the model can see this)
    recent_high = close.rolling(lookback).max()
    recent_low = close.rolling(lookback).min()
    range_width = (recent_high - recent_low).replace(0, np.nan)

    # Future close position within current range
    fwd_close = close.shift(-horizon)
    fwd_position = ((fwd_close - recent_low) / range_width).clip(0, 1)

    labels = pd.Series(2, index=features.index, dtype="Int64")  # default: q3 (middle)

    valid = fwd_position.notna()
    labels[valid & (fwd_position < 0.2)] = 0
    labels[valid & (fwd_position >= 0.2) & (fwd_position < 0.4)] = 1
    labels[valid & (fwd_position >= 0.4) & (fwd_position < 0.6)] = 2
    labels[valid & (fwd_position >= 0.6) & (fwd_position < 0.8)] = 3
    labels[valid & (fwd_position >= 0.8)] = 4
    labels[~valid] = pd.NA

    return labels


def generate_all_labels(features):
    """Generate all 4 forward-looking pretraining labels from feature DataFrame."""
    labels = pd.DataFrame(index=features.index)
    labels["regime_label"] = generate_regime_labels(features)
    labels["volatility_label"] = generate_volatility_labels(features)
    labels["structure_label"] = generate_structure_labels(features)
    labels["range_label"] = generate_range_labels(features)
    return labels


def print_label_distribution(labels):
    label_maps = {
        "regime_label": REGIME_LABELS, "volatility_label": VOLATILITY_LABELS,
        "structure_label": STRUCTURE_LABELS, "range_label": RANGE_LABELS,
    }
    for col in labels.columns:
        print(f"\n{'='*50}\n  {col}\n{'='*50}")
        counts = labels[col].value_counts().sort_index()
        total = len(labels[col].dropna())
        for val, count in counts.items():
            name = label_maps.get(col, {}).get(val, str(val))
            print(f"  {val} ({name:>20s}): {count:>8,d}  ({count/total*100:5.1f}%)")