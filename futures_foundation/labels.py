"""
Label generation for pretraining tasks.

All labels derived automatically from OHLCV — no manual annotation needed.

IMPORTANT: Labels are FORWARD-LOOKING. Each label describes what happens
in the NEXT N bars, not the current state. This forces the backbone to
learn predictive representations rather than identity mappings.

Tasks:
    1. Regime (next 20 bars): trending up/down, rotational, volatile expansion
    2. Volatility State (next 10 bars): low, normal, elevated, extreme
    3. Market Structure (next 20 bars): confirmed bullish/bearish via 1H structure + expansion
    4. Range Position (next 10 bars): quintile within recent range

The model sees features at time T and must predict labels computed from
bars T+1 through T+N. This is the core of self-supervised pretraining:
compress current market context into representations that predict future
market behavior.
"""

import numpy as np
import pandas as pd

# Sentinel used in regime and structure labels to mark low-confidence samples.
# CrossEntropyLoss(ignore_index=LABEL_CONFIDENCE_SENTINEL) skips these during training.
# Distinct from pd.NA (forward data unavailable); these rows remain in the dataset
# but contribute no gradient for the masked head.
LABEL_CONFIDENCE_SENTINEL = -100

REGIME_LABELS = {0: "trending_up", 1: "trending_down", 2: "rotational", 3: "volatile_expansion"}
VOLATILITY_LABELS = {0: "low", 1: "normal", 2: "elevated", 3: "extreme"}
STRUCTURE_LABELS = {0: "bullish", 1: "bearish"}
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

    # Default to ambiguous — only assign a label when the signal is unambiguous.
    # Borderline bars (near threshold) contribute noise, not signal; mask them out.
    labels = pd.Series(LABEL_CONFIDENCE_SENTINEL, index=features.index, dtype="Int64")

    # Mask rows where forward data is unavailable (tail of series)
    valid = fwd_return.notna() & fwd_volatility.notna() & vol_threshold.notna() & ret_threshold.notna()

    # Volatile: fwd_vol clearly above threshold
    volatile_mask = (fwd_volatility > vol_threshold) & valid
    labels[volatile_mask] = 3

    # Clear non-volatile: below 85% of threshold (avoids borderline expansion)
    not_volatile_mask = (fwd_volatility < vol_threshold * 0.85) & valid

    # Trending: return clearly exceeds threshold AND not volatile
    labels[(fwd_return > ret_threshold) & not_volatile_mask] = 0
    labels[(fwd_return < -ret_threshold) & not_volatile_mask] = 1

    # Rotational: return clearly below 60% of threshold AND clearly not volatile.
    # This is the tightest class — only labeled when consolidation is unambiguous.
    rotational_mask = (fwd_return.abs() < ret_threshold * 0.6) & not_volatile_mask
    labels[rotational_mask] = 2

    # pd.NA where forward data can't be computed (dataset will drop these rows)
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


def generate_structure_labels(features, horizon=48):
    """
    Predict the 1H market structure state N bars forward.

    Uses htf_1h_structure computed in derive_features: majority close direction
    of the last 3 completed 1H bars (+1=bullish, -1=bearish, 0=choppy/mixed).

    horizon must be >= 36 (3 × 12 bars/hour) so that ALL 3 contributing 1H bars
    at T+horizon are bars that completed AFTER T. horizon=20 caused label leakage:
    2 of the 3 contributing bars were identical to the current htf_1h_structure
    feature, making the task trivially easy (95%+ accuracy, no useful gradient).
    horizon=48 (4 hours) ensures 4 complete new 1H bars have passed — no overlap
    with the model's visible context at T.

      0 = bullish: majority of 3 completed 1H bars at T+horizon closed higher
      1 = bearish: majority of 3 completed 1H bars at T+horizon closed lower
      SENTINEL: mixed/choppy (0) or forward data unavailable — masked in training
    """
    raw = features["_1h_structure"]  # Int64: +1, -1, 0, or pd.NA

    fwd = raw.shift(-horizon)

    labels = pd.Series(LABEL_CONFIDENCE_SENTINEL, index=features.index, dtype="Int64")

    valid = fwd.notna()
    labels[valid & (fwd == 1)]  = 0   # bullish
    labels[valid & (fwd == -1)] = 1   # bearish
    # fwd == 0 (choppy) stays at SENTINEL — ambiguous, skip in training

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