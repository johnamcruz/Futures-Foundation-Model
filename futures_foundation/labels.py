"""
Label generation for pretraining tasks.

All labels derived automatically from OHLCV — no manual annotation needed.

Tasks:
    1. Regime: trending up/down, rotational, volatile expansion
    2. Volatility State: low, normal, elevated, extreme
    3. Market Structure: bullish, bearish, mixed
    4. Range Position: quintile within recent range
"""

import numpy as np
import pandas as pd


REGIME_LABELS = {0: "trending_up", 1: "trending_down", 2: "rotational", 3: "volatile_expansion"}
VOLATILITY_LABELS = {0: "low", 1: "normal", 2: "elevated", 3: "extreme"}
STRUCTURE_LABELS = {0: "bullish", 1: "bearish", 2: "mixed"}
RANGE_LABELS = {0: "q1_0_20", 1: "q2_20_40", 2: "q3_40_60", 3: "q4_60_80", 4: "q5_80_100"}


def generate_regime_labels(features, trend_lookback=20, atr_expansion_threshold=1.5, range_threshold=0.3):
    labels = pd.Series(2, index=features.index, dtype=int)
    momentum = features["ret_momentum_20"]
    atr_z = features["vty_atr_zscore"]
    structure = features["str_structure_state"]

    volatile_mask = atr_z.abs() > atr_expansion_threshold
    labels[volatile_mask] = 3
    labels[(momentum > range_threshold) & (structure >= 0) & (~volatile_mask)] = 0
    labels[(momentum < -range_threshold) & (structure <= 0) & (~volatile_mask)] = 1
    return labels


def generate_volatility_labels(features, lookback=50):
    atr = features.get("vty_atr_raw", features["vty_atr_zscore"])
    pct_rank = atr.rolling(lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    labels = pd.Series(1, index=features.index, dtype=int)
    labels[pct_rank < 0.25] = 0
    labels[(pct_rank >= 0.60) & (pct_rank < 0.85)] = 2
    labels[pct_rank >= 0.85] = 3
    return labels


def generate_structure_labels(features):
    structure = features["str_structure_state"]
    labels = pd.Series(2, index=features.index, dtype=int)
    labels[structure == 1] = 0
    labels[structure == -1] = 1
    return labels


def generate_range_labels(features, lookback=20):
    col = f"str_range_position_{lookback}"
    if col not in features.columns:
        col = "str_range_position_10"
    range_pos = features[col].clip(0, 1)

    labels = pd.Series(2, index=features.index, dtype=int)
    labels[range_pos < 0.2] = 0
    labels[(range_pos >= 0.2) & (range_pos < 0.4)] = 1
    labels[(range_pos >= 0.4) & (range_pos < 0.6)] = 2
    labels[(range_pos >= 0.6) & (range_pos < 0.8)] = 3
    labels[range_pos >= 0.8] = 4
    return labels


def generate_all_labels(features):
    """Generate all 4 pretraining labels from feature DataFrame."""
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