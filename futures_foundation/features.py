"""
Feature derivation from raw OHLCV data.

All features are instrument-agnostic via ATR normalization or z-scores,
ensuring the backbone learns transferable patterns across instruments.

Feature Groups (42 total):
    1. Bar anatomy (8) — body, wicks, range normalized by ATR
    2. Returns & momentum (8) — multi-horizon returns + acceleration
    3. Volume dynamics (6) — relative volume, delta proxy
    4. Volatility measures (6) — ATR z-score, range ratios, realized vol
    5. Session-relative context (5) — distance from session OHLC + VWAP
    6. Market structure (9) — swing distances, range position multi-lookback
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# =============================================================================
# Constants
# =============================================================================

INSTRUMENT_MAP = {
    "ES": 0, "NQ": 1, "RTY": 2, "YM": 3,
    "GC": 4, "SI": 5, "CL": 6, "NKD": 7,
}

SESSION_MAP = {"pre_market": 0, "london": 1, "ny_am": 2, "ny_pm": 3}


# =============================================================================
# Main Feature Derivation
# =============================================================================


def derive_features(
    df: pd.DataFrame,
    instrument: str,
    atr_period: int = 14,
    vol_lookbacks: Tuple[int, ...] = (5, 10, 20),
    structure_lookback: int = 10,
) -> pd.DataFrame:
    """
    Derive the full feature set from raw OHLCV data.

    Args:
        df: DataFrame with columns [datetime, open, high, low, close, volume]
        instrument: Instrument symbol (e.g., "ES", "NQ")
        atr_period: Period for ATR calculation
        vol_lookbacks: Lookback periods for volume ratios
        structure_lookback: Lookback for swing point detection

    Returns:
        DataFrame with 42 derived features + metadata columns
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    atr = _compute_atr(df, atr_period)
    atr_safe = atr.replace(0, np.nan)

    # bar_range can be 0 for zero-movement bars (common overnight/low liquidity).
    # Use a safe version for division but don't let it propagate NaN everywhere.
    bar_range = df["high"] - df["low"]
    bar_range_safe = bar_range.replace(0, np.nan)

    features = pd.DataFrame(index=df.index)

    # --- Group 1: Bar Anatomy (8 features) ---
    body = df["close"] - df["open"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    features["bar_range_atr"] = bar_range / atr_safe
    features["bar_body_atr"] = body / atr_safe
    features["bar_upper_wick_atr"] = upper_wick / atr_safe
    features["bar_lower_wick_atr"] = lower_wick / atr_safe
    features["bar_body_pct"] = (body.abs() / bar_range_safe).fillna(0)
    features["bar_upper_wick_pct"] = (upper_wick / bar_range_safe).fillna(0)
    features["bar_lower_wick_pct"] = (lower_wick / bar_range_safe).fillna(0)
    features["bar_direction"] = np.sign(body)

    # --- Group 2: Returns & Momentum (8 features) ---
    features["ret_close_1"] = df["close"].pct_change(1)
    features["ret_close_3"] = df["close"].pct_change(3)
    features["ret_close_5"] = df["close"].pct_change(5)
    features["ret_open_close"] = (df["close"] - df["open"]) / df["open"]
    for lb in [5, 10, 20]:
        features[f"ret_momentum_{lb}"] = features["ret_close_1"].rolling(lb).sum()
    features["ret_acceleration"] = (
        features["ret_momentum_5"] - features["ret_momentum_5"].shift(5)
    )

    # --- Group 3: Volume Dynamics (6 features) ---
    for lb in vol_lookbacks:
        vol_ma = df["volume"].rolling(lb).mean()
        features[f"vol_ratio_{lb}"] = df["volume"] / vol_ma.replace(0, np.nan)
    features["vol_change"] = df["volume"].pct_change(1)
    features["vol_close_position"] = ((df["close"] - df["low"]) / bar_range_safe).fillna(0.5)
    features["vol_delta_proxy"] = (features["vol_close_position"] - 0.5) * df["volume"]

    # --- Group 4: Volatility Measures (6 features) ---
    features["vty_atr_raw"] = atr  # For label generation; can drop before model input
    features["vty_atr_zscore"] = _zscore(atr, 50)
    features["vty_range_ratio_5"] = bar_range / bar_range.rolling(5).mean().replace(0, np.nan)
    features["vty_range_ratio_20"] = bar_range / bar_range.rolling(20).mean().replace(0, np.nan)
    features["vty_atr_of_atr"] = atr.rolling(14).std() / atr.rolling(14).mean().replace(0, np.nan)
    features["vty_realized_10"] = features["ret_close_1"].rolling(10).std()
    features["vty_realized_20"] = features["ret_close_1"].rolling(20).std()

    # --- Group 5: Session-Relative Context (5 features) ---
    session_ids, session_info = _compute_session_features(df)
    features["sess_id"] = session_ids
    features["sess_time_of_day"] = session_info["time_of_day"]
    features["sess_bars_elapsed"] = session_info["bars_elapsed"]
    features["sess_dist_from_open"] = session_info["dist_from_open"] / atr_safe
    features["sess_dist_from_high"] = session_info["dist_from_high"] / atr_safe
    features["sess_dist_from_low"] = session_info["dist_from_low"] / atr_safe

    vwap = _compute_session_vwap(df)
    features["sess_dist_from_vwap"] = (df["close"] - vwap) / atr_safe

    # --- Group 6: Market Structure (9 features) ---
    swing_highs, swing_lows = _detect_swings(df, lookback=structure_lookback)
    structure = _classify_structure(swing_highs, swing_lows, len(df))

    features["str_swing_high_dist"] = (
        (df["close"] - swing_highs.reindex(df.index, method="ffill")) / atr_safe
    )
    features["str_swing_low_dist"] = (
        (df["close"] - swing_lows.reindex(df.index, method="ffill")) / atr_safe
    )
    features["str_structure_state"] = structure

    for lb in [10, 20]:
        rolling_high = df["high"].rolling(lb).max()
        rolling_low = df["low"].rolling(lb).min()
        features[f"str_dist_from_high_{lb}"] = (df["close"] - rolling_high) / atr_safe
        features[f"str_dist_from_low_{lb}"] = (df["close"] - rolling_low) / atr_safe
        rng = (rolling_high - rolling_low).replace(0, np.nan)
        features[f"str_range_position_{lb}"] = ((df["close"] - rolling_low) / rng).fillna(0.5)

    # --- Temporal (for embeddings, not model features) ---
    features["tmp_day_of_week"] = df["datetime"].dt.dayofweek
    features["tmp_hour"] = df["datetime"].dt.hour

    # --- Metadata ---
    features["_datetime"] = df["datetime"]
    features["_instrument"] = instrument
    features["_instrument_id"] = INSTRUMENT_MAP.get(instrument, 0)
    features["_close"] = df["close"]
    features["_volume"] = df["volume"]

    return features


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _compute_session_features(df: pd.DataFrame):
    hours = df["datetime"].dt.hour
    minutes = df["datetime"].dt.minute
    time_decimal = hours + minutes / 60.0

    session_ids = pd.Series(0, index=df.index)
    session_ids[(time_decimal >= 3) & (time_decimal < 8)] = 1
    session_ids[(time_decimal >= 8) & (time_decimal < 12)] = 2
    session_ids[(time_decimal >= 12) & (time_decimal < 16)] = 3

    time_of_day = time_decimal / 24.0
    dates = df["datetime"].dt.date
    session_keys = dates.astype(str) + "_" + session_ids.astype(str)

    session_open = df.groupby(session_keys)["open"].transform("first")
    session_high = df.groupby(session_keys)["high"].transform("cummax")
    session_low = df.groupby(session_keys)["low"].transform("cummin")
    bars_elapsed = df.groupby(session_keys).cumcount()
    max_bars = bars_elapsed.groupby(session_keys).transform("max").replace(0, 1)

    session_info = pd.DataFrame({
        "time_of_day": time_of_day,
        "bars_elapsed": bars_elapsed / max_bars,
        "dist_from_open": df["close"] - session_open,
        "dist_from_high": df["close"] - session_high,
        "dist_from_low": df["close"] - session_low,
    }, index=df.index)

    return session_ids, session_info


def _compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    dates = df["datetime"].dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def _detect_swings(df: pd.DataFrame, lookback: int = 10):
    window = 2 * lookback + 1
    rolling_max = df["high"].rolling(window, center=True).max()
    rolling_min = df["low"].rolling(window, center=True).min()
    swing_highs = df["high"].where(df["high"] == rolling_max)
    swing_lows = df["low"].where(df["low"] == rolling_min)
    return swing_highs, swing_lows


def _classify_structure(swing_highs, swing_lows, length):
    structure = pd.Series(0, index=range(length))
    sh, sl = swing_highs.dropna(), swing_lows.dropna()

    if len(sh) >= 2:
        sh_diff = sh.diff()
        hh = (sh_diff > 0).reindex(range(length), method="ffill").fillna(False).infer_objects(copy=False)
        lh = (sh_diff < 0).reindex(range(length), method="ffill").fillna(False).infer_objects(copy=False)
    else:
        hh = lh = pd.Series(False, index=range(length))

    if len(sl) >= 2:
        sl_diff = sl.diff()
        hl = (sl_diff > 0).reindex(range(length), method="ffill").fillna(False).infer_objects(copy=False)
        ll = (sl_diff < 0).reindex(range(length), method="ffill").fillna(False).infer_objects(copy=False)
    else:
        hl = ll = pd.Series(False, index=range(length))

    structure[hh & hl] = 1
    structure[lh & ll] = -1
    return structure


# =============================================================================
# Feature Column Definitions
# =============================================================================


def get_model_feature_columns() -> list:
    """Returns feature columns that go into the model (excludes metadata)."""
    return [
        "bar_range_atr", "bar_body_atr", "bar_upper_wick_atr", "bar_lower_wick_atr",
        "bar_body_pct", "bar_upper_wick_pct", "bar_lower_wick_pct", "bar_direction",
        "ret_close_1", "ret_close_3", "ret_close_5", "ret_open_close",
        "ret_momentum_5", "ret_momentum_10", "ret_momentum_20", "ret_acceleration",
        "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
        "vol_change", "vol_close_position", "vol_delta_proxy",
        "vty_atr_zscore", "vty_range_ratio_5", "vty_range_ratio_20",
        "vty_atr_of_atr", "vty_realized_10", "vty_realized_20",
        "sess_bars_elapsed", "sess_dist_from_open", "sess_dist_from_high",
        "sess_dist_from_low", "sess_dist_from_vwap",
        "str_swing_high_dist", "str_swing_low_dist", "str_structure_state",
        "str_dist_from_high_10", "str_dist_from_low_10", "str_range_position_10",
        "str_dist_from_high_20", "str_dist_from_low_20", "str_range_position_20",
    ]