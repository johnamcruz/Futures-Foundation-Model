"""
CISDOTEStrategyV7 — production inference strategy for the v7.0 CISD+OTE hybrid model.

Drop this file into your algotrader repo and extend BaseStrategy.
The model reads 96 bars of 67-feature FFM context + a 10-feature CISD zone vector,
and returns (signal_logits [2], risk_prediction [1], confidence scalar).

Usage:
    from strategy_cisd_ote_v7 import CISDOTEStrategyV7
    strategy = CISDOTEStrategyV7(onnx_path='path/to/cisd_ote_hybrid.onnx', instrument='ES')
    signal, confidence = strategy.on_bar(df_history)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Optional, Tuple

# ── Assumes your algotrader repo provides this base class ──
# Adjust import path to match your repo structure.
try:
    from base_strategy import BaseStrategy
except ImportError:
    BaseStrategy = object

from futures_foundation import get_model_feature_columns
from futures_foundation.features import derive_features, INSTRUMENT_MAP

# ==============================================================================
# CONSTANTS — must match training exactly
# ==============================================================================

# CISD detection
SWING_PERIOD       = 6
TOLERANCE          = 0.70
EXPIRY_BARS        = 50
LIQUIDITY_LOOKBACK = 10
ZONE_MAX_BARS      = 40
FIB_1              = 0.618
FIB_2              = 0.786
HTF_RANGE_BARS     = 96

# Displacement filter
DISP_BODY_RATIO_MIN = 0.50
DISP_CLOSE_STR_MIN  = 0.60

# Session / timing
OPTIMAL_START_HOUR = 9
OPTIMAL_END_HOUR   = 11

# Risk normalisation
MAX_RISK_DOLLARS = 300.0
POINT_VALUES = {
    'ES': 50.0, 'NQ': 20.0, 'RTY': 10.0, 'YM': 5.0, 'GC': 100.0,
    'MES': 5.0, 'MNQ': 2.0, 'MRTY': 5.0, 'MYM': 0.50, 'MGC': 10.0,
}

# Model geometry
SEQ_LEN           = 96
NUM_FFM_FEATURES  = 67
NUM_CISD_FEATURES = 10

CISD_FEATURE_COLS = [
    'zone_height_vs_atr',
    'price_vs_zone_top',
    'price_vs_zone_bot',
    'zone_age_bars',
    'zone_is_bullish',
    'cisd_displacement_str',
    'had_liquidity_sweep',
    'entry_distance_pct',
    'risk_dollars_norm',
    'in_optimal_session',
]

# Confidence thresholds (match your risk tolerance)
THRESHOLD_CONSERVATIVE = 0.90
THRESHOLD_MODERATE     = 0.80
THRESHOLD_AGGRESSIVE   = 0.70


# ==============================================================================
# CISD DETECTION (stateless — runs on each new bar)
# ==============================================================================

def _detect_pivots_vectorized(highs: np.ndarray, lows: np.ndarray, period: int):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(highs)
    if n < 2 * period + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    win_h = sliding_window_view(highs, 2 * period + 1)
    win_l = sliding_window_view(lows,  2 * period + 1)
    win_max = win_h.max(axis=1); win_min = win_l.min(axis=1)
    center_h = win_h[:, period]; center_l = win_l[:, period]
    is_ph = (center_h == win_max) & (np.sum(win_h == center_h[:, None], axis=1) == 1)
    is_pl = (center_l == win_min) & (np.sum(win_l == center_l[:, None], axis=1) == 1)
    return np.where(is_ph)[0] + period, np.where(is_pl)[0] + period


def detect_cisd_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised CISD displacement detector. Adds columns:
        cisd_signal, fib_top, fib_bot, origin_level, had_sweep,
        displacement_strength, disp_body_ratio, disp_close_str,
        in_premium, in_discount
    """
    n      = len(df)
    opens  = df['open'].values;  highs  = df['high'].values
    lows   = df['low'].values;   closes = df['close'].values

    piv_high_bars, piv_low_bars = _detect_pivots_vectorized(highs, lows, SWING_PERIOD)
    sh_by_conf: dict = {}; sl_by_conf: dict = {}
    for b in piv_high_bars:
        conf = b + SWING_PERIOD
        if conf < n: sh_by_conf.setdefault(conf, []).append((highs[b], b))
    for b in piv_low_bars:
        conf = b + SWING_PERIOD
        if conf < n: sl_by_conf.setdefault(conf, []).append((lows[b], b))

    cisd_signal     = np.zeros(n, dtype=np.int8)
    fib_top         = np.full(n, np.nan)
    fib_bot         = np.full(n, np.nan)
    origin_level    = np.full(n, np.nan)
    had_sweep       = np.zeros(n, dtype=np.int8)
    disp_strength   = np.zeros(n, dtype=np.float32)
    disp_body_ratio = np.zeros(n, dtype=np.float32)
    disp_close_str  = np.zeros(n, dtype=np.float32)
    in_premium_arr  = np.zeros(n, dtype=np.int8)
    in_discount_arr = np.zeros(n, dtype=np.int8)

    active_sh = deque(); active_sl = deque()
    last_wicked_high = -999; last_wicked_low = -999
    bear_pots = deque(); bull_pots = deque()

    for bar in range(1, n):
        if bar in sh_by_conf:
            for p, b in sh_by_conf[bar]: active_sh.append((p, b))
        if bar in sl_by_conf:
            for p, b in sl_by_conf[bar]: active_sl.append((p, b))

        new_sh = deque()
        for p, b in active_sh:
            if bar - b >= EXPIRY_BARS: continue
            if highs[bar] >= p: last_wicked_high = bar
            else: new_sh.append((p, b))
        active_sh = new_sh

        new_sl = deque()
        for p, b in active_sl:
            if bar - b >= EXPIRY_BARS: continue
            if lows[bar] <= p: last_wicked_low = bar
            else: new_sl.append((p, b))
        active_sl = new_sl

        if closes[bar-1] < opens[bar-1] and closes[bar] > opens[bar]:
            bear_pots.append((opens[bar], bar))
        if closes[bar-1] > opens[bar-1] and closes[bar] < opens[bar]:
            bull_pots.append((opens[bar], bar))
        while bear_pots and bar - bear_pots[0][1] >= EXPIRY_BARS: bear_pots.popleft()
        while bull_pots and bar - bull_pots[0][1] >= EXPIRY_BARS: bull_pots.popleft()

        rng_h  = np.max(highs[max(0, bar - HTF_RANGE_BARS):bar])
        rng_l  = np.min(lows[max(0,  bar - HTF_RANGE_BARS):bar])
        pd_mid = (rng_h + rng_l) / 2.0

        # ── Bearish CISD ──
        while bear_pots:
            pot_price, pot_bar = bear_pots[0]
            if closes[bar] < pot_price:
                highest_c = closes[pot_bar:bar + 1].max()
                top_level = 0.0; idx = pot_bar + 1
                while idx < bar and closes[idx] < opens[idx]:
                    top_level = opens[idx]; idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > TOLERANCE:
                        full_range = highs[bar] - lows[bar]
                        body = abs(closes[bar] - opens[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (highs[bar] - closes[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
                            cisd_signal[bar]     = 1
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            in_premium_arr[bar]  = 1 if closes[bar] > pd_mid else 0
                            in_discount_arr[bar] = 0 if closes[bar] > pd_mid else 1
                            if (bar - last_wicked_high) <= LIQUIDITY_LOOKBACK:
                                had_sweep[bar] = 1
                            h_max = highs[pot_bar:bar + 1].max()
                            diff  = h_max - lows[bar]
                            fib_top[bar] = max(h_max - diff * FIB_1, h_max - diff * FIB_2)
                            fib_bot[bar] = min(h_max - diff * FIB_1, h_max - diff * FIB_2)
                            bear_pots.clear(); break
                        else: bear_pots.popleft(); continue
                    else: bear_pots.popleft(); continue
                else: bear_pots.popleft(); continue
            else: break

        # ── Bullish CISD ──
        while bull_pots:
            pot_price, pot_bar = bull_pots[0]
            if closes[bar] > pot_price:
                lowest_c     = closes[pot_bar:bar + 1].min()
                bottom_level = 0.0; idx = pot_bar + 1
                while idx < bar and closes[idx] > opens[idx]:
                    bottom_level = opens[idx]; idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > TOLERANCE:
                        full_range = highs[bar] - lows[bar]
                        body = abs(closes[bar] - opens[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (closes[bar] - lows[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
                            cisd_signal[bar]     = 2
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            in_premium_arr[bar]  = 1 if closes[bar] > pd_mid else 0
                            in_discount_arr[bar] = 0 if closes[bar] > pd_mid else 1
                            if (bar - last_wicked_low) <= LIQUIDITY_LOOKBACK:
                                had_sweep[bar] = 1
                            l_min = lows[pot_bar:bar + 1].min()
                            diff  = highs[bar] - l_min
                            fib_top[bar] = max(l_min + diff * FIB_1, l_min + diff * FIB_2)
                            fib_bot[bar] = min(l_min + diff * FIB_1, l_min + diff * FIB_2)
                            bull_pots.clear(); break
                        else: bull_pots.popleft(); continue
                    else: bull_pots.popleft(); continue
                else: bull_pots.popleft(); continue
            else: break

    df = df.copy()
    df['cisd_signal']           = cisd_signal
    df['fib_top']               = fib_top
    df['fib_bot']               = fib_bot
    df['origin_level']          = origin_level
    df['had_sweep']             = had_sweep
    df['displacement_strength'] = disp_strength
    df['disp_body_ratio']       = disp_body_ratio
    df['disp_close_str']        = disp_close_str
    df['in_premium']            = in_premium_arr
    df['in_discount']           = in_discount_arr
    return df


# ==============================================================================
# LIVE ZONE TRACKER
# ==============================================================================

class ZoneTracker:
    """Maintains active OTE zones across bar updates."""

    def __init__(self):
        self._zones: list = []

    def update(self, bar_idx: int, cisd_signal: int, fib_top: float, fib_bot: float,
               disp_str: float, had_sweep: bool,
               close: float, low: float, high: float) -> None:
        if cisd_signal in (1, 2) and not np.isnan(fib_top):
            is_bull = (cisd_signal == 2)
            self._zones.insert(0, {
                'fib_top':       fib_top,
                'fib_bot':       fib_bot,
                'created_bar':   bar_idx,
                'is_bullish':    is_bull,
                'had_sweep':     had_sweep,
                'disp_strength': disp_str,
                'entered_zone':  False,
                'signal_fired':  False,
            })
            if len(self._zones) > 20:
                self._zones.pop()

        to_remove = []
        for i, z in enumerate(self._zones):
            if bar_idx - z['created_bar'] > ZONE_MAX_BARS:
                to_remove.append(i); continue
            if low <= z['fib_top'] and high >= z['fib_bot']:
                z['entered_zone'] = True
            invalidated = (z['is_bullish']  and close < z['fib_bot']) or \
                          (not z['is_bullish'] and close > z['fib_top'])
            if invalidated:
                to_remove.append(i)
        for i in reversed(to_remove):
            if i < len(self._zones):
                self._zones.pop(i)

    def nearest_zone_for_close(self, close: float) -> Optional[dict]:
        if not self._zones:
            return None
        best, best_d = None, float('inf')
        for z in self._zones:
            mid = (z['fib_top'] + z['fib_bot']) / 2.0
            d = abs(close - mid)
            if d < best_d:
                best, best_d = z, d
        return best


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

class CISDOTEStrategyV7(BaseStrategy):
    """
    CISD+OTE v7.0 inference strategy.

    Handles FFM feature derivation, CISD zone detection, and ONNX inference.
    Returns (direction, confidence) when a signal fires, else (None, None).
    Direction: +1 = BUY, -1 = SELL.
    """

    def __init__(self, onnx_path: str, instrument: str):
        import onnxruntime as ort
        self._session    = ort.InferenceSession(onnx_path)
        self._instrument = instrument.upper()
        self._inst_id    = INSTRUMENT_MAP.get(self._instrument, 0)
        self._point_val  = POINT_VALUES.get(self._instrument, 20.0)
        self._tracker    = ZoneTracker()
        self._bar_idx    = 0

    # ── BaseStrategy interface ────────────────────────────────────────────────

    def get_sequence_length(self) -> int:
        return SEQ_LEN

    def get_warmup_length(self) -> int:
        return 300

    def get_feature_columns(self) -> List[str]:
        return get_model_feature_columns()

    def is_trading_allowed(self, timestamp) -> bool:
        return True

    # ── Main entry point ─────────────────────────────────────────────────────

    def on_bar(self, df_ohlcv: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
        """
        Process one new closed bar.

        Args:
            df_ohlcv: DataFrame with columns [datetime, open, high, low, close, volume].
                      Must contain at least get_warmup_length() rows.

        Returns:
            (direction, confidence) if a signal fires, else (None, None).
        """
        if len(df_ohlcv) < self.get_warmup_length():
            return None, None

        ffm_df = self._compute_ffm_features(df_ohlcv)
        df_with_cisd = detect_cisd_signals(df_ohlcv)

        bar = len(df_ohlcv) - 1
        c   = df_ohlcv['close'].values[-1]
        h   = df_ohlcv['high'].values[-1]
        l   = df_ohlcv['low'].values[-1]
        o   = df_ohlcv['open'].values[-1]

        sig = int(df_with_cisd['cisd_signal'].values[-1])
        ft  = float(df_with_cisd['fib_top'].values[-1])
        fb  = float(df_with_cisd['fib_bot'].values[-1])
        ds  = float(df_with_cisd['displacement_strength'].values[-1])
        hs  = bool(df_with_cisd['had_sweep'].values[-1])

        self._tracker.update(bar, sig, ft, fb, ds, hs, c, l, h)
        zone = self._tracker.nearest_zone_for_close(c)
        if zone is None:
            return None, None

        in_zone = l <= zone['fib_top'] and h >= zone['fib_bot']
        bounce  = in_zone and (c > o if zone['is_bullish'] else c < o)

        if not bounce or zone['signal_fired']:
            return None, None

        sl_price = zone['fib_bot'] if zone['is_bullish'] else zone['fib_top']
        risk_pts = (c - sl_price) if zone['is_bullish'] else (sl_price - c)
        if risk_pts <= 0:
            return None, None

        zone['signal_fired'] = True

        cisd_vec = self._build_cisd_feature_vector(
            close=c, zone=zone, current_bar=bar,
            risk_pts=risk_pts,
            timestamp=df_ohlcv['datetime'].values[-1],
            atr=self._get_atr(ffm_df),
        )

        return self.predict(ffm_df, cisd_vec)

    # ── FFM feature derivation ────────────────────────────────────────────────

    def _compute_ffm_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df_input.index, pd.DatetimeIndex):
            df = df_input.reset_index()
            if 'datetime' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'datetime'})
        else:
            df = df_input.copy()
        return derive_features(df, self._instrument)

    def _get_atr(self, ffm_df: pd.DataFrame) -> float:
        if 'vty_atr_raw' in ffm_df.columns:
            v = ffm_df['vty_atr_raw'].values[-1]
            return float(v) if not np.isnan(v) else 1e-6
        return 1e-6

    # ── CISD feature vector ───────────────────────────────────────────────────

    def _build_cisd_feature_vector(
        self, close: float, zone: dict, current_bar: int,
        risk_pts: float, timestamp, atr: float,
    ) -> np.ndarray:
        atr_safe = max(atr, 1e-6)
        zh = zone['fib_top'] - zone['fib_bot']
        zh_safe = max(zh, 1e-6)

        entry_dist = (close - zone['fib_top']) / zh_safe if zone['is_bullish'] \
                     else (zone['fib_bot'] - close) / zh_safe

        ts = pd.Timestamp(timestamp)
        in_optimal = 1.0 if OPTIMAL_START_HOUR <= ts.hour < OPTIMAL_END_HOUR else 0.0

        zone_age  = np.clip((current_bar - zone['created_bar']) / ZONE_MAX_BARS, 0.0, 5.0)
        risk_norm = np.clip(risk_pts * self._point_val / MAX_RISK_DOLLARS, 0.0, 5.0)

        vec = np.array([
            np.clip(zh / atr_safe, 0.0, 10.0),                         # zone_height_vs_atr
            np.clip((close - zone['fib_top']) / zh_safe, -10.0, 10.0), # price_vs_zone_top
            np.clip((close - zone['fib_bot']) / zh_safe, -10.0, 10.0), # price_vs_zone_bot
            zone_age,                                                    # zone_age_bars
            1.0 if zone['is_bullish'] else -1.0,                        # zone_is_bullish
            np.clip(zone['disp_strength'], 0.0, 5.0),                   # cisd_displacement_str
            1.0 if zone['had_sweep'] else 0.0,                          # had_liquidity_sweep
            np.clip(entry_dist, -2.0, 5.0),                             # entry_distance_pct
            risk_norm,                                                   # risk_dollars_norm
            in_optimal,                                                  # in_optimal_session
        ], dtype=np.float32)

        return np.nan_to_num(vec, nan=0.0)

    # ── ONNX inference ────────────────────────────────────────────────────────

    def predict(
        self, ffm_df: pd.DataFrame, cisd_vec: np.ndarray,
    ) -> Tuple[Optional[int], float]:
        feat_cols = get_model_feature_columns()
        seq = ffm_df[feat_cols].values[-SEQ_LEN:].astype(np.float32)
        if len(seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq), NUM_FFM_FEATURES), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        features_in = seq[np.newaxis]

        def _get_seq(col: str, dtype, default=0) -> np.ndarray:
            if col in ffm_df.columns:
                vals = ffm_df[col].values[-SEQ_LEN:]
                if len(vals) < SEQ_LEN:
                    vals = np.concatenate([np.full(SEQ_LEN - len(vals), default), vals])
                return vals[np.newaxis].astype(dtype)
            return np.zeros((1, SEQ_LEN), dtype=dtype)

        inputs = {
            'features':          features_in,
            'strategy_features': cisd_vec[np.newaxis],
            'candle_types':      _get_seq('candle_type',      np.int64),
            'time_of_day':       _get_seq('sess_time_of_day', np.float32),
            'day_of_week':       _get_seq('tmp_day_of_week',  np.int64),
            'instrument_ids':    np.array([self._inst_id], dtype=np.int64),
            'session_ids':       _get_seq('sess_id',          np.int64),
        }

        logits, _risk, conf_arr = self._session.run(
            ['signal_logits', 'risk_predictions', 'confidence'], inputs
        )
        confidence  = float(conf_arr[0])
        signal_prob = float(_softmax(logits[0])[1])

        if signal_prob < THRESHOLD_AGGRESSIVE:
            return None, confidence

        direction = +1 if cisd_vec[4] > 0 else -1
        return direction, confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
