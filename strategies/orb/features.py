"""
ORB Feature Generation
======================

20 features in 7 categories that capture the breakout-specific context
the FFM backbone can't see from general market features alone.
"""

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

SESSION_DEFS = {
    'Asia':   {'start_hour': 18, 'start_minute': 0,  'end_hour': 0,  'end_minute': 0,  'duration_min': 360},
    'London': {'start_hour': 3,  'start_minute': 0,  'end_hour': 11, 'end_minute': 30, 'duration_min': 510},
    'NY':     {'start_hour': 9,  'start_minute': 30, 'end_hour': 16, 'end_minute': 0,  'duration_min': 390},
}

MIN_ORB_RANGE = {'ES': 2.0, 'NQ': 10.0, 'RTY': 2.0, 'YM': 20.0, 'GC': 2.0}

ORB_FEATURE_COLS = [
    'orb_range_vs_atr', 'price_vs_orb_high', 'price_vs_orb_low',
    'orb_center_vs_ema20',
    'body_ratio', 'close_beyond_orb', 'bar_range_vs_atr',
    'volume_ratio', 'volume_trend', 'delta_ratio',
    'bars_since_orb_end', 'consecutive_bull', 'consecutive_bear',
    'trend_strength', 'price_vs_ema20', 'volatility_regime', 'gap_vs_orb',
    'session_progress', 'day_of_week_feat', 'session_encoded',
]

NUM_ORB_FEATURES = len(ORB_FEATURE_COLS)

DEFAULT_RR_TARGETS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


# =============================================================================
# Indicator Helpers
# =============================================================================

def compute_atr(high, low, close, length=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def compute_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


# =============================================================================
# Feature Generation
# =============================================================================

def create_orb_features(df, bar_minutes=5, orb_period_min=15):
    """Generate 20 ORB-specific features for hybrid model."""
    df = df.copy()

    df['atr'] = compute_atr(df['high'], df['low'], df['close'], 14).ffill().fillna(1e-6)
    hl2 = (df['high'] + df['low']) / 2.0
    df['ema_hl2_9'] = compute_ema(hl2, 9).ffill()
    df['ema20'] = compute_ema(df['close'], 20).ffill()
    df['ema50'] = compute_ema(df['close'], 50).ffill()
    df['volume_ma'] = df['volume'].rolling(20).mean().ffill().fillna(1)

    df['orb_range_vs_atr'] = df['orb_range'] / (df['atr'] + 1e-6)
    df['price_vs_orb_high'] = np.where(df['orb_range'] > 0, (df['close'] - df['orb_high']) / df['orb_range'], 0)
    df['price_vs_orb_low'] = np.where(df['orb_range'] > 0, (df['close'] - df['orb_low']) / df['orb_range'], 0)
    df['orb_center_vs_ema20'] = np.where(df['atr'] > 0, (df['orb_center'] - df['ema20']) / df['atr'], 0)

    bar_range = df['high'] - df['low']
    df['body_ratio'] = np.where(bar_range > 0, (df['close'] - df['open']).abs() / bar_range, 0)
    df['close_beyond_orb'] = np.where(
        df['orb_range'] > 0,
        np.where(df['close'] > df['orb_high'], (df['close'] - df['orb_high']) / df['orb_range'],
                 np.where(df['close'] < df['orb_low'], (df['orb_low'] - df['close']) / df['orb_range'], 0)), 0)
    df['bar_range_vs_atr'] = bar_range / (df['atr'] + 1e-6)

    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-6)
    vol_5 = df['volume'].rolling(5).mean().ffill()
    vol_20 = df['volume'].rolling(20).mean().ffill()
    df['volume_trend'] = np.where(vol_20 > 0, vol_5 / (vol_20 + 1e-6), 1.0)
    df['_bar_delta'] = np.where(df['close'] > df['open'], df['volume'],
                                np.where(df['close'] < df['open'], -df['volume'], 0))
    cum_delta = df.groupby(['date', 'session_id', 'session_name'])['_bar_delta'].cumsum()
    cum_vol = df.groupby(['date', 'session_id', 'session_name'])['volume'].cumsum()
    df['delta_ratio'] = np.where(cum_vol > 0, cum_delta / cum_vol, 0)
    df.drop(columns=['_bar_delta'], inplace=True)

    df['bars_since_orb_end'] = (df['minutes_since_open'] / bar_minutes - orb_period_min / bar_minutes).clip(lower=0)
    bull_bar = (df['close'] > df['open']).astype(int)
    df['consecutive_bull'] = bull_bar.groupby((bull_bar != bull_bar.shift()).cumsum()).cumsum() * bull_bar
    bear_bar = (df['close'] < df['open']).astype(int)
    df['consecutive_bear'] = bear_bar.groupby((bear_bar != bear_bar.shift()).cumsum()).cumsum() * bear_bar

    df['trend_strength'] = (df['ema20'] - df['ema50']) / (df['atr'] + 1e-6)
    df['price_vs_ema20'] = (df['close'] - df['ema20']) / (df['atr'] + 1e-6)
    atr_50 = df['atr'].rolling(50).mean().ffill()
    df['volatility_regime'] = df['atr'] / (atr_50 + 1e-6)
    prev_close = df.groupby('date')['close'].transform('first')
    df['gap_vs_orb'] = np.where(df['orb_range'] > 0, (prev_close - df['orb_center']) / df['orb_range'], 0)

    dur_map = {s: SESSION_DEFS[s]['duration_min'] for s in SESSION_DEFS}
    sess_dur = df['session_name'].map(dur_map).fillna(390)
    df['session_progress'] = np.clip(df['minutes_since_open'] / sess_dur, 0, 1)
    df['day_of_week_feat'] = df.index.dayofweek.astype(float)

    session_map = {'Asia': 0, 'London': 1, 'NY': 2}
    df['session_encoded'] = df['session_name'].map(session_map).fillna(-1).astype(float)

    return df
