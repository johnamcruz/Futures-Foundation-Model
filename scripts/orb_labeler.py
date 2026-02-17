"""
ORB BREAKOUT LABELER
==============================================

Generates per-bar labels aligned with FFM feature parquets:
  - signal_label: 0=HOLD, 1=BUY, 2=SELL (classification target)
  - max_rr: 0, 1, 1.5, 2, 3, 4, 5 (regression target for dynamic TP)
  - sl_distance: stop-loss distance from entry (for risk head)

Usage:
    python scripts/orb_labeler.py \\
        --data-dir /path/to/5min/csvs \\
        --features-dir /path/to/FFM_Prepared \\
        --output-dir /path/to/ORB_Labels \\
        --session NY

Or import and use directly:
    from scripts.orb_labeler import label_instrument
    labels_df = label_instrument(df_5min, ticker="ES", session="NY")
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Optional, Tuple
import warnings
import os
import time

warnings.filterwarnings('ignore')


# =========================================================
# CONFIGURATION
# =========================================================

# Session definitions (ET times) — matching Pine Script
SESSION_DEFS = {
    'Asia':   {'start_hour': 18, 'start_minute': 0,  'end_hour': 0,  'end_minute': 0},
    'London': {'start_hour': 3,  'start_minute': 0,  'end_hour': 11, 'end_minute': 30},
    'NY':     {'start_hour': 9,  'start_minute': 30, 'end_hour': 16, 'end_minute': 0},
}

# ORB detection
ORB_PERIOD_MINUTES = 15   # 15 min = 3 bars on 5-min data
BAR_MINUTES = 5           # 5-min bars (was 3-min in original)
USE_WICKS = False         # False = open/close bodies (matching Pine Script default)

MIN_ORB_RANGE = {
    'ES': 2.0, 'NQ': 10.0, 'RTY': 2.0, 'YM': 20.0, 'GC': 2.0,
}

# Entry
RETESTS_NEEDED = 0        # High sensitivity

# Stop loss
SL_METHOD = "Balanced"    # center of ORB range

# R:R targets
RR_TARGETS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
MIN_SIGNAL_RR = 2.0       # Must hit ≥2R to get BUY/SELL label


# =========================================================
# SESSION DETECTION
# =========================================================

def detect_session_bars(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """
    Tag bars that belong to a given session. Returns df with session columns.
    Assumes df.index is timezone-aware datetime in America/New_York.
    """
    sd = SESSION_DEFS[session_name]
    sh, sm = sd['start_hour'], sd['start_minute']
    eh, em_val = sd['end_hour'], sd['end_minute']

    df = df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    bar_min = df['hour'] * 60 + df['minute']
    start_min = sh * 60 + sm
    end_min = eh * 60 + em_val

    if session_name == 'Asia':
        # Wraps midnight: 18:00-23:59
        df['in_session'] = bar_min >= start_min
        df['is_session_start'] = (df['hour'] == sh) & (df['minute'] == sm)
        df['is_session_end'] = df['in_session'] & (bar_min >= (24 * 60 - BAR_MINUTES))
    else:
        df['in_session'] = (bar_min >= start_min) & (bar_min < end_min)
        df['is_session_start'] = (df['hour'] == sh) & (df['minute'] == sm)
        df['is_session_end'] = df['in_session'] & (bar_min >= end_min - BAR_MINUTES)

    # Only keep session bars
    df = df[df['in_session']].copy()
    if len(df) == 0:
        return df

    # Session ID per date
    df['session_id'] = df.groupby('date')['is_session_start'].cumsum()
    df['minutes_since_open'] = df.groupby(['date', 'session_id']).cumcount() * BAR_MINUTES

    # ORB period
    df['in_orb_period'] = (
        (df['minutes_since_open'] < ORB_PERIOD_MINUTES) &
        (df['session_id'] > 0)
    )
    df['after_orb'] = (
        (df['minutes_since_open'] >= ORB_PERIOD_MINUTES) &
        (df['session_id'] > 0)
    )

    df['session_name'] = session_name
    return df


# =========================================================
# ORB RANGE DETECTION
# =========================================================

def compute_orb_range(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute ORB high/low/range for each session day."""
    min_range = MIN_ORB_RANGE.get(ticker, 2.0)

    if USE_WICKS:
        orb_stats = df[df['in_orb_period']].groupby(['date', 'session_id']).agg(
            orb_high=('high', 'max'),
            orb_low=('low', 'min'),
        )
    else:
        orb_bars = df[df['in_orb_period']].copy()
        orb_bars['bar_high'] = orb_bars[['open', 'close']].max(axis=1)
        orb_bars['bar_low'] = orb_bars[['open', 'close']].min(axis=1)
        orb_stats = orb_bars.groupby(['date', 'session_id']).agg(
            orb_high=('bar_high', 'max'),
            orb_low=('bar_low', 'min'),
        )

    if len(orb_stats) == 0:
        return df

    orb_stats['orb_range'] = orb_stats['orb_high'] - orb_stats['orb_low']
    orb_stats['orb_center'] = (orb_stats['orb_high'] + orb_stats['orb_low']) / 2.0
    orb_stats['orb_valid'] = orb_stats['orb_range'] >= min_range

    df = df.join(orb_stats, on=['date', 'session_id'], how='left')
    for col in ['orb_high', 'orb_low', 'orb_range', 'orb_center', 'orb_valid']:
        df[col] = df.groupby(['date', 'session_id'])[col].ffill()
    df['orb_valid'] = df['orb_valid'].fillna(False)

    return df


# =========================================================
# STOP LOSS
# =========================================================

def calculate_stop_loss(is_bullish: bool, orb_high: float, orb_low: float) -> float:
    """Pine Script SL methods."""
    if SL_METHOD == "Extreme":
        return orb_low if is_bullish else orb_high
    elif SL_METHOD == "Balanced":
        return (orb_high + orb_low) / 2.0
    elif SL_METHOD == "Risky":
        center = (orb_high + orb_low) / 2.0
        return (center + orb_low) / 2.0 if is_bullish else (center + orb_high) / 2.0
    elif SL_METHOD == "Safer":
        center = (orb_high + orb_low) / 2.0
        return (center + orb_high) / 2.0 if is_bullish else (center + orb_low) / 2.0
    return (orb_high + orb_low) / 2.0


# =========================================================
# MULTI-R:R BARRIER (session-bounded)
# =========================================================

def apply_multi_rr_barriers(df: pd.DataFrame, entry_idx: int,
                             is_long: bool, entry_price: float,
                             sl_price: float) -> Dict:
    """Fixed R:R barriers bounded by session end."""
    n = len(df)
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    is_session_end = df['is_session_end'].values
    in_session = df['in_session'].values

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0}
                for rr in RR_TARGETS}

    targets = {}
    for rr in RR_TARGETS:
        targets[rr] = entry_price + (stop_dist * rr * (1 if is_long else -1))

    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0}
               for rr in RR_TARGETS}

    for j in range(entry_idx + 1, n):
        if is_session_end[j] or not in_session[j]:
            final_price = c[j] if in_session[j] else c[j - 1]
            final_rr = ((final_price - entry_price) / stop_dist) if is_long else \
                       ((entry_price - final_price) / stop_dist)
            for rr in RR_TARGETS:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'session_end'
                    results[rr]['realized_rr'] = final_rr
            break

        # Stop loss check first (conservative)
        stopped = False
        if is_long and l[j] <= sl_price:
            stopped = True
        elif not is_long and h[j] >= sl_price:
            stopped = True

        if stopped:
            for rr in RR_TARGETS:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'stopped'
                    results[rr]['realized_rr'] = -1.0
            break

        # Check targets ascending
        for rr in sorted(RR_TARGETS):
            if results[rr]['outcome'] is not None:
                continue
            if is_long and h[j] >= targets[rr]:
                results[rr]['hit'] = True
                results[rr]['outcome'] = 'target_hit'
                results[rr]['realized_rr'] = rr
            elif not is_long and l[j] <= targets[rr]:
                results[rr]['hit'] = True
                results[rr]['outcome'] = 'target_hit'
                results[rr]['realized_rr'] = rr

        if all(results[rr]['outcome'] is not None for rr in RR_TARGETS):
            break

    for rr in RR_TARGETS:
        if results[rr]['outcome'] is None:
            results[rr]['outcome'] = 'data_end'
            results[rr]['realized_rr'] = 0.0

    return results


# =========================================================
# MAIN LABELING — per session per instrument
# =========================================================

def label_session_breakouts(df: pd.DataFrame, ticker: str,
                             session_name: str) -> pd.DataFrame:
    """
    Pine Script state machine for ORB breakout detection + R:R labeling.
    One entry per session day (first valid breakout).

    Returns df with signal_label, max_rr, sl_distance columns.
    """
    df = df.copy()
    n = len(df)

    signal_labels = np.zeros(n, dtype=np.int8)
    max_rr_arr = np.zeros(n, dtype=np.float32)
    sl_distances_arr = np.full(n, np.nan, dtype=np.float32)

    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    orb_high = df['orb_high'].values
    orb_low = df['orb_low'].values
    orb_valid = df['orb_valid'].values
    after_orb = df['after_orb'].values

    session_state = {}
    stats = {'trades': 0, 'buys': 0, 'sells': 0, 'rr_dist': Counter()}

    for i in range(n):
        if not after_orb[i] or not orb_valid[i]:
            continue

        date = df.index[i].date()
        session_id = df['session_id'].iloc[i]
        sk = (date, session_id)

        if sk not in session_state:
            session_state[sk] = {'state': 'waiting', 'bo_idx': None, 'bo_dir': None, 'retests': 0, 'done': False}

        ss = session_state[sk]
        if ss['done']:
            continue

        # WAITING FOR BREAKOUT
        if ss['state'] == 'waiting':
            is_bull = c[i] > orb_high[i]
            is_bear = c[i] < orb_low[i]
            if is_bull or is_bear:
                ss['state'] = 'in_breakout'
                ss['bo_idx'] = i
                ss['bo_dir'] = 'bull' if is_bull else 'bear'
                ss['retests'] = 0
                if RETESTS_NEEDED == 0:
                    ss['state'] = 'entry'

        # IN BREAKOUT (retest tracking)
        elif ss['state'] == 'in_breakout':
            is_bull = ss['bo_dir'] == 'bull'
            if (is_bull and c[i] < orb_high[i]) or (not is_bull and c[i] > orb_low[i]):
                ss['state'] = 'waiting'
                ss['bo_idx'] = None
                ss['bo_dir'] = None
                ss['retests'] = 0
                continue

            if i > ss['bo_idx']:
                if is_bull and c[i] > orb_high[i] and l[i] < orb_high[i]:
                    ss['retests'] += 1
                elif not is_bull and c[i] < orb_low[i] and h[i] > orb_low[i]:
                    ss['retests'] += 1

            if ss['retests'] >= RETESTS_NEEDED:
                ss['state'] = 'entry'

        # ENTRY
        if ss['state'] == 'entry':
            is_bull = ss['bo_dir'] == 'bull'
            entry_price = c[i]
            sl_price = calculate_stop_loss(is_bull, orb_high[i], orb_low[i])

            if (is_bull and sl_price >= entry_price) or (not is_bull and sl_price <= entry_price):
                ss['done'] = True
                continue

            ss['done'] = True
            stop_dist = abs(entry_price - sl_price)

            results = apply_multi_rr_barriers(df, i, is_bull, entry_price, sl_price)

            best_rr = 0.0
            for rr in RR_TARGETS:
                if results[rr]['outcome'] == 'target_hit':
                    best_rr = max(best_rr, rr)

            max_rr_arr[i] = best_rr
            sl_distances_arr[i] = stop_dist

            if best_rr >= MIN_SIGNAL_RR:
                signal_labels[i] = 1 if is_bull else 2
                stats['buys' if is_bull else 'sells'] += 1

            stats['trades'] += 1
            stats['rr_dist'][best_rr] += 1

    df['signal_label'] = signal_labels
    df['max_rr'] = max_rr_arr
    df['sl_distance'] = sl_distances_arr

    buys = stats['buys']
    sells = stats['sells']
    trades = stats['trades']
    print(f"  [{ticker}/{session_name}] {trades} trades → {buys}B + {sells}S signals "
          f"| RR: {dict(sorted(stats['rr_dist'].items()))}")

    return df


# =========================================================
# FULL INSTRUMENT PIPELINE
# =========================================================

def label_instrument(df_raw: pd.DataFrame, ticker: str,
                      sessions: list = None) -> pd.DataFrame:
    """
    Run ORB labeling on raw 5-min OHLCV data for one instrument.

    Args:
        df_raw: DataFrame with columns [datetime, open, high, low, close, volume]
        ticker: Instrument symbol (ES, NQ, etc.)
        sessions: List of sessions to process (default: ['NY'])

    Returns:
        DataFrame indexed by original datetime with columns:
            signal_label (0/1/2), max_rr (float), sl_distance (float)
        Only rows within active sessions are labeled; all others are HOLD.
    """
    sessions = sessions or ['NY']

    # Ensure datetime index in ET
    df = df_raw.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(df.index.tz) != 'America/New_York':
        df.index = df.index.tz_convert('America/New_York')

    df.sort_index(inplace=True)

    # Initialize output aligned to original index
    out = pd.DataFrame(index=df.index)
    out['signal_label'] = 0
    out['max_rr'] = 0.0
    out['sl_distance'] = np.nan

    for sess_name in sessions:
        print(f"  Processing {ticker} / {sess_name}...")
        df_sess = detect_session_bars(df, sess_name)
        if len(df_sess) == 0:
            print(f"    No bars in {sess_name} session")
            continue

        df_sess = compute_orb_range(df_sess, ticker)
        df_sess = label_session_breakouts(df_sess, ticker, sess_name)

        # Merge back — only overwrite where we found signals
        signal_mask = df_sess['signal_label'] > 0
        for idx in df_sess.index[signal_mask]:
            if idx in out.index:
                out.loc[idx, 'signal_label'] = df_sess.loc[idx, 'signal_label']
                out.loc[idx, 'max_rr'] = df_sess.loc[idx, 'max_rr']
                out.loc[idx, 'sl_distance'] = df_sess.loc[idx, 'sl_distance']

        # Also store max_rr for non-signal trades (regression data)
        trade_mask = df_sess['max_rr'] > 0
        for idx in df_sess.index[trade_mask & ~signal_mask]:
            if idx in out.index:
                out.loc[idx, 'max_rr'] = df_sess.loc[idx, 'max_rr']
                out.loc[idx, 'sl_distance'] = df_sess.loc[idx, 'sl_distance']

    return out


# =========================================================
# ALIGN WITH FFM FEATURES
# =========================================================

def align_orb_labels_with_ffm(features_path: str, orb_labels: pd.DataFrame,
                                ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FFM features parquet and align ORB labels by datetime.

    Returns:
        (features_df, labels_df) — same length, aligned by datetime
    """
    features_df = pd.read_parquet(features_path)

    # FFM features have _datetime column
    if '_datetime' in features_df.columns:
        feat_dt = pd.to_datetime(features_df['_datetime'])
        if feat_dt.dt.tz is None:
            feat_dt = feat_dt.dt.tz_localize('UTC').tz_convert('America/New_York')
        features_df['_dt_et'] = feat_dt
    else:
        raise ValueError("FFM features missing _datetime column")

    # ORB labels indexed by datetime (already ET)
    orb_dt = orb_labels.index
    if orb_dt.tz is None:
        orb_dt = orb_dt.tz_localize('America/New_York')

    # Build lookup
    orb_lookup = orb_labels.copy()
    orb_lookup.index = orb_dt

    # Match by datetime
    labels_df = pd.DataFrame(index=features_df.index)
    labels_df['signal_label'] = 0
    labels_df['max_rr'] = 0.0
    labels_df['sl_distance'] = np.nan

    matched = 0
    signals = 0
    for i, dt in enumerate(features_df['_dt_et']):
        if dt in orb_lookup.index:
            row = orb_lookup.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]  # Handle duplicates
            labels_df.iloc[i, labels_df.columns.get_loc('signal_label')] = int(row['signal_label'])
            labels_df.iloc[i, labels_df.columns.get_loc('max_rr')] = float(row['max_rr'])
            if not np.isnan(row['sl_distance']):
                labels_df.iloc[i, labels_df.columns.get_loc('sl_distance')] = float(row['sl_distance'])
            if row['signal_label'] > 0:
                signals += 1
            matched += 1

    print(f"  [{ticker}] Aligned: {matched} matched timestamps, {signals} signals")

    return features_df, labels_df


# =========================================================
# CLI ENTRY POINT
# =========================================================

def main():
    """
    Run ORB labeling on all instruments and save aligned with FFM features.
    """
    import argparse
    parser = argparse.ArgumentParser(description="ORB V2.5 Labeler for 5-min FFM data")
    parser.add_argument("--data-dir", required=True, help="Directory with 5-min OHLCV CSVs")
    parser.add_argument("--features-dir", required=True, help="Directory with FFM feature parquets")
    parser.add_argument("--output-dir", required=True, help="Output directory for ORB labels")
    parser.add_argument("--sessions", nargs="+", default=["NY"], help="Sessions to label")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tickers = ['ES', 'NQ', 'RTY', 'YM', 'GC']

    for ticker in tickers:
        csv_path = os.path.join(args.data_dir, f"{ticker}_5min.csv")
        feat_path = os.path.join(args.features_dir, f"{ticker}_features.parquet")

        if not os.path.exists(csv_path):
            print(f"  Skipping {ticker} — no CSV at {csv_path}")
            continue
        if not os.path.exists(feat_path):
            print(f"  Skipping {ticker} — no features at {feat_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {ticker} — ORB Labeling")
        print(f"{'='*60}")

        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        if 'date' in df_raw.columns and 'datetime' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'date': 'datetime'})

        orb_labels = label_instrument(df_raw, ticker, sessions=args.sessions)

        features_df, labels_df = align_orb_labels_with_ffm(feat_path, orb_labels, ticker)

        out_path = os.path.join(args.output_dir, f"{ticker}_orb_labels.parquet")
        labels_df.to_parquet(out_path, index=False)
        print(f"  ✓ Saved: {out_path}")

        # Summary
        sig_counts = labels_df['signal_label'].value_counts()
        print(f"  Labels: HOLD={sig_counts.get(0, 0)}, BUY={sig_counts.get(1, 0)}, SELL={sig_counts.get(2, 0)}")


if __name__ == '__main__':
    main()