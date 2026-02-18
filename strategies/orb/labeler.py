"""
ORB Labeling (V2.5 State Machine)
==================================

Pine Script-compatible breakout detection with multi-R:R barrier evaluation.
Generates per-bar labels aligned with FFM feature parquets.

Labels:
  - signal_label: 0=HOLD, 1=BUY, 2=SELL
  - max_rr: highest R:R target hit (0, 1, 1.5, 2, 3, 4, 5)
  - sl_distance: stop-loss distance from entry price
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Optional, Tuple

from .features import (
    SESSION_DEFS, ORB_FEATURE_COLS, NUM_ORB_FEATURES,
    detect_session_bars, compute_orb_range, create_orb_features,
)


DEFAULT_RR_TARGETS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


# =============================================================================
# Stop Loss & R:R Barriers
# =============================================================================

def calculate_stop_loss(is_bull, orb_high, orb_low, method="Balanced"):
    """Calculate stop-loss price for ORB trade."""
    if method == "Balanced":
        return (orb_high + orb_low) / 2.0
    elif method == "Extreme":
        return orb_low if is_bull else orb_high
    elif method == "Risky":
        c = (orb_high + orb_low) / 2.0
        return (c + orb_low) / 2.0 if is_bull else (c + orb_high) / 2.0
    elif method == "Safer":
        c = (orb_high + orb_low) / 2.0
        return (c + orb_high) / 2.0 if is_bull else (c + orb_low) / 2.0
    return (orb_high + orb_low) / 2.0


def apply_rr_barriers(df, entry_idx, is_long, entry_price, sl_price,
                      rr_targets=None):
    """Walk forward from entry and check R:R target barriers."""
    rr_targets = rr_targets or DEFAULT_RR_TARGETS
    n = len(df)
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    is_end = df['is_session_end'].values
    in_sess = df['in_session'].values
    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0} for rr in rr_targets}

    targets = {rr: entry_price + stop_dist * rr * (1 if is_long else -1) for rr in rr_targets}
    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0} for rr in rr_targets}

    for j in range(entry_idx + 1, n):
        if is_end[j] or not in_sess[j]:
            fp = c[j] if in_sess[j] else c[j - 1]
            fr = ((fp - entry_price) / stop_dist) if is_long else ((entry_price - fp) / stop_dist)
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'session_end'
                    results[rr]['realized_rr'] = fr
            break
        if (is_long and l[j] <= sl_price) or (not is_long and h[j] >= sl_price):
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'stopped'
                    results[rr]['realized_rr'] = -1.0
            break
        for rr in sorted(rr_targets):
            if results[rr]['outcome'] is not None:
                continue
            if (is_long and h[j] >= targets[rr]) or (not is_long and l[j] <= targets[rr]):
                results[rr]['hit'] = True
                results[rr]['outcome'] = 'target_hit'
                results[rr]['realized_rr'] = rr
        if all(r['outcome'] is not None for r in results.values()):
            break

    for rr in rr_targets:
        if results[rr]['outcome'] is None:
            results[rr]['outcome'] = 'data_end'
            results[rr]['realized_rr'] = 0.0
    return results


# =============================================================================
# Session Labeling
# =============================================================================

def label_session(df, ticker, session_name, sl_method="Balanced",
                  retests_needed=0, min_signal_rr=2.0, rr_targets=None,
                  verbose=True):
    """
    Apply V2.5 state machine labeling to a session DataFrame.

    Returns DataFrame with signal_label, max_rr, sl_distance columns.
    """
    rr_targets = rr_targets or DEFAULT_RR_TARGETS
    df = df.copy()
    n = len(df)
    signal_labels = np.zeros(n, dtype=np.int8)
    max_rr_arr = np.zeros(n, dtype=np.float32)
    sl_dist_arr = np.full(n, np.nan, dtype=np.float32)
    c, h, l = df['close'].values, df['high'].values, df['low'].values
    orb_h, orb_l = df['orb_high'].values, df['orb_low'].values
    orb_v, after = df['orb_valid'].values, df['after_orb'].values
    states = {}
    stats = {'trades': 0, 'buys': 0, 'sells': 0, 'rr': Counter()}

    for i in range(n):
        if not after[i] or not orb_v[i]:
            continue
        sk = (df.index[i].date(), df['session_id'].iloc[i])
        if sk not in states:
            states[sk] = {'st': 'wait', 'bi': None, 'bd': None, 'rt': 0, 'done': False}
        ss = states[sk]
        if ss['done']:
            continue

        if ss['st'] == 'wait':
            bull = c[i] > orb_h[i]
            bear = c[i] < orb_l[i]
            if bull or bear:
                ss['st'] = 'bo'
                ss['bi'] = i
                ss['bd'] = 'bull' if bull else 'bear'
                ss['rt'] = 0
                if retests_needed == 0:
                    ss['st'] = 'entry'
        elif ss['st'] == 'bo':
            ib = ss['bd'] == 'bull'
            if (ib and c[i] < orb_h[i]) or (not ib and c[i] > orb_l[i]):
                ss['st'] = 'wait'
                ss['bi'] = None
                ss['bd'] = None
                ss['rt'] = 0
                continue
            if i > ss['bi']:
                if ib and c[i] > orb_h[i] and l[i] < orb_h[i]:
                    ss['rt'] += 1
                elif not ib and c[i] < orb_l[i] and h[i] > orb_l[i]:
                    ss['rt'] += 1
            if ss['rt'] >= retests_needed:
                ss['st'] = 'entry'

        if ss['st'] == 'entry':
            ib = ss['bd'] == 'bull'
            ep = c[i]
            sl = calculate_stop_loss(ib, orb_h[i], orb_l[i], sl_method)
            if (ib and sl >= ep) or (not ib and sl <= ep):
                ss['done'] = True
                continue
            ss['done'] = True
            sd = abs(ep - sl)
            res = apply_rr_barriers(df, i, ib, ep, sl, rr_targets)
            best = max((rr for rr in rr_targets if res[rr]['hit']), default=0.0)
            max_rr_arr[i] = best
            sl_dist_arr[i] = sd
            if best >= min_signal_rr:
                signal_labels[i] = 1 if ib else 2
                stats['buys' if ib else 'sells'] += 1
            stats['trades'] += 1
            stats['rr'][best] += 1

    df['signal_label'] = signal_labels
    df['max_rr'] = max_rr_arr
    df['sl_distance'] = sl_dist_arr

    if verbose:
        b, s, t = stats['buys'], stats['sells'], stats['trades']
        print(f"    [{ticker}/{session_name}] {t} trades → {b}B + {s}S signals")
    return df


# =============================================================================
# Full Instrument Pipeline
# =============================================================================

def label_instrument(df_raw, ticker, sessions=None, bar_minutes=5,
                     orb_period_min=15, use_wicks=False, sl_method="Balanced",
                     retests_needed=0, min_signal_rr=2.0, rr_targets=None,
                     verbose=True):
    """
    Complete labeling + feature pipeline for one instrument.

    Args:
        df_raw: Raw OHLCV DataFrame with DatetimeIndex (tz-aware, ET).
        ticker: Instrument name (ES, NQ, etc.).
        sessions: List of sessions. Default: ['Asia', 'London', 'NY'].

    Returns:
        List of session DataFrames with labels and ORB features.
    """
    sessions = sessions or ['Asia', 'London', 'NY']
    rr_targets = rr_targets or DEFAULT_RR_TARGETS
    session_results = []

    for sess in sessions:
        df_sess = detect_session_bars(df_raw, sess, bar_minutes, orb_period_min)
        if len(df_sess) == 0:
            continue
        df_sess = compute_orb_range(df_sess, ticker, use_wicks)
        df_sess = create_orb_features(df_sess, bar_minutes, orb_period_min)
        df_sess = label_session(df_sess, ticker, sess, sl_method,
                                retests_needed, min_signal_rr, rr_targets, verbose)
        session_results.append(df_sess)

    return session_results


def align_orb_to_ffm(session_results, features_df):
    """
    Align ORB labels and features to FFM feature parquet datetimes.

    Priority: signal bars > higher-priority session (NY > London > Asia).

    Returns:
        Tuple of (labels_df, orb_features_df) aligned to features_df rows.
    """
    feat_dt = pd.to_datetime(features_df['_datetime'])
    if feat_dt.dt.tz is None:
        feat_dt = feat_dt.dt.tz_localize('UTC').tz_convert('America/New_York')

    n_ffm = len(features_df)
    session_priority = {'NY': 3, 'London': 2, 'Asia': 1}

    # Build lookup: datetime → best session data
    orb_lookup = {}
    for df_sess in session_results:
        sess_name = df_sess['session_name'].iloc[0] if len(df_sess) > 0 else 'NY'
        priority = session_priority.get(sess_name, 0)

        for idx in range(len(df_sess)):
            dt = df_sess.index[idx]
            row = df_sess.iloc[idx]
            sig = int(row.get('signal_label', 0))
            mr = float(row.get('max_rr', 0))
            sld = float(row.get('sl_distance', 0))

            orb_feat_vals = {}
            for col in ORB_FEATURE_COLS:
                val = row.get(col, 0.0)
                orb_feat_vals[col] = float(val) if pd.notna(val) else 0.0

            if dt not in orb_lookup:
                orb_lookup[dt] = (sig, mr, sld, orb_feat_vals, priority)
            else:
                existing = orb_lookup[dt]
                if sig > 0 and existing[0] == 0:
                    orb_lookup[dt] = (sig, mr, sld, orb_feat_vals, priority)
                elif sig == existing[0] and priority > existing[4]:
                    orb_lookup[dt] = (sig, mr, sld, orb_feat_vals, priority)

    # Build aligned arrays
    sig_arr = np.zeros(n_ffm, dtype=np.int8)
    mr_arr = np.zeros(n_ffm, dtype=np.float32)
    sld_arr = np.zeros(n_ffm, dtype=np.float32)
    orb_arr = np.zeros((n_ffm, NUM_ORB_FEATURES), dtype=np.float32)

    matched = 0
    for i in range(len(feat_dt)):
        dt = feat_dt.iloc[i]
        if dt in orb_lookup:
            sig, mr, sld, orb_feats, _ = orb_lookup[dt]
            sig_arr[i] = sig
            mr_arr[i] = np.float32(mr)
            sld_arr[i] = np.float32(sld) if not np.isnan(sld) else np.float32(0.0)
            for j, col in enumerate(ORB_FEATURE_COLS):
                orb_arr[i, j] = np.float32(orb_feats.get(col, 0.0))
            matched += 1

    labels_df = pd.DataFrame({
        'signal_label': sig_arr,
        'max_rr': mr_arr,
        'sl_distance': sld_arr,
    })
    orb_features_df = pd.DataFrame(orb_arr, columns=ORB_FEATURE_COLS)

    return labels_df, orb_features_df
