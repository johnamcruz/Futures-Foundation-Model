# ==============================================================================
# CISD+OTE HYBRID FINE-TUNING TRAINER — v7.0
# ==============================================================================
# Changes from v6.0:
#   [Cell 2] CISD features: 28 → 10 (backbone-derivable market context removed)
#            Remaining 10 features are zone geometry + CISD-specific trade mechanics:
#            zone_height_vs_atr, price_vs_zone_top/bot, zone_age_bars,
#            zone_is_bullish, cisd_displacement_str, had_liquidity_sweep,
#            entry_distance_pct, risk_dollars_norm, in_optimal_session
#   [Cell 3] compute_trend_60min removed → htf_1h_structure read from FFM parquet
#            The prepared parquet already contains the causal 1H structure signal;
#            resampling raw CSV and recomputing it was redundant and inconsistent
#   [Cell 3] ATR recomputation removed → vty_atr_raw read from FFM parquet
#            Zone geometry features (zone_height_vs_atr) now use the same ATR
#            the backbone was trained on, ensuring consistent normalisation
#   [Cell 3] Timestamp alignment: dict loop O(n) → pd.reindex O(n log n)
#   [Cell 2] Cache dir v6 → v7; v6 outputs preserved for baseline comparison
#   [Cell 4] Dual checkpoint: saves best val_loss AND best signal_f1 separately;
#            loads best signal_f1 for test evaluation (matches v5.1 behaviour)
#   [Cell 4] Warm start: each fold inherits weights from the previous fold
#            (F1→F2→F3→F4), matching v5.1's progressive training
#   [Cell 4] ONNX device fix: model moved to CPU before export (v5.1 had this bug)
#   [Cell 5] Full evaluation: AvgRR, PF, per-fold breakdown, learning verification
#            vs mechanical baselines — matches v5.1 output format for comparison
# Changes from v5.1 (carried forward from v6.0):
#   BACKBONE_PATH / PREPARED_DIR: 5min_ prefix fix
#   SEQ_LEN: 64 → 96 (matches pretraining context window)
#   candle_types wired end-to-end through backbone
# ==============================================================================


# ==============================================================================
# CELL 1 — SETUP
# ==============================================================================

import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive')

print('📥 Cloning FFM repo...')
os.system('rm -rf /content/Futures-Foundation-Model')
result = subprocess.run(
    ['git', 'clone', 'https://github.com/johnamcruz/Futures-Foundation-Model.git',
     '/content/Futures-Foundation-Model'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f'❌ Clone failed:\n{result.stderr}')
    raise RuntimeError('Git clone failed')
print('✅ Cloned')

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')

try:
    from futures_foundation import FFMConfig, FFMBackbone, get_model_feature_columns
    print(f'✅ FFM installed — {len(get_model_feature_columns())} features')
except (ImportError, ValueError) as e:
    print(f'⚠️  Import failed: {e}')
    print('🔄 Restarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ==============================================================================
# CELL 2 — CONFIGURATION
# ==============================================================================

import torch, shutil

# ── PATHS ──
RAW_DATA_DIR   = '/content/drive/MyDrive/Futures Data/5min'
PREPARED_DIR   = '/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared'        # FIXED v6.0
BACKBONE_PATH  = '/content/drive/MyDrive/AI_Cache/5min_FFM_Checkpoints/best_backbone.pt'  # FIXED v6.0
CISD_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/CISD_OTE_Labels_v7'
OUTPUT_DIR     = '/content/drive/MyDrive/AI_Models/CISD_OTE_Hybrid_v7'

# ── TICKERS ──
DATA_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC']
MICRO_TO_FULL = {
    'MES':  'ES',
    'MNQ':  'NQ',
    'MRTY': 'RTY',
    'MYM':  'YM',
    'MGC':  'GC',
}
TICKERS = DATA_TICKERS

# ── CISD DETECTION PARAMS ──
TREND_FILTER_TF    = '60min'
SWING_PERIOD       = 6
TOLERANCE          = 0.70
EXPIRY_BARS        = 50
LIQUIDITY_LOOKBACK = 10
ZONE_MAX_BARS      = 40
FIB_1              = 0.618
FIB_2              = 0.786
HTF_RANGE_BARS     = 96

# ── DISPLACEMENT FILTER ──
DISP_BODY_RATIO_MIN = 0.50
DISP_CLOSE_STR_MIN  = 0.60

# ── SESSION ──
SESSION_START_HOUR = 7
SESSION_START_MIN  = 0
SESSION_END_HOUR   = 16
SESSION_END_MIN    = 0
OPTIMAL_START_HOUR = 9
OPTIMAL_END_HOUR   = 11

# ── HARD TRAINING FILTERS ──
USE_PD_FILTER             = True
USE_SWEEP_OVERRIDE_PD     = True   # bears with had_sweep=1 bypass P/D gate
USE_TREND_FILTER          = False

# ── SIGNAL CLASS ──
NUM_LABELS = 2

# ── OTHER FILTERS ──
USE_CONFLUENCE     = False
MIN_SCORE          = 0
USE_RISK_FILTER    = False
MAX_RISK_DOLLARS   = 300.0
POINT_VALUES       = {
    'ES': 50.0, 'NQ': 20.0, 'RTY': 10.0, 'YM': 5.0, 'GC': 100.0,
    'MES': 5.0, 'MNQ': 2.0, 'MRTY': 5.0, 'MYM': 0.50, 'MGC': 10.0,
}
USE_CANDLE_FILTER  = False
USE_ENTRY_DISTANCE = False
MAX_CANDLE_MULT    = 3.0
CANDLE_AVG_LEN     = 20

# ── LABELING ──
MIN_SIGNAL_RR = 1.0
RR_TARGETS    = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# ── SPLITS ──
TRAIN_END = '2023-01-01'
VAL_END   = '2024-01-01'

# ── TRAINING ──
SEQ_LEN           = 96       # FIXED v6.0: was 64; matches pretraining context
BATCH_SIZE        = 256
EPOCHS            = 40
LR                = 5e-5
FREEZE_RATIO      = 0.66
RISK_WEIGHT       = 0.1
MISS_PENALTY      = 1.0
FALSE_PENALTY     = 1.0
SIGNAL_OVERSAMPLE = 1.0
SIG_PER_BATCH     = 8
SIGNAL_RATIO      = SIG_PER_BATCH / BATCH_SIZE
FOCAL_GAMMA       = 1.0
FOCAL_SMOOTHING   = 0.10

# ── EARLY STOPPING ──
PATIENCE       = 15
MAX_RATIO      = 2.5
RATIO_PATIENCE = 8

# ── WARM START (fold-to-fold weight transfer) ──
# 'selective': transfer backbone only, cold-start strategy heads each fold
#              so heads re-calibrate to the new fold's regime from scratch.
# 'full':      transfer entire model (use to compare against selective).
WARM_START_MODE       = 'selective'
# LR multiplier for backbone params when warm-starting.
# Heads train at full LR; backbone erodes slowly so prior knowledge is preserved.
# Set to 1.0 to disable layerwise LR and use uniform LR for all params.
BACKBONE_LR_MULTIPLIER = 0.1

# ── MECHANICAL BASELINE WR ──
BASELINE_WR = {
    'ES': 0.275, 'NQ': 0.393, 'RTY': 0.30, 'YM': 0.30, 'GC': 0.355,
    'MES': 0.275, 'MNQ': 0.393, 'MRTY': 0.30, 'MYM': 0.30, 'MGC': 0.355,
}

# ── WALK-FORWARD FOLDS ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── CISD+OTE FEATURES (10) ──
# Only zone geometry and trade mechanics that the backbone cannot derive.
# Market context (trend, wick, volume, session, EMA, HTF alignment, etc.) is
# already captured by the 67-feature backbone sequence — no duplication.
CISD_FEATURE_COLS = [
    'zone_height_vs_atr',    # OTE zone width relative to ATR
    'price_vs_zone_top',     # price distance from zone top (ATR-normalised)
    'price_vs_zone_bot',     # price distance from zone bottom (ATR-normalised)
    'zone_age_bars',         # bars since zone was created
    'zone_is_bullish',       # zone direction: +1 bull / -1 bear
    'cisd_displacement_str', # displacement candle strength (sweep ratio)
    'had_liquidity_sweep',   # liquidity swept before this CISD signal
    'entry_distance_pct',    # depth into OTE zone at entry bar
    'risk_dollars_norm',     # stop loss size (normalised by MAX_RISK_DOLLARS)
    'in_optimal_session',    # 09:00–11:00 NY window flag
]
NUM_CISD_FEATURES = len(CISD_FEATURE_COLS)  # 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n🖥️  Device: {device}')
print(f'🏗️  FFM 256-dim + {NUM_CISD_FEATURES} CISD features | {NUM_LABELS}-class')
print(f'🔧  LR:{LR} | Freeze:{FREEZE_RATIO:.0%} | SEQ_LEN:{SEQ_LEN} | SIG/batch:{SIG_PER_BATCH}')
print(f'🔁  WarmStart:{WARM_START_MODE} | BackboneLR×{BACKBONE_LR_MULTIPLIER}')
print(f'📅  Train<{TRAIN_END} | Val<{VAL_END} | Test≥{VAL_END}')
print(f'📊  CISD on 5min | Trend:{TREND_FILTER_TF} | Swing:{SWING_PERIOD} | Tol:{TOLERANCE}')
print(f'🔴  Displacement: body>={DISP_BODY_RATIO_MIN} + close_str>={DISP_CLOSE_STR_MIN}')
print(f'🔴  Training filters: P/D={USE_PD_FILTER} | Trend={USE_TREND_FILTER}')
print(f'🆕  Bear P/D override: had_sweep=1 → bears bypass P/D gate')
print(f'🆕  Session: {SESSION_START_HOUR}:00–{SESSION_END_HOUR}:00 NY | optimal feature: {OPTIMAL_START_HOUR}–{OPTIMAL_END_HOUR} ET')
print(f'🎯  Tickers: {TICKERS}')
print(f'🎯  Backbone: {BACKBONE_PATH}')

shutil.rmtree(CISD_CACHE_DIR, ignore_errors=True)
os.makedirs(CISD_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# CELL 3 — CISD+OTE LABELING + FEATURE GENERATION (5min)
# ==============================================================================

import os, time
import numpy as np
import pandas as pd
from collections import Counter, deque


def detect_pivots_vectorized(highs, lows, period):
    n = len(highs)
    if n < 2 * period + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    from numpy.lib.stride_tricks import sliding_window_view
    win_h = sliding_window_view(highs, 2 * period + 1)
    win_l = sliding_window_view(lows,  2 * period + 1)
    win_max = win_h.max(axis=1); win_min = win_l.min(axis=1)
    center_h = win_h[:, period]; center_l = win_l[:, period]
    is_ph = (center_h == win_max) & (np.sum(win_h == center_h[:, None], axis=1) == 1)
    is_pl = (center_l == win_min) & (np.sum(win_l == center_l[:, None], axis=1) == 1)
    return np.where(is_ph)[0] + period, np.where(is_pl)[0] + period


def rolling_mean_np(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    out = np.full_like(arr, np.nan)
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(window - 1):
        out[i] = np.mean(arr[:i + 1])
    return out


def detect_5min_cisd_signals(df_5m, tolerance, swing_period, expiry_bars,
                              liquidity_lookback, body_ratio_min=0.60, close_str_min=0.60):
    n      = len(df_5m)
    opens  = df_5m['open'].values; highs = df_5m['high'].values
    lows   = df_5m['low'].values;  closes = df_5m['close'].values
    piv_high_bars, piv_low_bars = detect_pivots_vectorized(highs, lows, swing_period)
    sh_by_conf = {}; sl_by_conf = {}
    for b in piv_high_bars:
        conf = b + swing_period
        if conf < n: sh_by_conf.setdefault(conf, []).append((highs[b], b))
    for b in piv_low_bars:
        conf = b + swing_period
        if conf < n: sl_by_conf.setdefault(conf, []).append((lows[b], b))

    cisd_signal     = np.zeros(n, dtype=np.int8)
    fib_top         = np.full(n, np.nan, dtype=np.float64)
    fib_bot         = np.full(n, np.nan, dtype=np.float64)
    origin_level    = np.full(n, np.nan, dtype=np.float64)
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
            if bar - b >= expiry_bars: continue
            if highs[bar] >= p: last_wicked_high = bar
            else: new_sh.append((p, b))
        active_sh = new_sh

        new_sl = deque()
        for p, b in active_sl:
            if bar - b >= expiry_bars: continue
            if lows[bar] <= p: last_wicked_low = bar
            else: new_sl.append((p, b))
        active_sl = new_sl

        if closes[bar-1] < opens[bar-1] and closes[bar] > opens[bar]:
            bear_pots.append((opens[bar], bar))
        if closes[bar-1] > opens[bar-1] and closes[bar] < opens[bar]:
            bull_pots.append((opens[bar], bar))
        while bear_pots and bar - bear_pots[0][1] >= expiry_bars: bear_pots.popleft()
        while bull_pots and bar - bull_pots[0][1] >= expiry_bars: bull_pots.popleft()

        rng_h = np.max(highs[max(0, bar - HTF_RANGE_BARS):bar])
        rng_l = np.min(lows[max(0,  bar - HTF_RANGE_BARS):bar])
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
                    if ratio > tolerance:
                        full_range = highs[bar] - lows[bar]
                        body = abs(closes[bar] - opens[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (highs[bar] - closes[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 1
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            in_premium_arr[bar]  = 1 if closes[bar] > pd_mid else 0
                            in_discount_arr[bar] = 0 if closes[bar] > pd_mid else 1
                            if (bar - last_wicked_high) <= liquidity_lookback:
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
                    if ratio > tolerance:
                        full_range = highs[bar] - lows[bar]
                        body = abs(closes[bar] - opens[bar])
                        br   = body / full_range if full_range > 0 else 0.0
                        cs   = (closes[bar] - lows[bar]) / full_range if full_range > 0 else 0.0
                        if br >= body_ratio_min and cs >= close_str_min:
                            cisd_signal[bar]     = 2
                            origin_level[bar]    = pot_price
                            disp_strength[bar]   = ratio
                            disp_body_ratio[bar] = br
                            disp_close_str[bar]  = cs
                            in_premium_arr[bar]  = 1 if closes[bar] > pd_mid else 0
                            in_discount_arr[bar] = 0 if closes[bar] > pd_mid else 1
                            if (bar - last_wicked_low) <= liquidity_lookback:
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

    df_5m = df_5m.copy()
    df_5m['cisd_signal']           = cisd_signal
    df_5m['fib_top']               = fib_top
    df_5m['fib_bot']               = fib_bot
    df_5m['origin_level']          = origin_level
    df_5m['had_sweep']             = had_sweep
    df_5m['displacement_strength'] = disp_strength
    df_5m['disp_body_ratio']       = disp_body_ratio
    df_5m['disp_close_str']        = disp_close_str
    df_5m['in_premium']            = in_premium_arr
    df_5m['in_discount']           = in_discount_arr
    return df_5m


def apply_rr_barriers(highs, lows, closes, is_session_end,
                      entry_idx, is_long, entry_price, sl_price, n):
    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0} for rr in RR_TARGETS}
    targets = {rr: entry_price + stop_dist * rr * (1 if is_long else -1) for rr in RR_TARGETS}
    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0} for rr in RR_TARGETS}
    for j in range(entry_idx + 1, n):
        if is_session_end[j]:
            fp = closes[j]
            fr = ((fp - entry_price) / stop_dist) if is_long else ((entry_price - fp) / stop_dist)
            for rr in RR_TARGETS:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'session_end'; results[rr]['realized_rr'] = fr
            break
        if (is_long and lows[j] <= sl_price) or (not is_long and highs[j] >= sl_price):
            for rr in RR_TARGETS:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome'] = 'stopped'; results[rr]['realized_rr'] = -1.0
            break
        for rr in sorted(RR_TARGETS):
            if results[rr]['outcome'] is not None: continue
            if (is_long and highs[j] >= targets[rr]) or (not is_long and lows[j] <= targets[rr]):
                results[rr]['hit'] = True; results[rr]['outcome'] = 'target_hit'
                results[rr]['realized_rr'] = rr
        if all(r['outcome'] is not None for r in results.values()): break
    for rr in RR_TARGETS:
        if results[rr]['outcome'] is None:
            results[rr]['outcome'] = 'data_end'; results[rr]['realized_rr'] = 0.0
    return results


def label_cisd_ote_zones_5min(df_5m, htf_1h_struct, ticker):
    """
    htf_1h_struct: float32 array aligned to df_5m.index, values in {-1, 0, +1}.
                   Sourced directly from the FFM prepared parquet (htf_1h_structure).
                   Replaces the old resample+compute_trend_60min pipeline.
    """
    point_value = POINT_VALUES.get(ticker, 20.0)
    n = len(df_5m)

    o5 = df_5m['open'].values;  h5 = df_5m['high'].values
    l5 = df_5m['low'].values;   c5 = df_5m['close'].values
    cisd_sig    = df_5m['cisd_signal'].values
    ft_arr      = df_5m['fib_top'].values
    fb_arr      = df_5m['fib_bot'].values
    disp_arr    = df_5m['displacement_strength'].values
    sweep_arr   = df_5m['had_sweep'].values
    premium_arr = df_5m['in_premium'].values
    discount_arr= df_5m['in_discount'].values

    hours   = df_5m.index.hour
    minutes = df_5m.index.minute
    time_mins  = hours * 60 + minutes
    sess_start = SESSION_START_HOUR * 60 + SESSION_START_MIN
    sess_end   = SESSION_END_HOUR   * 60 + SESSION_END_MIN
    opt_start  = OPTIMAL_START_HOUR * 60
    opt_end    = OPTIMAL_END_HOUR   * 60
    in_session = (time_mins >= sess_start) & (time_mins < sess_end)

    is_session_end = np.zeros(n, dtype=bool)
    is_session_end[:-1] = in_session[:-1] & ~in_session[1:]

    candle_range = h5 - l5
    candle_sma   = rolling_mean_np(candle_range, CANDLE_AVG_LEN)

    signal_labels = np.zeros(n, dtype=np.int8)
    max_rr_arr    = np.zeros(n, dtype=np.float32)
    sl_dist_arr   = np.full(n, np.nan, dtype=np.float32)

    zone_top     = np.full(n, np.nan, dtype=np.float64)
    zone_bot     = np.full(n, np.nan, dtype=np.float64)
    zone_bull    = np.zeros(n, dtype=np.float32)
    zone_age     = np.zeros(n, dtype=np.float32)
    zone_sweep   = np.zeros(n, dtype=np.float32)
    zone_disp    = np.zeros(n, dtype=np.float32)
    risk_dollars_arr   = np.zeros(n, dtype=np.float32)
    in_optimal_arr     = np.zeros(n, dtype=np.float32)
    entry_distance_arr = np.zeros(n, dtype=np.float32)

    for bar_i in range(n):
        t_m = hours[bar_i] * 60 + minutes[bar_i]
        in_optimal_arr[bar_i] = 1.0 if opt_start <= t_m < opt_end else 0.0

    active_zones = []
    stats = Counter()

    for bar in range(SWING_PERIOD * 3, n):
        cur_trend = int(htf_1h_struct[bar])
        sig = cisd_sig[bar]

        if sig in (1, 2) and not np.isnan(ft_arr[bar]):
            is_bull = (sig == 2)
            skip = False
            if USE_PD_FILTER:
                if is_bull and not discount_arr[bar]:
                    skip = True
                if not is_bull and not premium_arr[bar]:
                    if USE_SWEEP_OVERRIDE_PD and sweep_arr[bar] == 1:
                        skip = False
                    else:
                        skip = True
            if USE_TREND_FILTER and not skip:
                if is_bull  and cur_trend == -1: skip = True
                if not is_bull and cur_trend == 1:  skip = True

            if not skip:
                zone = {
                    'fib_top': ft_arr[bar], 'fib_bot': fb_arr[bar],
                    'created_bar': bar, 'is_bullish': is_bull,
                    'signal_fired': False, 'entered_zone': False,
                    'had_sweep': bool(sweep_arr[bar]),
                    'disp_strength': float(disp_arr[bar]),
                }
                active_zones.insert(0, zone)
                if len(active_zones) > 20: active_zones.pop()
                stats['zones_created'] += 1

        nearest_zone = None; nearest_dist = float('inf')
        for z in active_zones:
            mid = (z['fib_top'] + z['fib_bot']) / 2
            d   = abs(c5[bar] - mid)
            if d < nearest_dist: nearest_dist = d; nearest_zone = z

        if nearest_zone is not None:
            zone_top[bar]  = nearest_zone['fib_top']
            zone_bot[bar]  = nearest_zone['fib_bot']
            zone_bull[bar] = 1.0 if nearest_zone['is_bullish'] else -1.0
            zone_age[bar]  = bar - nearest_zone['created_bar']
            zone_sweep[bar]= 1.0 if nearest_zone['had_sweep'] else 0.0
            zone_disp[bar] = nearest_zone['disp_strength']

        zones_to_remove = []
        for zi, z in enumerate(active_zones):
            invalidated = (z['is_bullish']  and c5[bar] < z['fib_bot']) or \
                          (not z['is_bullish'] and c5[bar] > z['fib_top'])
            if l5[bar] <= z['fib_top'] and h5[bar] >= z['fib_bot']:
                z['entered_zone'] = True
            if bar - z['created_bar'] > ZONE_MAX_BARS:
                zones_to_remove.append(zi); continue

            if z['entered_zone'] and not z['signal_fired'] and not invalidated \
               and bar > z['created_bar']:
                cr  = candle_range[bar]; cb = abs(c5[bar] - o5[bar])
                touched = l5[bar] <= z['fib_top'] and h5[bar] >= z['fib_bot']
                zh  = z['fib_top'] - z['fib_bot']

                entry_dist_pct = 0.0
                if zh > 0:
                    entry_dist_pct = (c5[bar] - z['fib_top']) / zh if z['is_bullish'] \
                                     else (z['fib_bot'] - c5[bar]) / zh

                if z['is_bullish']:
                    sl = z['fib_bot']; risk_pts = c5[bar] - sl
                else:
                    sl = z['fib_top']; risk_pts = sl - c5[bar]
                risk_dollars = risk_pts * point_value if risk_pts > 0 else 0.0

                if z['is_bullish']: bounce = touched and c5[bar] > o5[bar]
                else:               bounce = touched and c5[bar] < o5[bar]

                if bounce and risk_pts > 0:
                    trade_ok = True
                    if USE_CANDLE_FILTER:
                        if candle_sma[bar] > 0 and cr > candle_sma[bar] * MAX_CANDLE_MULT:
                            trade_ok = False
                    if trade_ok:
                        z['signal_fired']       = True
                        risk_dollars_arr[bar]   = risk_dollars / MAX_RISK_DOLLARS
                        entry_distance_arr[bar] = entry_dist_pct
                        is_long = z['is_bullish']
                        res = apply_rr_barriers(h5, l5, c5, is_session_end,
                                                bar, is_long, c5[bar], sl, n)
                        best = max((rr for rr in RR_TARGETS if res[rr]['hit']), default=0.0)
                        max_rr_arr[bar]  = best
                        sl_dist_arr[bar] = risk_pts
                        if best >= MIN_SIGNAL_RR:
                            signal_labels[bar] = 1 if is_long else 2
                            stats['buys' if is_long else 'sells'] += 1
                        stats['trades'] += 1

            if invalidated: zones_to_remove.append(zi)

        for zi in reversed(zones_to_remove):
            if zi < len(active_zones): active_zones.pop(zi)

    meta = {
        'zone_top': zone_top, 'zone_bot': zone_bot, 'zone_bull': zone_bull,
        'zone_age': zone_age, 'zone_sweep': zone_sweep, 'zone_disp': zone_disp,
        'risk_dollars_arr': risk_dollars_arr,
        'in_session_arr': in_optimal_arr, 'entry_distance_arr': entry_distance_arr,
    }
    return signal_labels, max_rr_arr, sl_dist_arr, meta, stats


def create_cisd_features(df_5m, zone_meta, raw_atr):
    """
    Produces 10 CISD-specific features covering zone geometry and trade mechanics.

    Market context (HTF trend, wick ratios, volume, session progress, EMA distance,
    etc.) is already encoded in the 67-feature backbone sequence fed to FFMBackbone.
    Only information that is unique to the CISD zone and trade setup lives here.

    raw_atr: float64 array aligned to df_5m.index (vty_atr_raw from FFM parquet).
    """
    c = df_5m['close'].values
    n = len(df_5m)

    atr_safe = np.where(np.isnan(raw_atr) | (raw_atr <= 0), 1e-6, raw_atr)

    zt = zone_meta['zone_top']; zb = zone_meta['zone_bot']
    zh = np.where(np.isnan(zt) | np.isnan(zb), 0.0, zt - zb)
    zh_safe = np.where(zh > 0, zh, 1e-6)

    features = np.zeros((n, NUM_CISD_FEATURES), dtype=np.float32)
    features[:, 0] = zh / atr_safe                                          # zone_height_vs_atr
    features[:, 1] = np.where(zh > 0, (c - zt) / zh_safe, 0.0)            # price_vs_zone_top
    features[:, 2] = np.where(zh > 0, (c - zb) / zh_safe, 0.0)            # price_vs_zone_bot
    features[:, 3] = np.clip(zone_meta['zone_age'] / ZONE_MAX_BARS, 0, 5)  # zone_age_bars
    features[:, 4] = zone_meta['zone_bull']                                  # zone_is_bullish
    features[:, 5] = np.clip(zone_meta['zone_disp'], 0, 5)                  # cisd_displacement_str
    features[:, 6] = zone_meta['zone_sweep']                                 # had_liquidity_sweep
    features[:, 7] = np.clip(zone_meta['entry_distance_arr'], -2, 5)        # entry_distance_pct
    features[:, 8] = np.clip(zone_meta['risk_dollars_arr'], 0, 5)           # risk_dollars_norm
    features[:, 9] = zone_meta['in_session_arr']                             # in_optimal_session

    return np.nan_to_num(np.clip(features, -10, 10), nan=0.0)


# ── CISD+OTE LABELER (wraps strategy logic for the fine-tuning framework) ──

from futures_foundation.finetune import StrategyLabeler, run_labeling


class CISDOTELabeler(StrategyLabeler):
    """Bridges CISD+OTE strategy logic with the generic fine-tuning framework."""

    @property
    def name(self):
        return 'cisd_ote'

    @property
    def feature_cols(self):
        return CISD_FEATURE_COLS

    def run(self, df_raw, ffm_df, ticker):
        htf_struct = (ffm_df['htf_1h_structure']
                      .reindex(df_raw.index, fill_value=0.0)
                      .values.astype(np.float32))
        raw_atr = (ffm_df['vty_atr_raw']
                   .reindex(df_raw.index, fill_value=np.nan)
                   .values.astype(np.float64))

        df_raw = detect_5min_cisd_signals(
            df_raw, TOLERANCE, SWING_PERIOD, EXPIRY_BARS, LIQUIDITY_LOOKBACK,
            body_ratio_min=DISP_BODY_RATIO_MIN, close_str_min=DISP_CLOSE_STR_MIN)

        sig_arr, rr_arr, sl_arr, zone_meta, stats = label_cisd_ote_zones_5min(
            df_raw, htf_struct, ticker)
        print(f'  Labels: {stats["buys"]}B + {stats["sells"]}S from {stats["trades"]} evaluated')

        cisd_feats = create_cisd_features(df_raw, zone_meta, raw_atr)

        sig_s   = pd.Series(sig_arr.astype(np.int8),   index=df_raw.index)
        rr_s    = pd.Series(rr_arr.astype(np.float32), index=df_raw.index)
        sl_s    = pd.Series(sl_arr.astype(np.float32), index=df_raw.index)
        feats_s = pd.DataFrame(cisd_feats, index=df_raw.index, columns=CISD_FEATURE_COLS)

        aligned_sig   = (sig_s.reindex(ffm_df.index).fillna(0).values > 0).astype(np.int8)
        aligned_rr    = rr_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_sl    = sl_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_feats = feats_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)

        labels_df = pd.DataFrame({
            'signal_label': aligned_sig,
            'max_rr':       aligned_rr,
            'sl_distance':  aligned_sl,
        })
        feats_df = pd.DataFrame(aligned_feats, columns=CISD_FEATURE_COLS)
        return feats_df, labels_df


labeler = CISDOTELabeler()
run_labeling(labeler, TICKERS, RAW_DATA_DIR, PREPARED_DIR, CISD_CACHE_DIR,
             micro_to_full=MICRO_TO_FULL)

# ── Win-rate diagnostic ──
print(f"\n{'='*60}\n  📊 LABEL WIN RATE DIAGNOSTIC\n{'='*60}")
for ticker in TICKERS:
    label_path = os.path.join(CISD_CACHE_DIR, f'{ticker}_strategy_labels.parquet')
    feat_path  = os.path.join(CISD_CACHE_DIR, f'{ticker}_strategy_features.parquet')
    if not os.path.exists(label_path): continue
    ldf   = pd.read_parquet(label_path)
    total = (ldf['signal_label'] > 0).sum()
    if total == 0: print(f'  {ticker}: no labels'); continue
    sig     = ldf[ldf['signal_label'] > 0]
    wr_2r   = (sig['max_rr'] >= 2.0).sum() / total
    wr_3r   = (sig['max_rr'] >= 3.0).sum() / total
    baseline = BASELINE_WR.get(ticker, 0.30)
    status  = '✅' if wr_2r >= baseline - 0.05 else '⚠️ '
    bull_str = ''
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        if len(fdf) == len(ldf):
            zb = fdf.loc[ldf['signal_label'] > 0, 'zone_is_bullish']
            bull_str = f' | Bull:{(zb > 0).sum()} Bear:{(zb < 0).sum()}'
    print(f'  {status} {ticker}: {total} signals{bull_str} | '
          f'WR@2R:{wr_2r*100:.1f}% WR@3R:{wr_3r*100:.1f}% | baseline:{baseline*100:.0f}%')
print(f"{'='*60}")



# ==============================================================================
# CELL 4 — WALK-FORWARD HYBRID FINE-TUNING
# ==============================================================================

import gc, importlib
import futures_foundation
importlib.reload(futures_foundation)
from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune import TrainingConfig, run_walk_forward, export_onnx, validate_setup

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_cfg = TrainingConfig(
    seq_len               = SEQ_LEN,
    batch_size            = BATCH_SIZE,
    sig_per_batch         = SIG_PER_BATCH,
    epochs                = EPOCHS,
    lr                    = LR,
    freeze_ratio          = FREEZE_RATIO,
    risk_weight           = RISK_WEIGHT,
    miss_penalty          = MISS_PENALTY,
    false_penalty         = FALSE_PENALTY,
    focal_gamma           = FOCAL_GAMMA,
    focal_smoothing       = FOCAL_SMOOTHING,
    patience              = PATIENCE,
    max_ratio             = MAX_RATIO,
    ratio_patience        = RATIO_PATIENCE,
    num_labels            = NUM_LABELS,
    warm_start_mode       = WARM_START_MODE,
    backbone_lr_multiplier= BACKBONE_LR_MULTIPLIER,
    baseline_wr           = BASELINE_WR,
)

ffm_config = FFMConfig(
    num_features  = len(get_model_feature_columns()),
    label_smoothing = FOCAL_SMOOTHING,
)

validate_setup(
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = CISD_CACHE_DIR,
    backbone_path         = BACKBONE_PATH,
    strategy_feature_cols = CISD_FEATURE_COLS,
    num_strategy_features = NUM_CISD_FEATURES,
    micro_to_full         = MICRO_TO_FULL,
)

fold_results = run_walk_forward(
    folds                 = FOLDS,
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = CISD_CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    backbone_path         = BACKBONE_PATH,
    ffm_config            = ffm_config,
    training_cfg          = training_cfg,
    num_strategy_features = NUM_CISD_FEATURES,
    strategy_feature_cols = CISD_FEATURE_COLS,
    micro_to_full         = MICRO_TO_FULL,
)

# ── ONNX export (F4 model — production deployment) ──
last_model = fold_results.get('_model')
if last_model is not None:
    print(f'\n{"="*60}\n  ONNX EXPORT\n{"="*60}')
    onnx_path = os.path.join(OUTPUT_DIR, 'cisd_ote_hybrid.onnx')
    try:
        export_onnx(last_model, onnx_path,
                    seq_len=SEQ_LEN,
                    num_ffm_features=len(get_model_feature_columns()),
                    num_strategy_features=NUM_CISD_FEATURES)
    except Exception as e:
        print(f'  ❌ ONNX export failed: {e}')


# ==============================================================================
# CELL 5 — WALK-FORWARD EVALUATION SUMMARY
# ==============================================================================

from futures_foundation.finetune import print_eval_summary
print_eval_summary(fold_results, baseline_wr=BASELINE_WR, output_dir=OUTPUT_DIR)
