# ==============================================================================
# CISD+OTE HYBRID FINE-TUNING TRAINER — v6.0
# ==============================================================================
# Changes from v5.1:
#   [Cell 2] BACKBONE_PATH: FFM_Checkpoints/ → 5min_FFM_Checkpoints/
#            Aligns with the 5min pretraining pipeline output
#   [Cell 2] PREPARED_DIR: FFM_Prepared → 5min_FFM_Prepared
#            Same fix — FFM feature parquets now live in 5min_FFM_Prepared
#   [Cell 2] SEQ_LEN: 64 → 96
#            Matches pretraining context window; gives 2 extra hours of prior
#            session structure for each CISD signal evaluation
#   [Cell 4] candle_types wired through backbone end-to-end:
#            HybridCISDDataset: extracts candle_type from feature parquet
#            HybridCISDModel.forward(): accepts + passes candle_types to FFMBackbone
#            train_one_epoch / evaluate: batch["candle_types"] passed to model
#            Previously candle_types was silently dropped — backbone was pretrained
#            with candle embeddings but fine-tuning never used them
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
CISD_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/CISD_OTE_Labels_v6'
OUTPUT_DIR     = '/content/drive/MyDrive/AI_Models/CISD_OTE_Hybrid_v6'

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

# ── CISD+OTE FEATURES (28) ──
CISD_FEATURE_COLS = [
    'zone_height_vs_atr', 'price_vs_zone_top', 'price_vs_zone_bot',
    'zone_age_bars', 'zone_is_bullish', 'cisd_displacement_strength',
    'had_liquidity_sweep', 'htf_trend_direction', 'trend_alignment',
    'rejection_wick_ratio', 'close_position', 'volume_trend',
    'cumulative_delta_ratio', 'price_vs_ema20', 'gap_from_prior_close',
    'session_progress', 'day_of_week_feat', 'confluence_score',
    'risk_dollars_norm', 'in_optimal_session', 'entry_distance_pct',
    'ffm_sess_dist_from_vwap', 'ffm_str_structure_state', 'ffm_ret_acceleration',
    'ffm_vty_atr_of_atr', 'ffm_sess_dist_from_open', 'ffm_ret_momentum_10',
    'ffm_vol_delta_proxy',
]
NUM_CISD_FEATURES = len(CISD_FEATURE_COLS)  # 28

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n🖥️  Device: {device}')
print(f'🏗️  FFM 256-dim + {NUM_CISD_FEATURES} CISD features | {NUM_LABELS}-class')
print(f'🔧  LR:{LR} | Freeze:{FREEZE_RATIO:.0%} | SEQ_LEN:{SEQ_LEN} | SIG/batch:{SIG_PER_BATCH}')
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


def resample_ohlcv(df, tf):
    return df.resample(tf, closed='right', label='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


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


def compute_atr_np(high, low, close, period=14):
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    alpha = 1.0 / period
    seed  = float(np.mean(tr[:period]))
    atr   = np.empty(len(tr), dtype=np.float64)
    atr[:period] = seed
    atr[period]  = seed * (1 - alpha) + tr[period] * alpha
    if len(tr) > period + 1:
        rest = pd.Series(np.concatenate([[atr[period]], tr[period + 1:]])).ewm(
            alpha=alpha, adjust=False).mean().values
        atr[period:] = rest
    return atr


def compute_ema_np(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def rolling_mean_np(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    out = np.full_like(arr, np.nan)
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(window - 1):
        out[i] = np.mean(arr[:i + 1])
    return out


def compute_trend_60min(df_60m, swing_period=5, structure_lookback=3):
    highs = df_60m['high'].values; lows = df_60m['low'].values; n = len(df_60m)
    piv_h, piv_l = detect_pivots_vectorized(highs, lows, swing_period)
    trend = np.zeros(n, dtype=np.int8)
    recent_h = deque(maxlen=structure_lookback); recent_l = deque(maxlen=structure_lookback)
    sh_conf = {}; sl_conf = {}
    for b in piv_h:
        c = b + swing_period
        if c < n: sh_conf.setdefault(c, []).append(highs[b])
    for b in piv_l:
        c = b + swing_period
        if c < n: sl_conf.setdefault(c, []).append(lows[b])
    for bar in range(n):
        if bar in sh_conf:
            for p in sh_conf[bar]: recent_h.appendleft(p)
        if bar in sl_conf:
            for p in sl_conf[bar]: recent_l.appendleft(p)
        if len(recent_h) >= 2 and len(recent_l) >= 2:
            hh = recent_h[0] > recent_h[1]; hl = recent_l[0] > recent_l[1]
            ll = recent_l[0] < recent_l[1]; lh = recent_h[0] < recent_h[1]
            if hh and hl:    trend[bar] = 1
            elif ll and lh:  trend[bar] = -1
            elif hh or hl:   trend[bar] = 1
            elif ll or lh:   trend[bar] = -1
        elif bar > 0:
            trend[bar] = trend[bar - 1]
    df_60m = df_60m.copy(); df_60m['trend'] = trend
    return df_60m


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


def label_cisd_ote_zones_5min(df_5m, df_60m, ticker):
    point_value = POINT_VALUES.get(ticker, 20.0)
    n = len(df_5m)

    trend_times     = df_60m.index.values
    ltf_times       = df_5m.index.values
    trend_idx_align = np.clip(np.searchsorted(trend_times, ltf_times, side='right') - 1,
                               0, len(df_60m) - 1)
    trend_vals = df_60m['trend'].values

    o5 = df_5m['open'].values;  h5 = df_5m['high'].values
    l5 = df_5m['low'].values;   c5 = df_5m['close'].values
    v5 = df_5m['volume'].values
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

    atr          = compute_atr_np(h5, l5, c5, 14)
    vol_sma      = rolling_mean_np(v5.astype(np.float64), 20)
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
    zone_trend   = np.zeros(n, dtype=np.float32)
    zone_aligned = np.zeros(n, dtype=np.float32)
    confluence_scores  = np.zeros(n, dtype=np.float32)
    risk_dollars_arr   = np.zeros(n, dtype=np.float32)
    in_optimal_arr     = np.zeros(n, dtype=np.float32)
    entry_distance_arr = np.zeros(n, dtype=np.float32)

    for bar_i in range(n):
        t_m = hours[bar_i] * 60 + minutes[bar_i]
        in_optimal_arr[bar_i] = 1.0 if opt_start <= t_m < opt_end else 0.0

    active_zones = []
    stats = Counter()

    for bar in range(SWING_PERIOD * 3, n):
        ti = trend_idx_align[bar]; cur_trend = trend_vals[ti]
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
            zone_top[bar]    = nearest_zone['fib_top']
            zone_bot[bar]    = nearest_zone['fib_bot']
            zone_bull[bar]   = 1.0 if nearest_zone['is_bullish'] else -1.0
            zone_age[bar]    = bar - nearest_zone['created_bar']
            zone_sweep[bar]  = 1.0 if nearest_zone['had_sweep'] else 0.0
            zone_disp[bar]   = nearest_zone['disp_strength']
            zone_trend[bar]  = float(cur_trend)
            is_aligned = (nearest_zone['is_bullish'] and cur_trend >= 0) or \
                         (not nearest_zone['is_bullish'] and cur_trend <= 0)
            zone_aligned[bar] = 1.0 if is_aligned else 0.0

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
                cr   = candle_range[bar]; cb = abs(c5[bar] - o5[bar])
                upper_wick = h5[bar] - max(o5[bar], c5[bar])
                lower_wick = min(o5[bar], c5[bar]) - l5[bar]
                touched    = l5[bar] <= z['fib_top'] and h5[bar] >= z['fib_bot']
                closed_in  = z['fib_bot'] <= c5[bar] <= z['fib_top']
                zh = z['fib_top'] - z['fib_bot']

                score = 0
                if z['had_sweep']:       score += 2
                if z['is_bullish']:
                    if c5[bar] > o5[bar]: score += 1
                    if cb > 0 and lower_wick >= cb * 0.3: score += 1
                    if c5[bar] >= (h5[bar] + l5[bar]) / 2: score += 1
                else:
                    if c5[bar] < o5[bar]: score += 1
                    if cb > 0 and upper_wick >= cb * 0.3: score += 1
                    if c5[bar] <= (h5[bar] + l5[bar]) / 2: score += 1
                if cr > 0 and (cb / cr) >= 0.5: score += 1
                if vol_sma[bar] > 0 and v5[bar] > vol_sma[bar] * 1.2: score += 1
                if closed_in: score += 1
                score += 1

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
                        confluence_scores[bar]  = score
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
        'zone_trend': zone_trend, 'zone_aligned': zone_aligned,
        'confluence_scores': confluence_scores, 'risk_dollars_arr': risk_dollars_arr,
        'in_session_arr': in_optimal_arr, 'entry_distance_arr': entry_distance_arr,
    }
    return signal_labels, max_rr_arr, sl_dist_arr, meta, stats


def create_cisd_features(df_5m, zone_meta, ffm_features=None):
    o = df_5m['open'].values;  h = df_5m['high'].values
    l = df_5m['low'].values;   c = df_5m['close'].values
    v = df_5m['volume'].values.astype(np.float64)
    n = len(df_5m)
    atr      = compute_atr_np(h, l, c, 14)
    ema20    = compute_ema_np(c, 20)
    atr_safe = np.where(atr > 0, atr, 1e-6)
    zt = zone_meta['zone_top']; zb = zone_meta['zone_bot']
    zh = np.where(np.isnan(zt) | np.isnan(zb), 0, zt - zb)
    zh_safe = np.where(zh > 0, zh, 1e-6)
    features = np.zeros((n, NUM_CISD_FEATURES), dtype=np.float32)
    features[:, 0]  = zh / atr_safe
    features[:, 1]  = np.where(zh > 0, (c - zt) / zh_safe, 0)
    features[:, 2]  = np.where(zh > 0, (c - zb) / zh_safe, 0)
    features[:, 3]  = np.clip(zone_meta['zone_age'] / ZONE_MAX_BARS, 0, 5)
    features[:, 4]  = zone_meta['zone_bull']
    features[:, 5]  = np.clip(zone_meta['zone_disp'], 0, 5)
    features[:, 6]  = zone_meta['zone_sweep']
    features[:, 7]  = zone_meta['zone_trend']
    features[:, 8]  = zone_meta['zone_aligned']
    cr       = h - l; cr_safe = np.where(cr > 0, cr, 1e-6)
    body     = np.abs(c - o)
    upper_wick = h - np.maximum(o, c); lower_wick = np.minimum(o, c) - l
    bull_mask  = zone_meta['zone_bull'] > 0
    wick       = np.where(bull_mask, lower_wick, upper_wick)
    body_safe  = np.where(body > 0, body, 1e-6)
    features[:, 9]  = wick / body_safe
    features[:, 10] = np.where(cr > 0, (c - l) / cr_safe, 0.5)
    vol_sma20    = rolling_mean_np(v, 20); vol_sma5_arr = rolling_mean_np(v, 5)
    vol5_safe    = np.where(vol_sma5_arr > 0, vol_sma5_arr, 1e-6)
    vol_safe     = np.where(vol_sma20    > 0, vol_sma20,    1e-6)
    features[:, 11] = vol5_safe / vol_safe
    bar_delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    _df_tmp = pd.DataFrame({'delta': bar_delta, 'vol': v, 'date': df_5m.index.date})
    cum_delta = _df_tmp.groupby('date')['delta'].cumsum().values
    cum_vol   = _df_tmp.groupby('date')['vol'].cumsum().values
    cv_safe   = np.where(cum_vol > 0, cum_vol, 1e-6)
    features[:, 12] = cum_delta / cv_safe
    features[:, 13] = (c - ema20) / atr_safe
    prev_close = np.roll(c, 1); prev_close[0] = c[0]
    features[:, 14] = (c - prev_close) / atr_safe
    hours_arr = df_5m.index.hour; mins_arr = df_5m.index.minute
    sess_mins = (hours_arr * 60 + mins_arr - 9 * 60).astype(np.float64)
    features[:, 15] = np.clip(sess_mins / 420.0, 0, 1)
    features[:, 16] = df_5m.index.dayofweek.values.astype(np.float32)
    features[:, 17] = zone_meta['confluence_scores'] / 10.0
    features[:, 18] = np.clip(zone_meta['risk_dollars_arr'], 0, 5)
    features[:, 19] = zone_meta['in_session_arr']
    features[:, 20] = np.clip(zone_meta['entry_distance_arr'], -2, 5)

    def get_ffm(col, scale=1.0, clip=10.0):
        if ffm_features is not None and col in ffm_features:
            arr = np.array(ffm_features[col], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
            return np.clip(arr * scale, -clip, clip)
        return np.zeros(n, dtype=np.float32)

    features[:, 21] = get_ffm('sess_dist_from_vwap', clip=10.0)
    features[:, 22] = get_ffm('str_structure_state',  clip=5.0)
    features[:, 23] = get_ffm('ret_acceleration',     scale=100.0)
    features[:, 24] = get_ffm('vty_atr_of_atr',       clip=5.0)
    features[:, 25] = get_ffm('sess_dist_from_open',  clip=10.0)
    features[:, 26] = get_ffm('ret_momentum_10',      scale=100.0)
    features[:, 27] = get_ffm('vol_delta_proxy',      scale=0.001)
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(features, -10, 10)


# ── MAIN LABELING PIPELINE ──

print(f"\n{'='*60}")
print(f'  STEP 3: 5min CISD+OTE LABELS + FEATURES')
print(f'  Session: {SESSION_START_HOUR}:00–{SESSION_END_HOUR}:00 NY | optimal feature: {OPTIMAL_START_HOUR}–{OPTIMAL_END_HOUR} ET')
print(f'  Displacement: body>={DISP_BODY_RATIO_MIN} + close_str>={DISP_CLOSE_STR_MIN}')
print(f'  P/D: longs=discount | bears=premium OR had_sweep=1')
print(f"{'='*60}")

total_signals = 0; total_bars = 0

for ticker in TICKERS:
    label_path = os.path.join(CISD_CACHE_DIR, f'{ticker}_cisd_labels.parquet')
    feat_path  = os.path.join(CISD_CACHE_DIR, f'{ticker}_cisd_features.parquet')

    if os.path.exists(label_path) and os.path.exists(feat_path):
        cached = pd.read_parquet(label_path)
        sigs   = (cached['signal_label'] > 0).sum()
        print(f'\n  {ticker}: ⚡ cached — {len(cached):,} bars, {sigs} signals')
        total_signals += sigs; total_bars += len(cached)
        continue

    data_ticker   = MICRO_TO_FULL.get(ticker, ticker)
    csv_path      = os.path.join(RAW_DATA_DIR, f'{data_ticker}_5min.csv')
    ffm_feat_path = os.path.join(PREPARED_DIR,  f'{data_ticker}_features.parquet')
    if not os.path.exists(csv_path) or not os.path.exists(ffm_feat_path):
        print(f'\n  ⚠ Skip {ticker} — missing data'); continue

    is_micro = ticker in MICRO_TO_FULL
    print(f"\n{'─'*60}\n  {ticker}{f' (micro → {data_ticker})' if is_micro else ''}\n{'─'*60}")
    t0 = time.time()

    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    if 'date' in df_raw.columns and 'datetime' not in df_raw.columns:
        df_raw = df_raw.rename(columns={'date': 'datetime'})
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw.set_index('datetime', inplace=True)
    df_raw.sort_index(inplace=True)
    try:
        df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('America/New_York')
    except TypeError:
        if df_raw.index.tz is not None:
            df_raw.index = df_raw.index.tz_convert('America/New_York')
    print(f'  Loaded {len(df_raw):,} 5m bars')

    df_60m = resample_ohlcv(df_raw, '60min')
    df_60m = compute_trend_60min(df_60m)

    df_raw = detect_5min_cisd_signals(df_raw, TOLERANCE, SWING_PERIOD, EXPIRY_BARS,
                                       LIQUIDITY_LOOKBACK, body_ratio_min=DISP_BODY_RATIO_MIN,
                                       close_str_min=DISP_CLOSE_STR_MIN)
    cisd_count = (df_raw['cisd_signal'] > 0).sum()
    print(f'  5min CISDs: {cisd_count}')

    sig_arr, rr_arr, sl_arr, zone_meta, stats = label_cisd_ote_zones_5min(df_raw, df_60m, ticker)
    print(f'  Labels: {stats["buys"]}B + {stats["sells"]}S from {stats["trades"]} evaluated')

    ffm_df  = pd.read_parquet(ffm_feat_path)
    ffm_dt  = pd.to_datetime(ffm_df['_datetime'])
    if ffm_dt.dt.tz is None:
        ffm_dt = ffm_dt.dt.tz_localize('UTC').tz_convert('America/New_York')
    ffm_lookup = {dt: i for i, dt in enumerate(ffm_dt)}
    FFM_PULL_COLS = ['sess_dist_from_vwap', 'str_structure_state', 'ret_acceleration',
                     'vty_atr_of_atr', 'sess_dist_from_open', 'ret_momentum_10', 'vol_delta_proxy']
    FFM_PULL_COLS = [c for c in FFM_PULL_COLS if c in ffm_df.columns]
    raw_index = df_raw.index
    ffm_aligned = {}
    for col in FFM_PULL_COLS:
        arr = np.zeros(len(df_raw), dtype=np.float32)
        ffm_col_vals = ffm_df[col].values
        for raw_i, raw_dt in enumerate(raw_index):
            ffm_i = ffm_lookup.get(raw_dt)
            if ffm_i is not None: arr[raw_i] = ffm_col_vals[ffm_i]
        ffm_aligned[col] = arr

    cisd_feats = create_cisd_features(df_raw, zone_meta, ffm_features=ffm_aligned)

    n_ffm      = len(ffm_df)
    raw_lookup = {dt: i for i, dt in enumerate(df_raw.index)}
    aligned_sig   = np.zeros(n_ffm, dtype=np.int8)
    aligned_rr    = np.zeros(n_ffm, dtype=np.float32)
    aligned_sl    = np.zeros(n_ffm, dtype=np.float32)
    aligned_feats = np.zeros((n_ffm, NUM_CISD_FEATURES), dtype=np.float32)

    for i in range(n_ffm):
        dt = ffm_dt.iloc[i]
        if dt in raw_lookup:
            j = raw_lookup[dt]
            aligned_sig[i]   = 1 if sig_arr[j] > 0 else 0
            aligned_rr[i]    = rr_arr[j]
            aligned_sl[i]    = sl_arr[j] if not np.isnan(sl_arr[j]) else 0.0
            aligned_feats[i] = cisd_feats[j]

    labels_df = pd.DataFrame({'signal_label': aligned_sig, 'max_rr': aligned_rr,
                               'sl_distance': aligned_sl})
    feats_df  = pd.DataFrame(aligned_feats, columns=CISD_FEATURE_COLS)
    labels_df.to_parquet(label_path, index=False)
    feats_df.to_parquet(feat_path,   index=False)

    signals = (aligned_sig == 1).sum()
    total_signals += signals; total_bars += n_ffm
    print(f'  ✓ {ticker}: {signals} signals | ({time.time() - t0:.1f}s)')

print(f"\n{'='*60}")
print(f'  ✅ STEP 3 COMPLETE — {total_bars:,} bars | {total_signals} signals')
print(f'  {"✅ Label density OK" if total_signals >= 500 else "⚠️  Label density LOW"}')
print(f"{'='*60}")

print(f"\n{'='*60}\n  📊 LABEL WIN RATE DIAGNOSTIC\n{'='*60}")
for ticker in TICKERS:
    label_path = os.path.join(CISD_CACHE_DIR, f'{ticker}_cisd_labels.parquet')
    if not os.path.exists(label_path): continue
    ldf   = pd.read_parquet(label_path)
    total = (ldf['signal_label'] > 0).sum()
    if total == 0: print(f'  {ticker}: no labels'); continue
    sig      = ldf[ldf['signal_label'] > 0]
    wins_2r  = (sig['max_rr'] >= 2.0).sum()
    wins_3r  = (sig['max_rr'] >= 3.0).sum()
    baseline = BASELINE_WR.get(ticker, 0.30)
    wr_2r    = wins_2r / total
    status   = '✅' if wr_2r >= baseline - 0.05 else '⚠️ '
    feat_path_d = os.path.join(CISD_CACHE_DIR, f'{ticker}_cisd_features.parquet')
    bull_str = ''
    if os.path.exists(feat_path_d):
        fdf = pd.read_parquet(feat_path_d)
        sig_mask = ldf['signal_label'] > 0
        if len(fdf) == len(ldf):
            zone_bull = fdf.loc[sig_mask, 'zone_is_bullish']
            bull_str  = f' | Bull:{(zone_bull > 0).sum()} Bear:{(zone_bull < 0).sum()}'
    print(f'  {status} {ticker}: {total} signals{bull_str} | '
          f'WR@2R:{wr_2r*100:.1f}% WR@3R:{wins_3r/total*100:.1f}% | baseline:{baseline*100:.0f}%')
print(f"{'='*60}")


# ==============================================================================
# CELL 4 — WALK-FORWARD HYBRID FINE-TUNING
# ==============================================================================

import hashlib, json as _json, random, gc, math, importlib

_RESUME_CONFIG = {
    'signal_ratio': SIGNAL_RATIO, 'miss': MISS_PENALTY, 'lr': LR,
    'freeze': FREEZE_RATIO, 'num_labels': NUM_LABELS,
    'body_ratio': DISP_BODY_RATIO_MIN, 'focal_gamma': FOCAL_GAMMA,
    'smoothing': FOCAL_SMOOTHING, 'sess_start': SESSION_START_HOUR,
    'seq_len': SEQ_LEN,
}
CONFIG_HASH = hashlib.md5(
    _json.dumps(_RESUME_CONFIG, sort_keys=True).encode()
).hexdigest()[:8]
print(f'Config hash: {CONFIG_HASH}')

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import futures_foundation
importlib.reload(futures_foundation)
from futures_foundation import FFMConfig, FFMBackbone, get_model_feature_columns

os.makedirs(OUTPUT_DIR, exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, label_smoothing=0.10):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs    = F.log_softmax(logits, dim=-1)
        ce           = -(smooth_targets * log_probs).sum(dim=-1)
        pt           = torch.exp(-F.cross_entropy(logits, targets, weight=self.weight, reduction='none'))
        focal_weight = (1 - pt) ** self.gamma
        loss         = focal_weight * ce
        if self.weight is not None: loss = loss * self.weight[targets]
        return loss.mean()


class HybridCISDModel(nn.Module):
    def __init__(self, config, num_cisd_features=28, num_labels=2,
                 num_risk_targets=1, risk_weight=0.1):
        super().__init__()
        self.num_labels = num_labels; self.risk_weight = risk_weight
        self.backbone = FFMBackbone(config)
        self.cisd_projection = nn.Sequential(
            nn.Linear(num_cisd_features, 64), nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob), nn.Linear(64, 64))
        combined_dim = config.hidden_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_size), nn.GELU(),
            nn.LayerNorm(config.hidden_size), nn.Dropout(config.hidden_dropout_prob))
        self.signal_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2), nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_labels))
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2), nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, num_risk_targets), nn.Softplus())
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4), nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 4, 1), nn.Sigmoid())

    def load_backbone(self, path):
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        # best_backbone.pt is model.backbone.state_dict() — keys have no prefix
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f'  ⚠ Backbone unexpected keys: {len(unexpected)}')
        print(f'  ✅ Backbone loaded — {len(state_dict)} tensors, {len(missing)} missing')

    def freeze_backbone(self, freeze_ratio=0.66):
        groups = self.backbone.get_layer_groups()
        num_freeze = int(len(groups) * freeze_ratio)
        frozen = trainable = 0
        for i, (name, params) in enumerate(groups):
            for p in params:
                p.requires_grad = i >= num_freeze
                if i < num_freeze: frozen   += p.numel()
                else:              trainable += p.numel()
        head_params = sum(p.numel() for m in [self.cisd_projection, self.fusion,
                          self.signal_head, self.risk_head, self.confidence_head]
                          for p in m.parameters())
        trainable += head_params
        print(f'  Frozen {num_freeze}/{len(groups)} layers | '
              f'{frozen:,} frozen, {trainable:,} trainable')

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, features, cisd_features, candle_types=None,    # FIXED v6.0
                time_of_day=None, day_of_week=None,
                instrument_ids=None, session_ids=None, attention_mask=None):
        embedding  = self.backbone(
            features=features,
            candle_types=candle_types,                                 # FIXED v6.0
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            instrument_ids=instrument_ids,
            session_ids=session_ids,
            attention_mask=attention_mask,
            output_sequence=False,
        )
        cisd_embed = self.cisd_projection(cisd_features)
        fused      = self.fusion(torch.cat([embedding, cisd_embed], dim=-1))
        return {
            'signal_logits':    self.signal_head(fused),
            'risk_predictions': self.risk_head(fused),
            'confidence':       self.confidence_head(fused).squeeze(-1),
        }


class HybridCISDDataset(Dataset):
    def __init__(self, features_df, cisd_features_df, cisd_labels_df, seq_len=96, stride=1):
        self.seq_len      = seq_len
        self.feature_cols = get_model_feature_columns()

        valid = features_df[self.feature_cols].notna().all(axis=1)
        self.features   = features_df[valid].reset_index(drop=True)
        self.cisd_feats = cisd_features_df[valid].reset_index(drop=True)
        self.labels     = cisd_labels_df[valid].reset_index(drop=True)

        self.window_starts = list(range(0, len(self.features) - seq_len + 1, stride))

        self._f    = np.nan_to_num(self.features[self.feature_cols].values.astype(np.float32))
        self._cisd = np.nan_to_num(self.cisd_feats[CISD_FEATURE_COLS].values.astype(np.float32))

        self._inst   = self.features.get('_instrument_id',
                       pd.Series(0, index=self.features.index)).values.astype(np.int64)
        self._sess   = self.features.get('sess_id',
                       pd.Series(0, index=self.features.index)).values.astype(np.int64)
        self._tod    = self.features.get('sess_time_of_day',
                       pd.Series(0.0, index=self.features.index)).values.astype(np.float32)
        self._dow    = self.features.get('tmp_day_of_week',
                       pd.Series(0, index=self.features.index)).values.astype(np.int64)
        # FIXED v6.0: extract candle_type so backbone uses its trained embedding
        self._candle = self.features.get('candle_type',
                       pd.Series(0, index=self.features.index)).fillna(0).values.astype(np.int64)

        self._labels = self.labels['signal_label'].values.astype(np.int64)
        self._max_rr = self.labels['max_rr'].values.astype(np.float32)

        # Pre-compute signal window indices for balanced sampling
        self.signal_indices = [
            i for i in range(len(self.window_starts))
            if self._labels[self.window_starts[i] + seq_len - 1] > 0
        ]

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end   = start + self.seq_len
        last  = end - 1
        return {
            'features':       torch.from_numpy(self._f[start:end]),
            'cisd_features':  torch.from_numpy(self._cisd[last]),
            'candle_types':   torch.from_numpy(self._candle[start:end]),   # FIXED v6.0
            'instrument_ids': torch.tensor(self._inst[start], dtype=torch.long),
            'session_ids':    torch.from_numpy(self._sess[start:end]),
            'time_of_day':    torch.from_numpy(self._tod[start:end]),
            'day_of_week':    torch.from_numpy(self._dow[start:end]),
            'signal_label':   torch.tensor(self._labels[last], dtype=torch.long),
            'max_rr':         torch.tensor(self._max_rr[last], dtype=torch.float32),
        }


def make_balanced_loader(dataset, batch_size, sig_per_batch, shuffle=True, num_workers=2):
    """WeightedRandomSampler that delivers ~sig_per_batch signals per batch."""
    if not dataset.signal_indices or len(dataset.signal_indices) < sig_per_batch:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    n_total  = len(dataset)
    n_signal = len(dataset.signal_indices)
    n_noise  = n_total - n_signal
    # Weight signals so they appear at rate sig_per_batch / batch_size
    target_signal_frac = sig_per_batch / batch_size
    w_signal = target_signal_frac / (n_signal / n_total) if n_signal > 0 else 1.0
    w_noise  = (1.0 - target_signal_frac) / (n_noise / n_total) if n_noise > 0 else 1.0
    labels_last = [dataset._labels[s + dataset.seq_len - 1] for s in dataset.window_starts]
    weights = [w_signal if l > 0 else w_noise for l in labels_last]
    sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                     num_workers=num_workers, pin_memory=True, drop_last=True)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0; n_batches = 0
    correct = total = sig_correct = sig_total = 0

    for batch in loader:
        feats    = batch['features'].to(device)
        cisd     = batch['cisd_features'].to(device)
        candles  = batch['candle_types'].to(device)          # FIXED v6.0
        inst     = batch['instrument_ids'].to(device)
        sess     = batch['session_ids'].to(device)
        tod      = batch['time_of_day'].to(device)
        dow      = batch['day_of_week'].to(device)
        labels   = batch['signal_label'].to(device)
        max_rr   = batch['max_rr'].to(device)

        optimizer.zero_grad()
        out = model(features=feats, cisd_features=cisd, candle_types=candles,  # FIXED v6.0
                    time_of_day=tod, day_of_week=dow,
                    instrument_ids=inst, session_ids=sess)

        cls_loss  = loss_fn(out['signal_logits'], labels)
        risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
        loss = cls_loss + model.risk_weight * risk_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item(); n_batches += 1
        preds = out['signal_logits'].argmax(dim=-1)
        correct  += (preds == labels).sum().item(); total += labels.size(0)
        sig_mask  = labels > 0
        if sig_mask.any():
            sig_correct += (preds[sig_mask] == labels[sig_mask]).sum().item()
            sig_total   += sig_mask.sum().item()

    return {
        'loss':        total_loss / max(n_batches, 1),
        'acc':         correct / max(total, 1),
        'sig_recall':  sig_correct / max(sig_total, 1),
        'sig_total':   sig_total,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0; n_batches = 0
    correct = total = 0
    tp = fp = fn = 0
    all_conf = []; all_labels = []; all_preds = []

    for batch in loader:
        feats    = batch['features'].to(device)
        cisd     = batch['cisd_features'].to(device)
        candles  = batch['candle_types'].to(device)          # FIXED v6.0
        inst     = batch['instrument_ids'].to(device)
        sess     = batch['session_ids'].to(device)
        tod      = batch['time_of_day'].to(device)
        dow      = batch['day_of_week'].to(device)
        labels   = batch['signal_label'].to(device)
        max_rr   = batch['max_rr'].to(device)

        out = model(features=feats, cisd_features=cisd, candle_types=candles,  # FIXED v6.0
                    time_of_day=tod, day_of_week=dow,
                    instrument_ids=inst, session_ids=sess)

        cls_loss  = loss_fn(out['signal_logits'], labels)
        risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
        loss = cls_loss + model.risk_weight * risk_loss
        total_loss += loss.item(); n_batches += 1

        preds = out['signal_logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        tp += ((preds > 0) & (labels > 0)).sum().item()
        fp += ((preds > 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels > 0)).sum().item()
        all_conf.extend(out['confidence'].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'loss':      total_loss / max(n_batches, 1),
        'acc':       correct / max(total, 1),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': all_conf, 'all_labels': all_labels, 'all_preds': all_preds,
    }


def load_fold_data(fold, tickers, ffm_dir, cisd_dir, seq_len):
    """Load and time-slice data for a single fold, returning train/val/test datasets."""
    train_end = pd.Timestamp(fold['train_end'])
    val_end   = pd.Timestamp(fold['val_end'])
    test_end  = pd.Timestamp(fold['test_end'])

    train_dsets = []; val_dsets = []; test_dsets = []

    for ticker in tickers:
        data_ticker  = MICRO_TO_FULL.get(ticker, ticker)
        ffm_path     = os.path.join(ffm_dir,  f'{data_ticker}_features.parquet')
        cisd_f_path  = os.path.join(cisd_dir, f'{ticker}_cisd_features.parquet')
        cisd_l_path  = os.path.join(cisd_dir, f'{ticker}_cisd_labels.parquet')

        if not all(os.path.exists(p) for p in [ffm_path, cisd_f_path, cisd_l_path]):
            print(f'  ⚠ Skip {ticker} — missing parquet'); continue

        ffm_df  = pd.read_parquet(ffm_path)
        cisd_f  = pd.read_parquet(cisd_f_path)
        cisd_l  = pd.read_parquet(cisd_l_path)

        dt_col = pd.to_datetime(ffm_df['_datetime'])
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize('UTC').tz_convert('America/New_York')

        tr_mask   = dt_col <  train_end
        val_mask  = (dt_col >= train_end) & (dt_col < val_end)
        test_mask = (dt_col >= val_end)   & (dt_col < test_end)

        for mask, dset_list, stride, tag in [
            (tr_mask,   train_dsets, 1, 'train'),
            (val_mask,  val_dsets,   1, 'val'),
            (test_mask, test_dsets,  1, 'test'),
        ]:
            idx = np.where(mask.values)[0]
            if len(idx) < seq_len + 1: continue
            lo, hi = idx[0], idx[-1] + 1
            ds = HybridCISDDataset(
                ffm_df.iloc[lo:hi].reset_index(drop=True),
                cisd_f.iloc[lo:hi].reset_index(drop=True),
                cisd_l.iloc[lo:hi].reset_index(drop=True),
                seq_len=seq_len, stride=stride,
            )
            if len(ds.signal_indices) == 0:
                print(f'  ⚠ {ticker} {tag}: 0 signals — skipping')
                continue
            dset_list.append(ds)
            print(f'  {ticker} {tag}: {len(ds):,} windows, {len(ds.signal_indices)} signals')

    return train_dsets, val_dsets, test_dsets


def train_fold(fold, config):
    fold_name = fold['name']
    ckpt_path = os.path.join(OUTPUT_DIR, f'{fold_name}_{CONFIG_HASH}_best.pt')

    print(f"\n{'='*60}")
    print(f'  FOLD {fold_name} | train<{fold["train_end"]} val<{fold["val_end"]}')
    print(f"{'='*60}")

    # ── Build datasets ──
    train_dsets, val_dsets, test_dsets = load_fold_data(
        fold, TICKERS, PREPARED_DIR, CISD_CACHE_DIR, SEQ_LEN)

    if not train_dsets or not val_dsets:
        print(f'  ⚠ {fold_name}: insufficient data — skipping'); return None

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_dsets)
    val_ds   = ConcatDataset(val_dsets)
    test_ds  = ConcatDataset(test_dsets) if test_dsets else None

    # Build signal index list across ConcatDataset for balanced sampling
    train_ds.signal_indices = []
    train_ds.seq_len = SEQ_LEN
    train_ds._labels = np.concatenate([
        d._labels[d.window_starts[i] + d.seq_len - 1:
                  d.window_starts[i] + d.seq_len]
        for d in train_dsets for i in range(len(d))
    ])
    offset = 0
    for d in train_dsets:
        for local_i in d.signal_indices:
            train_ds.signal_indices.append(offset + local_i)
        offset += len(d)

    n_sig   = len(train_ds.signal_indices)
    n_total = len(train_ds)
    print(f'\n  Train: {n_total:,} windows, {n_sig} signals ({n_sig/n_total*100:.2f}%)')
    print(f'  Val:   {len(val_ds):,} windows')

    train_loader = make_balanced_loader(train_ds, BATCH_SIZE, SIG_PER_BATCH)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ──
    model = HybridCISDModel(config, NUM_CISD_FEATURES, NUM_LABELS,
                            risk_weight=RISK_WEIGHT).to(device)

    if os.path.exists(BACKBONE_PATH):
        print(f'  Loading backbone: {BACKBONE_PATH}')
        model.load_backbone(BACKBONE_PATH)
    else:
        print(f'  ⚠ Backbone not found at {BACKBONE_PATH} — training from scratch')

    model.freeze_backbone(FREEZE_RATIO)

    # ── Loss + optimizer ──
    class_weights = torch.tensor([FALSE_PENALTY, MISS_PENALTY], dtype=torch.float32).to(device)
    loss_fn   = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights,
                          label_smoothing=FOCAL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_loader)
    warmup      = min(500, total_steps // 10)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=warmup / total_steps, anneal_strategy='cos',
    )

    # ── Resume check ──
    start_epoch = 0
    best_val_loss = float('inf')
    patience_ctr  = 0
    ratio_bad_ctr = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get('config_hash') == CONFIG_HASH:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            start_epoch   = ckpt['epoch'] + 1
            best_val_loss = ckpt['val_loss']
            patience_ctr  = ckpt.get('patience_ctr', 0)
            print(f'  ▶ Resumed from epoch {start_epoch} (val_loss={best_val_loss:.4f})')
        else:
            print(f'  ℹ Config changed — starting fresh')

    # ── Training loop ──
    for epoch in range(start_epoch, EPOCHS):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)

        scheduler.step()

        ratio = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        improved = va['loss'] < best_val_loss
        save_str = ''

        if improved:
            best_val_loss = va['loss']
            patience_ctr  = 0
            ratio_bad_ctr = 0
            torch.save({
                'config_hash': CONFIG_HASH, 'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': best_val_loss, 'patience_ctr': patience_ctr,
            }, ckpt_path)
            save_str = ' ✅ SAVE'

        overfitting = ratio > MAX_RATIO
        if overfitting:
            ratio_bad_ctr += 1
            ratio_str = f'🚨 ratio={ratio:.2f} ({ratio_bad_ctr}/{RATIO_PATIENCE})'
        else:
            ratio_bad_ctr = 0
            ratio_str = f'OK ratio={ratio:.2f}'

        if not improved:
            patience_ctr += 1

        print(f'  E{epoch+1:2d}/{EPOCHS} | '
              f'TrL:{tr["loss"]:.4f} VL:{va["loss"]:.4f} | '
              f'Acc:{va["acc"]:.3f} P:{va["precision"]:.3f} R:{va["recall"]:.3f} F1:{va["f1"]:.3f} | '
              f'{ratio_str}{save_str}')

        if patience_ctr >= PATIENCE:
            print(f'  ⏹ Early stop — patience {PATIENCE} exhausted'); break
        if ratio_bad_ctr >= RATIO_PATIENCE:
            print(f'  ⏹ Early stop — overfitting ratio exceeded {RATIO_PATIENCE}x'); break

    # ── Load best and evaluate on test ──
    if os.path.exists(ckpt_path):
        best = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best['model_state'])

    test_metrics = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        print(f'\n  📊 Test | Acc:{test_metrics["acc"]:.3f} '
              f'P:{test_metrics["precision"]:.3f} R:{test_metrics["recall"]:.3f} '
              f'F1:{test_metrics["f1"]:.3f} | '
              f'TP:{test_metrics["tp"]} FP:{test_metrics["fp"]} FN:{test_metrics["fn"]}')

    return model, test_metrics


# ── Walk-forward ──
print(f'\n{"="*60}')
print(f'  WALK-FORWARD FINE-TUNING — {len(FOLDS)} folds')
print(f'  Backbone: {BACKBONE_PATH}')
print(f'{"="*60}')

if not os.path.exists(BACKBONE_PATH):
    print(f'⚠️  Backbone checkpoint not found: {BACKBONE_PATH}')
    print('   Run the pretraining script first (ffm_pretrain_5min.py)')
    raise FileNotFoundError(BACKBONE_PATH)

config = FFMConfig(
    num_features=len(get_model_feature_columns()),
    label_smoothing=FOCAL_SMOOTHING,
)

fold_results = {}
last_model   = None

for fold in FOLDS:
    result = train_fold(fold, config)
    if result is not None:
        last_model, test_metrics = result
        fold_results[fold['name']] = test_metrics
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── ONNX export (F4 model — production deployment) ──
if last_model is not None:
    print(f'\n{"="*60}\n  ONNX EXPORT\n{"="*60}')
    last_model.eval()
    onnx_path = os.path.join(OUTPUT_DIR, 'cisd_ote_hybrid.onnx')
    dummy = {
        'features':       torch.randn(1, SEQ_LEN, len(get_model_feature_columns())).to(device),
        'cisd_features':  torch.randn(1, NUM_CISD_FEATURES).to(device),
        'candle_types':   torch.zeros(1, SEQ_LEN, dtype=torch.long).to(device),
        'instrument_ids': torch.zeros(1, dtype=torch.long).to(device),
        'session_ids':    torch.zeros(1, SEQ_LEN, dtype=torch.long).to(device),
        'time_of_day':    torch.zeros(1, SEQ_LEN).to(device),
        'day_of_week':    torch.zeros(1, SEQ_LEN, dtype=torch.long).to(device),
    }
    try:
        torch.onnx.export(
            last_model,
            (dummy['features'], dummy['cisd_features'], dummy['candle_types'],
             dummy['time_of_day'], dummy['day_of_week'],
             dummy['instrument_ids'], dummy['session_ids']),
            onnx_path,
            input_names=['features', 'cisd_features', 'candle_types',
                         'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'],
            output_names=['signal_logits', 'risk_predictions', 'confidence'],
            dynamic_axes={
                'features':       {0: 'batch', 1: 'seq'},
                'cisd_features':  {0: 'batch'},
                'candle_types':   {0: 'batch', 1: 'seq'},
                'time_of_day':    {0: 'batch', 1: 'seq'},
                'day_of_week':    {0: 'batch', 1: 'seq'},
                'instrument_ids': {0: 'batch'},
                'session_ids':    {0: 'batch', 1: 'seq'},
                'signal_logits':  {0: 'batch'},
                'risk_predictions': {0: 'batch'},
                'confidence':     {0: 'batch'},
            },
            opset_version=17,
        )
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f'  ✅ ONNX exported: {onnx_path} ({size_mb:.1f} MB)')
    except Exception as e:
        print(f'  ❌ ONNX export failed: {e}')


# ==============================================================================
# CELL 5 — EVALUATION SUMMARY
# ==============================================================================

print(f'\n{"="*60}')
print(f'  WALK-FORWARD RESULTS SUMMARY')
print(f'{"="*60}')

if not fold_results:
    print('  No fold results available.')
else:
    for fname, metrics in fold_results.items():
        if metrics is None:
            print(f'  {fname}: no test data'); continue
        conf_arr   = np.array(metrics['all_conf'])
        label_arr  = np.array(metrics['all_labels'])
        pred_arr   = np.array(metrics['all_preds'])
        print(f'\n  {fname}:')
        print(f'    Overall  — Acc:{metrics["acc"]:.3f} P:{metrics["precision"]:.3f} '
              f'R:{metrics["recall"]:.3f} F1:{metrics["f1"]:.3f}')
        print(f'    Signals  — TP:{metrics["tp"]} FP:{metrics["fp"]} FN:{metrics["fn"]}')

        # Confidence threshold sweep
        print(f'    Conf threshold analysis:')
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            high_conf = conf_arr >= thresh
            if high_conf.sum() == 0: continue
            hc_labels = label_arr[high_conf]
            hc_preds  = pred_arr[high_conf]
            hc_tp = ((hc_preds > 0) & (hc_labels > 0)).sum()
            hc_fp = ((hc_preds > 0) & (hc_labels == 0)).sum()
            hc_fn = ((hc_preds == 0) & (hc_labels > 0)).sum()
            hc_p  = hc_tp / max(hc_tp + hc_fp, 1)
            hc_r  = hc_tp / max(hc_tp + hc_fn, 1)
            print(f'      conf≥{thresh:.0%}: {high_conf.sum():,} windows | '
                  f'P:{hc_p:.3f} R:{hc_r:.3f} | TP:{hc_tp} FP:{hc_fp}')

    print(f'\n  ONNX model: {os.path.join(OUTPUT_DIR, "cisd_ote_hybrid.onnx")}')
    print(f'  Checkpoints: {OUTPUT_DIR}')
