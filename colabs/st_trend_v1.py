# ==============================================================================
# SUPERTREND TREND FOLLOW v1.0 — PHASE 1: SIGNAL CLASSIFIER
# ==============================================================================
# Labels 5min bars where a SuperTrend (ST10/2.0) direction flip occurs AND the
# 1-hour SuperTrend (ST10/3.0) agrees — capturing trend-following entries that
# avoid counter-trend chop.
#
# Signal: 5min ST flips bull/bear + 1h ST alignment confirmed
# Stop:   ATR × 1.5 from entry close
# Target: 1R minimum (for label); walk-forward training targets WR@2R
#
# Validated metrics (local diagnostic, 5 tickers, 5yr):
#   NQ:  1420 signals | WR@2R 71.4% | WR@3R 56.1%
#   GC:  1702 signals | WR@2R 74.3% | WR@3R 61.5%
#   ES:   820 signals | WR@2R 69.1% | WR@3R 52.6%
#   RTY: 3052 signals | WR@2R 69.0% | WR@3R 52.9%
#   YM:  2329 signals | WR@2R 71.0% | WR@3R 56.9%
#   TOTAL: 9323 signals | F1 train: 2311 | year-by-year: 64.7%–74.5%
# ==============================================================================


# ==============================================================================
# CELL 1 — SETUP
# ==============================================================================

import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive')

print('Cloning FFM repo...')
os.system('rm -rf /content/Futures-Foundation-Model')
result = subprocess.run(
    ['git', 'clone', 'https://github.com/johnamcruz/Futures-Foundation-Model.git',
     '/content/Futures-Foundation-Model'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f'Clone failed:\n{result.stderr}')
    raise RuntimeError('Git clone failed')
print('Cloned')

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')
os.system('pip install onnxscript -q 2>&1 | tail -1')

try:
    from futures_foundation import FFMConfig, FFMBackbone, get_model_feature_columns
    print(f'FFM installed — {len(get_model_feature_columns())} features')
except (ImportError, ValueError) as e:
    print(f'Import failed: {e}')
    print('Restarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ==============================================================================
# CELL 2 — CONFIGURATION
# ==============================================================================

import torch, shutil

# ── PATHS ──
RAW_DATA_DIR = '/content/drive/MyDrive/Futures Data/5min'
PREPARED_DIR = '/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared'
BACKBONE_PATH = '/content/drive/MyDrive/AI_Cache/5min_FFM_Checkpoints/best_backbone.pt'
ST_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/ST_Trend_Labels_v1'
OUTPUT_DIR   = '/content/drive/MyDrive/AI_Models/ST_Trend_v1'

# ── TICKERS ──
DATA_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC']
MICRO_TO_FULL = {
    'MES': 'ES', 'MNQ': 'NQ', 'MRTY': 'RTY', 'MYM': 'YM', 'MGC': 'GC',
}
TICKERS = DATA_TICKERS

# ── SUPERTREND PARAMS ──
ST_PERIOD    = 10     # 5min SuperTrend period
ST_MULT      = 2.0    # 5min SuperTrend multiplier
HTF_PERIOD   = 10     # 1h SuperTrend period
HTF_MULT     = 3.0    # 1h SuperTrend multiplier (wider = fewer false flips)
LOOKAHEAD    = 192    # bars to measure outcome (192 × 5min ≈ 16 hours)
SL_ATR_MULT  = 1.5    # stop = close ± ATR × SL_ATR_MULT
MIN_SIGNAL_RR = 1.0   # minimum R:R for signal label
ATR_RANK_WINDOW = 200 # bars for rolling ATR percentile

# ── SIGNAL CLASS ──
NUM_LABELS = 2

# ── ST FEATURES (8) ──
# Only features the FFM backbone cannot derive: ST state and prior trend geometry.
# HTF trend direction, volatility regime, session context, and price structure
# are already in the 67-feature backbone sequence.
ST_FEATURE_COLS = [
    'st_direction',              # current 5m ST direction: +1=bull, -1=bear
    'st_line_distance',          # (close - ST_line) / ATR — signed distance from ST
    'prior_trend_duration_norm', # bars in prior trend (before flip) / 100, capped 0-5
    'prior_trend_extent_norm',   # prior trend price range / ATR, capped 0-10
    'atr_rank_pct',              # rolling ATR percentile (200-bar window), 0-1
    'htf_st_direction',          # 1h ST direction: +1=bull, -1=bear
    'htf_st_age_norm',           # bars since last 1h ST flip / 100, capped 0-5
    'flip_bar_body_pct',         # flip candle body / range (0 on non-signal bars)
]
NUM_ST_FEATURES = len(ST_FEATURE_COLS)  # 8

# ── LABELING ──
MIN_SIGNAL_RR = 1.0
RR_TARGETS    = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# ── WALK-FORWARD FOLDS ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── TRAINING ──
SEQ_LEN           = 96
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

# ── WARM START ──
WARM_START_MODE        = 'selective'
BACKBONE_LR_MULTIPLIER = 0.1

# ── MECHANICAL BASELINE WR@2R (validated locally, 5yr diagnostic) ──
BASELINE_WR = {
    'ES': 0.69, 'NQ': 0.71, 'RTY': 0.69, 'YM': 0.71, 'GC': 0.74,
    'MES': 0.69, 'MNQ': 0.71, 'MRTY': 0.69, 'MYM': 0.71, 'MGC': 0.74,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
print(f'ST Trend Follow v1.0 | ST{ST_PERIOD}/{ST_MULT} + HTF ST{HTF_PERIOD}/{HTF_MULT}')
print(f'FFM {SEQ_LEN}-bar context + {NUM_ST_FEATURES} ST features | {NUM_LABELS}-class signal')
print(f'LR:{LR} | Freeze:{FREEZE_RATIO:.0%} | SIG/batch:{SIG_PER_BATCH}')
print(f'WarmStart:{WARM_START_MODE} | BackboneLR×{BACKBONE_LR_MULTIPLIER}')
print(f'Tickers: {TICKERS}')
print(f'Backbone: {BACKBONE_PATH}')

shutil.rmtree(ST_CACHE_DIR, ignore_errors=True)
os.makedirs(ST_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# CELL 3 — SUPERTREND LABELING + FEATURE GENERATION (5min)
# ==============================================================================

import time
import numpy as np
import pandas as pd


def compute_st(h, l, c, period, mult):
    """
    Wilder's ATR SuperTrend. Returns (direction, st_line, atr).
    direction: +1=bull, -1=bear.
    CRITICAL: initialize direction to 1 (bull) so the state machine starts correctly.
    """
    n = len(c)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i - 1]) / period
    atr = np.maximum(atr, 1e-6)

    hl2 = (h + l) / 2.0
    ub = (hl2 + mult * atr).copy()
    lb = (hl2 - mult * atr).copy()

    d = np.ones(n, dtype=np.int8)       # start bullish
    st_line = lb.copy()                  # start at lower band (bull side)

    for i in range(1, n):
        ub[i] = ub[i] if (ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = lb[i] if (lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
        if d[i - 1] == -1:
            d[i] = 1 if c[i] > ub[i - 1] else -1
        else:
            d[i] = -1 if c[i] < lb[i - 1] else 1
        st_line[i] = lb[i] if d[i] == 1 else ub[i]

    return d, st_line, atr


def get_htf_direction(df_5m):
    """Resample 5min to 1h, compute ST, forward-fill back to 5min index."""
    df1h = df_5m.resample('1h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()
    fd1h, _, _ = compute_st(
        df1h['high'].values, df1h['low'].values, df1h['close'].values,
        HTF_PERIOD, HTF_MULT,
    )
    return (pd.Series(fd1h, index=df1h.index)
              .reindex(df_5m.index, method='ffill')
              .fillna(1)
              .values.astype(np.int8))


def label_supertrend_signals(df_5m, htf_direction):
    """
    Produce signal labels, max_rr, sl_distance, and 8 strategy features.

    Signal fires when:
      1. 5min ST flips direction (bull→bear or bear→bull)
      2. 1h ST agrees with the new 5min direction

    Stop:  close ± ATR × SL_ATR_MULT
    Gain:  measure forward, stop-first (break on stop hit before updating best)
    """
    n = len(df_5m)
    opens  = df_5m['open'].values
    highs  = df_5m['high'].values
    lows   = df_5m['low'].values
    closes = df_5m['close'].values

    fd, st_line, atr = compute_st(highs, lows, closes, ST_PERIOD, ST_MULT)

    # Rolling ATR percentile (200-bar window)
    atr_rank = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        start  = max(0, i - ATR_RANK_WINDOW)
        window = atr[start:i]
        atr_rank[i] = float(np.sum(window < atr[i])) / max(len(window), 1)

    # Bars since last 1h ST flip (forward-filled to 5min resolution)
    htf_age   = np.zeros(n, dtype=np.float32)
    htf_flip_bar = 0
    for i in range(1, n):
        if htf_direction[i] != htf_direction[i - 1]:
            htf_flip_bar = i
        htf_age[i] = min((i - htf_flip_bar) / 100.0, 5.0)

    # Output arrays
    signal_labels = np.zeros(n, dtype=np.int8)
    max_rr_arr    = np.zeros(n, dtype=np.float32)
    sl_dist_arr   = np.zeros(n, dtype=np.float32)
    features      = np.zeros((n, NUM_ST_FEATURES), dtype=np.float32)

    # Trend tracking for prior-trend stats (computed at each flip)
    trend_start      = 0
    trend_phase_high = highs[0]
    trend_phase_low  = lows[0]

    for i in range(1, n):
        is_flip = (fd[i] != fd[i - 1])

        # Update running extent for the CURRENT trend (non-flip bars only)
        if not is_flip:
            if fd[i] == 1:
                trend_phase_high = max(trend_phase_high, highs[i])
            else:
                trend_phase_low = min(trend_phase_low, lows[i])

        # Per-bar features populated for every bar
        st_dist      = (closes[i] - st_line[i]) / max(atr[i], 1e-6)
        features[i, 0] = fd[i]                               # st_direction
        features[i, 1] = float(np.clip(st_dist, -10, 10))   # st_line_distance
        features[i, 4] = atr_rank[i]                         # atr_rank_pct
        features[i, 5] = htf_direction[i]                    # htf_st_direction
        features[i, 6] = htf_age[i]                          # htf_st_age_norm

        if is_flip:
            # Compute prior-trend stats BEFORE resetting state
            prior_duration = i - trend_start
            prior_range    = trend_phase_high - trend_phase_low
            prior_dur_norm = min(prior_duration / 100.0, 5.0)
            prior_ext_norm = min(prior_range / max(atr[i], 1e-6), 10.0)

            bar_range  = highs[i] - lows[i]
            flip_body  = abs(closes[i] - opens[i]) / max(bar_range, 1e-6)

            features[i, 2] = prior_dur_norm   # prior_trend_duration_norm
            features[i, 3] = prior_ext_norm   # prior_trend_extent_norm
            features[i, 7] = flip_body        # flip_bar_body_pct

            # Reset trend tracking for the new trend
            trend_start      = i
            trend_phase_high = highs[i]
            trend_phase_low  = lows[i]

            # Signal: HTF must agree with the new 5min direction
            if htf_direction[i] == fd[i]:
                is_long     = (fd[i] == 1)
                sl_dist     = atr[i] * SL_ATR_MULT
                entry_price = closes[i]
                sl_price    = entry_price - sl_dist if is_long else entry_price + sl_dist

                if sl_dist > 0:
                    best = 0.0
                    for j in range(i + 1, min(i + LOOKAHEAD + 1, n)):
                        if is_long:
                            if lows[j] <= sl_price:
                                break
                            best = max(best, (highs[j] - entry_price) / sl_dist)
                        else:
                            if highs[j] >= sl_price:
                                break
                            best = max(best, (entry_price - lows[j]) / sl_dist)

                    max_rr_arr[i]  = best
                    sl_dist_arr[i] = sl_dist
                    if best >= MIN_SIGNAL_RR:
                        signal_labels[i] = 1 if is_long else 2

    return signal_labels, max_rr_arr, sl_dist_arr, features


# ── ST TREND LABELER (wraps strategy logic for the fine-tuning framework) ──

from futures_foundation.finetune import StrategyLabeler, run_labeling


class STTrendLabeler(StrategyLabeler):
    """Bridges SuperTrend flip strategy with the generic fine-tuning framework."""

    @property
    def name(self):
        return 'st_trend'

    @property
    def feature_cols(self):
        return ST_FEATURE_COLS

    def run(self, df_raw, ffm_df, ticker):
        htf_dir = get_htf_direction(df_raw)
        sig_arr, rr_arr, sl_arr, feats = label_supertrend_signals(df_raw, htf_dir)

        n_long  = int((sig_arr == 1).sum())
        n_short = int((sig_arr == 2).sum())
        print(f'  Labels: {n_long}L + {n_short}S = {n_long + n_short} signals')

        sig_s   = pd.Series(sig_arr.astype(np.int8),   index=df_raw.index)
        rr_s    = pd.Series(rr_arr.astype(np.float32), index=df_raw.index)
        sl_s    = pd.Series(sl_arr.astype(np.float32), index=df_raw.index)
        feats_df = pd.DataFrame(feats, index=df_raw.index, columns=ST_FEATURE_COLS)

        aligned_sig   = (sig_s.reindex(ffm_df.index).fillna(0).values > 0).astype(np.int8)
        aligned_rr    = rr_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_sl    = sl_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_feats = feats_df.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)

        labels_df = pd.DataFrame({
            'signal_label': aligned_sig,
            'max_rr':       aligned_rr,
            'sl_distance':  aligned_sl,
        })
        feats_out = pd.DataFrame(aligned_feats, columns=ST_FEATURE_COLS)
        return feats_out, labels_df


labeler = STTrendLabeler()
run_labeling(labeler, TICKERS, RAW_DATA_DIR, PREPARED_DIR, ST_CACHE_DIR,
             micro_to_full=MICRO_TO_FULL)

# ── Win-rate diagnostic ──
print(f"\n{'='*60}\n  LABEL WIN RATE DIAGNOSTIC\n{'='*60}")
total_signals = 0
for ticker in TICKERS:
    label_path = os.path.join(ST_CACHE_DIR, f'{ticker}_strategy_labels.parquet')
    feat_path  = os.path.join(ST_CACHE_DIR, f'{ticker}_strategy_features.parquet')
    if not os.path.exists(label_path):
        continue
    ldf   = pd.read_parquet(label_path)
    total = int((ldf['signal_label'] > 0).sum())
    if total == 0:
        print(f'  {ticker}: no labels')
        continue
    sig    = ldf[ldf['signal_label'] > 0]
    wr_2r  = (sig['max_rr'] >= 2.0).sum() / total
    wr_3r  = (sig['max_rr'] >= 3.0).sum() / total
    total_signals += total
    dir_str = ''
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        if len(fdf) == len(ldf):
            dirs = fdf.loc[ldf['signal_label'] > 0, 'st_direction']
            dir_str = f' | L:{(dirs > 0).sum()} S:{(dirs < 0).sum()}'
    baseline = BASELINE_WR.get(ticker, 0.70)
    status   = 'OK' if wr_2r >= baseline - 0.03 else 'LOW'
    print(f'  [{status}] {ticker}: {total} signals{dir_str} | '
          f'WR@2R:{wr_2r*100:.1f}% WR@3R:{wr_3r*100:.1f}% | baseline:{baseline*100:.0f}%')
print(f'  TOTAL: {total_signals} signals')
print(f"{'='*60}")


# ==============================================================================
# CELL 4 — WALK-FORWARD HYBRID FINE-TUNING
# ==============================================================================

import gc, importlib, json
import futures_foundation
importlib.reload(futures_foundation)
from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune import TrainingConfig, run_walk_forward, export_onnx, validate_setup

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_cfg = TrainingConfig(
    seq_len                = SEQ_LEN,
    batch_size             = BATCH_SIZE,
    sig_per_batch          = SIG_PER_BATCH,
    epochs                 = EPOCHS,
    lr                     = LR,
    freeze_ratio           = FREEZE_RATIO,
    risk_weight            = RISK_WEIGHT,
    miss_penalty           = MISS_PENALTY,
    false_penalty          = FALSE_PENALTY,
    focal_gamma            = FOCAL_GAMMA,
    focal_smoothing        = FOCAL_SMOOTHING,
    patience               = PATIENCE,
    max_ratio              = MAX_RATIO,
    ratio_patience         = RATIO_PATIENCE,
    num_labels             = NUM_LABELS,
    warm_start_mode        = WARM_START_MODE,
    backbone_lr_multiplier = BACKBONE_LR_MULTIPLIER,
    baseline_wr            = BASELINE_WR,
)

ffm_config = FFMConfig(
    num_features    = len(get_model_feature_columns()),
    label_smoothing = FOCAL_SMOOTHING,
)

validate_setup(
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = ST_CACHE_DIR,
    backbone_path         = BACKBONE_PATH,
    strategy_feature_cols = ST_FEATURE_COLS,
    num_strategy_features = NUM_ST_FEATURES,
    micro_to_full         = MICRO_TO_FULL,
)

fold_results = run_walk_forward(
    folds                 = FOLDS,
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = ST_CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    backbone_path         = BACKBONE_PATH,
    ffm_config            = ffm_config,
    training_cfg          = training_cfg,
    num_strategy_features = NUM_ST_FEATURES,
    strategy_feature_cols = ST_FEATURE_COLS,
    micro_to_full         = MICRO_TO_FULL,
)

# ── ONNX export (F4 model — production deployment) ──
last_model = fold_results.get('_model')
if last_model is not None:
    print(f'\n{"="*60}\n  ONNX EXPORT\n{"="*60}')
    onnx_path = os.path.join(OUTPUT_DIR, 'st_trend_v1.onnx')
    meta_path = onnx_path.replace('.onnx', '_metadata.json')
    try:
        export_onnx(
            last_model, onnx_path,
            seq_len               = SEQ_LEN,
            num_ffm_features      = len(get_model_feature_columns()),
            num_strategy_features = NUM_ST_FEATURES,
        )

        import torch as _torch
        wf_results = {}
        for fname, metrics in fold_results.items():
            if fname == '_model' or metrics is None:
                continue
            confs  = _torch.tensor(metrics['all_conf'])
            labels = _torch.tensor(metrics['all_labels'])
            entry  = {'signals': int((labels > 0).sum())}
            for key, thr in [('prec_at_70', 0.70), ('prec_at_80', 0.80), ('prec_at_90', 0.90)]:
                mask = confs >= thr
                entry[key] = round(float((labels[mask] > 0).float().mean()), 3) if mask.sum() > 0 else None
            wf_results[fname] = entry

        metadata = {
            'version':          'st_trend_v1_0',
            'seq_len':          SEQ_LEN,
            'num_labels':       NUM_LABELS,
            'num_features':     len(get_model_feature_columns()),
            'num_st_features':  NUM_ST_FEATURES,
            'feature_cols':     get_model_feature_columns(),
            'st_feature_cols':  ST_FEATURE_COLS,
            'tickers':          TICKERS,
            'labeling': {
                'st_period':    ST_PERIOD,
                'st_mult':      ST_MULT,
                'htf_period':   HTF_PERIOD,
                'htf_mult':     HTF_MULT,
                'lookahead':    LOOKAHEAD,
                'sl_atr_mult':  SL_ATR_MULT,
            },
            'inference': {
                'output_0':   '[B,2] signal_logits — softmax; index 1 = signal probability',
                'output_1':   '[B]   confidence — max(softmax), range [0.5, 1.0]',
                'output_2':   '[B,1] risk_predictions — predicted R:R',
                'direction':  'st_feature_cols index 0 (st_direction): >0=LONG, <=0=SHORT',
                'sl_sizing':  'sl_distance = ATR × 1.5',
                'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
                'entry_rule': 'softmax(signal_logits)[:,1] >= threshold at 5min ST flip bars',
            },
            'walk_forward_results': wf_results,
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'  Metadata: {meta_path}')
    except Exception as e:
        print(f'  ONNX export failed: {e}')


# ==============================================================================
# CELL 5 — WALK-FORWARD EVALUATION SUMMARY
# ==============================================================================

from futures_foundation.finetune import print_eval_summary
print_eval_summary(fold_results, baseline_wr=BASELINE_WR, output_dir=OUTPUT_DIR)
