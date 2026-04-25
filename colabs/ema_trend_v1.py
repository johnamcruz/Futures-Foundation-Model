# ==============================================================================
# EMA TREND FOLLOW — HYBRID FINE-TUNING v1.0
# ==============================================================================
# Strategy: EMA compression → expansion transitions on 5-min bars.
# FFM backbone (67 features × 96 bars) provides all market context — chop
# detection, HTF structure, volume, session, volatility regime. These 8 strategy
# features give the model only what is unique: EMA geometry at the entry bar.
#
# Phase 1 (this script): signal classifier — noise vs compression→expansion signal
# Phase 2 (risk head):   use cisd_ote_v71_riskhead.py pattern on this checkpoint
#
# EMA config (5-min only — no 1-hour component; warmup: 150 bars ≈ 12h):
#   Fast: [9, 20, 50, 100]   Medium: [20, 50, 100, 150]
#
# Signal: compression→expansion transition, fast + medium aligned, both expanding
# SL: entry ± SL_ATR_MULT × ATR (ATR from FFM parquet, same normalisation as backbone)
# max_rr: best RR reached before stop or HORIZON_BARS expire
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
os.system('pip install onnxscript -q 2>&1 | tail -1')

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
RAW_DATA_DIR  = '/content/drive/MyDrive/Futures Data/5min'
PREPARED_DIR  = '/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared'
BACKBONE_PATH = '/content/drive/MyDrive/AI_Cache/5min_FFM_Checkpoints/best_backbone.pt'
EMA_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/EMA_Trend_Labels_v1'
OUTPUT_DIR    = '/content/drive/MyDrive/AI_Models/EMA_Trend_v1'

# ── TICKERS ──
DATA_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC']
MICRO_TO_FULL = {
    'MES': 'ES', 'MNQ': 'NQ', 'MRTY': 'RTY', 'MYM': 'YM', 'MGC': 'GC',
}
TICKERS = DATA_TICKERS

# ── EMA PARAMS ──
FAST_PERIODS    = [9, 20, 50, 100]
MED_PERIODS     = [20, 50, 100, 150]
MIN_WARMUP_BARS = 200   # skip detection until EMAs stabilise (~12h of 5min bars)
MIN_COMP_BARS   = 3     # minimum consecutive bars in compression before signal fires

# ── COMPRESSION / EXPANSION THRESHOLDS (ATR-normalised) ──
# Spread = (EMA_fast - EMA_slow) / ATR. Tune if signal count is outside 500–5000/ticker.
COMPRESSION_THR_FAST = 0.5   # |spread_fast| < thr → compressed
EXPANSION_THR_FAST   = 1.5   # |spread_fast| > thr → expanding
COMPRESSION_THR_MED  = 0.6
EXPANSION_THR_MED    = 1.8

# ── RISK / LABELING ──
SL_ATR_MULT   = 1.5    # stop = entry ± SL_ATR_MULT × ATR
HORIZON_BARS  = 48     # max forward scan for max_rr (48 × 5min = 4 hours)
RR_TARGETS    = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
MIN_SIGNAL_RR = 1.0    # discard signals that can't reach 1R before stopping out

# ── SIGNAL CLASS ──
NUM_LABELS = 2

# ── TRAINING ──
SEQ_LEN                = 96
BATCH_SIZE             = 256
EPOCHS                 = 40
LR                     = 5e-5
FREEZE_RATIO           = 0.66
RISK_WEIGHT            = 0.1
MISS_PENALTY           = 1.0
FALSE_PENALTY          = 1.0
SIG_PER_BATCH          = 8
SIGNAL_RATIO           = SIG_PER_BATCH / BATCH_SIZE
FOCAL_GAMMA            = 1.0
FOCAL_SMOOTHING        = 0.10
PATIENCE               = 15
MAX_RATIO              = 2.5
RATIO_PATIENCE         = 8
WARM_START_MODE        = 'selective'
BACKBONE_LR_MULTIPLIER = 0.1

# ── MECHANICAL BASELINE WR @ 2R (tune after first label run) ──
BASELINE_WR = {
    'ES': 0.35, 'NQ': 0.35, 'RTY': 0.35, 'YM': 0.35, 'GC': 0.35,
    'MES': 0.35, 'MNQ': 0.35, 'MRTY': 0.35, 'MYM': 0.35, 'MGC': 0.35,
}

# ── WALK-FORWARD FOLDS ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── EMA TREND FEATURES (8) ──
# Only EMA geometry unique to this strategy. FFM backbone already captures
# all market context: momentum, volume, session, HTF structure, volatility regime.
EMA_FEATURE_COLS = [
    'fast_spread_norm',      # (EMA9 − EMA100) / ATR  — signed: + bull, − bear
    'med_spread_norm',       # (EMA20 − EMA150) / ATR
    'fast_alignment',        # +1 fully bull (9>20>50>100), −1 fully bear, 0 mixed
    'med_alignment',         # +1 fully bull (20>50>100>150), −1 fully bear, 0 mixed
    'compression_bars_norm', # bars in prior compression state / 50 (clipped 0–5)
    'expansion_delta',       # spread growth rate over last 5 bars (ATR-normalised)
    'price_vs_ema9_norm',    # (close − EMA9) / ATR — entry pullback depth
    'mtf_agreement',         # +1 both bull, −1 both bear, 0 disagreement
]
NUM_EMA_FEATURES = len(EMA_FEATURE_COLS)


# ==============================================================================
# CELL 3 — EMA LABELER
# ==============================================================================

import os, time
import numpy as np
import pandas as pd
from futures_foundation.finetune import StrategyLabeler, run_labeling


def _ema(close_arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(close_arr).ewm(span=period, adjust=False).mean().values


def label_ema_trend_signals(df_5m, atr_arr):
    """
    Detect compression→expansion transitions on 5-min bars.

    Signal fires when:
      - Both fast and medium EMAs were compressed for >= MIN_COMP_BARS bars
      - Both cross into expansion simultaneously
      - Fast and medium EMA ordering agree on direction (both fully bull or both fully bear)

    Returns:
      signal_labels  int8[n]     0=noise, 1=BUY, 2=SELL
      max_rr_arr     float32[n]  best RR reached before stop
      sl_dist_arr    float32[n]  stop distance in price points
      feature_arrays dict[str→float32[n]]
      stats          dict
    """
    n     = len(df_5m)
    close = df_5m['close'].values.astype(np.float64)
    high  = df_5m['high'].values.astype(np.float64)
    low   = df_5m['low'].values.astype(np.float64)
    atr   = np.where(atr_arr > 0, atr_arr, 1e-6).astype(np.float64)

    # EMAs
    e9   = _ema(close, 9)
    e20  = _ema(close, 20)
    e50  = _ema(close, 50)
    e100 = _ema(close, 100)
    e150 = _ema(close, 150)
    # e20, e50, e100 shared between fast and medium sets

    spread_fast = (e9  - e100) / atr   # signed, ATR-normalised
    spread_med  = (e20 - e150) / atr
    abs_fast    = np.abs(spread_fast)
    abs_med     = np.abs(spread_med)

    fast_bull  = (e9 > e20)  & (e20 > e50)  & (e50 > e100)
    fast_bear  = (e9 < e20)  & (e20 < e50)  & (e50 < e100)
    fast_align = np.where(fast_bull, 1.0, np.where(fast_bear, -1.0, 0.0)).astype(np.float32)

    med_bull   = (e20 > e50) & (e50 > e100) & (e100 > e150)
    med_bear   = (e20 < e50) & (e50 < e100) & (e100 < e150)
    med_align  = np.where(med_bull, 1.0, np.where(med_bear, -1.0, 0.0)).astype(np.float32)

    # Per-bar feature arrays (populated for all bars, not just signals)
    compression_bars_norm = np.zeros(n, dtype=np.float32)
    expansion_delta       = np.zeros(n, dtype=np.float32)
    price_vs_ema9         = ((close - e9) / atr).astype(np.float32)
    mtf_agreement         = np.zeros(n, dtype=np.float32)

    signal_labels = np.zeros(n, dtype=np.int8)
    sl_dist_arr   = np.zeros(n, dtype=np.float32)
    comp_fast = comp_med = 0
    stats = {'buys': 0, 'sells': 0, 'trades': 0}

    for i in range(1, n):
        # Update compression counters from previous bar
        comp_fast = (comp_fast + 1) if abs_fast[i-1] < COMPRESSION_THR_FAST else 0
        comp_med  = (comp_med  + 1) if abs_med[i-1]  < COMPRESSION_THR_MED  else 0

        compression_bars_norm[i] = float(np.clip(min(comp_fast, comp_med) / 50.0, 0.0, 5.0))

        lb = min(i, 5)
        expansion_delta[i] = float((abs_fast[i] - abs_fast[i - lb]) / lb)

        fa = fast_align[i]; ma = med_align[i]
        mtf_agreement[i] = fa if (fa != 0.0 and fa == ma) else 0.0

        if i < MIN_WARMUP_BARS:
            continue

        was_compressed  = comp_fast >= MIN_COMP_BARS and comp_med >= MIN_COMP_BARS
        now_expanding   = abs_fast[i] > EXPANSION_THR_FAST and abs_med[i] > EXPANSION_THR_MED
        direction       = int(fa)
        direction_agree = direction != 0 and direction == int(ma)

        if was_compressed and now_expanding and direction_agree:
            signal_labels[i] = 1 if direction > 0 else 2
            sl_dist_arr[i]   = float(atr[i] * SL_ATR_MULT)
            stats['trades'] += 1
            stats['buys' if direction > 0 else 'sells'] += 1

    # Compute max_rr for each signal bar
    max_rr_arr = np.zeros(n, dtype=np.float32)
    for i in range(MIN_WARMUP_BARS, n):
        if signal_labels[i] == 0:
            continue
        is_long  = signal_labels[i] == 1
        entry    = close[i]
        sl_dist  = float(sl_dist_arr[i])
        if sl_dist <= 0:
            signal_labels[i] = 0
            continue
        sl_price = entry - sl_dist if is_long else entry + sl_dist
        best_rr  = 0.0
        for j in range(i + 1, min(i + HORIZON_BARS + 1, n)):
            if is_long:
                if low[j] <= sl_price:
                    break
                best_rr = max(best_rr, (high[j] - entry) / sl_dist)
            else:
                if high[j] >= sl_price:
                    break
                best_rr = max(best_rr, (entry - low[j]) / sl_dist)
        max_rr_arr[i] = best_rr

    # Discard signals that never reached MIN_SIGNAL_RR
    signal_labels[max_rr_arr < MIN_SIGNAL_RR] = 0

    feature_arrays = {
        'fast_spread_norm':      spread_fast.astype(np.float32),
        'med_spread_norm':       spread_med.astype(np.float32),
        'fast_alignment':        fast_align,
        'med_alignment':         med_align,
        'compression_bars_norm': compression_bars_norm,
        'expansion_delta':       expansion_delta,
        'price_vs_ema9_norm':    price_vs_ema9,
        'mtf_agreement':         mtf_agreement,
    }
    return signal_labels, max_rr_arr, sl_dist_arr, feature_arrays, stats


def _create_ema_feature_matrix(feature_arrays):
    mat = np.column_stack([feature_arrays[c] for c in EMA_FEATURE_COLS])
    return np.nan_to_num(np.clip(mat, -10.0, 10.0), nan=0.0).astype(np.float32)


class EMATrendLabeler(StrategyLabeler):
    """EMA compression → expansion trend-following labeler for FFM fine-tuning."""

    @property
    def name(self):
        return 'ema_trend'

    @property
    def feature_cols(self):
        return EMA_FEATURE_COLS

    def run(self, df_raw, ffm_df, ticker):
        # Use ATR from FFM parquet — same normalisation as backbone training
        atr_arr = (ffm_df['vty_atr_raw']
                   .reindex(df_raw.index, fill_value=np.nan)
                   .values.astype(np.float64))
        atr_arr = pd.Series(atr_arr).ffill().bfill().values

        sig_arr, rr_arr, sl_arr, feat_arrs, stats = label_ema_trend_signals(df_raw, atr_arr)
        print(f'  Labels: {stats["buys"]}B + {stats["sells"]}S '
              f'({stats["trades"]} raw, {(sig_arr > 0).sum()} after RR filter)')

        feat_mat = _create_ema_feature_matrix(feat_arrs)

        sig_s   = pd.Series(sig_arr.astype(np.int8),   index=df_raw.index)
        rr_s    = pd.Series(rr_arr.astype(np.float32), index=df_raw.index)
        sl_s    = pd.Series(sl_arr.astype(np.float32), index=df_raw.index)
        feats_s = pd.DataFrame(feat_mat, index=df_raw.index, columns=EMA_FEATURE_COLS)

        aligned_sig   = (sig_s.reindex(ffm_df.index).fillna(0).values > 0).astype(np.int8)
        aligned_rr    = rr_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_sl    = sl_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_feats = feats_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)

        labels_df = pd.DataFrame({
            'signal_label': aligned_sig,
            'max_rr':       aligned_rr,
            'sl_distance':  aligned_sl,
        })
        feats_df = pd.DataFrame(aligned_feats, columns=EMA_FEATURE_COLS)
        return feats_df, labels_df


labeler = EMATrendLabeler()
run_labeling(labeler, TICKERS, RAW_DATA_DIR, PREPARED_DIR, EMA_CACHE_DIR,
             micro_to_full=MICRO_TO_FULL)

# ── Label diagnostic ──
print(f"\n{'='*60}\n  📊 LABEL DIAGNOSTIC\n{'='*60}")
print(f"  Thresholds — Compression fast:{COMPRESSION_THR_FAST} med:{COMPRESSION_THR_MED} "
      f"| Expansion fast:{EXPANSION_THR_FAST} med:{EXPANSION_THR_MED}")
for ticker in TICKERS:
    label_path = os.path.join(EMA_CACHE_DIR, f'{ticker}_strategy_labels.parquet')
    feat_path  = os.path.join(EMA_CACHE_DIR, f'{ticker}_strategy_features.parquet')
    if not os.path.exists(label_path):
        continue
    ldf   = pd.read_parquet(label_path)
    total = int((ldf['signal_label'] > 0).sum())
    if total == 0:
        print(f'  ⚠️  {ticker}: 0 signals — lower EXPANSION_THR or raise COMPRESSION_THR')
        continue
    sig      = ldf[ldf['signal_label'] > 0]
    wr_2r    = float((sig['max_rr'] >= 2.0).sum() / total)
    wr_3r    = float((sig['max_rr'] >= 3.0).sum() / total)
    baseline = BASELINE_WR.get(ticker, 0.35)
    status   = '✅' if wr_2r >= baseline - 0.05 else '⚠️ '
    bull_cnt = bear_cnt = 0
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        if len(fdf) == len(ldf):
            fa = fdf.loc[ldf['signal_label'] > 0, 'fast_alignment']
            bull_cnt = int((fa > 0).sum()); bear_cnt = int((fa < 0).sum())
    print(f'  {status} {ticker}: {total} signals | Bull:{bull_cnt} Bear:{bear_cnt} | '
          f'WR@2R:{wr_2r*100:.1f}% WR@3R:{wr_3r*100:.1f}% | baseline:{baseline*100:.0f}%')
print(f"{'='*60}")
print("  ⚙️  Threshold tuning guide:")
print("  Too few signals  → lower EXPANSION_THR (e.g. 1.5→1.2) or lower COMPRESSION_THR")
print("  WR@2R near 100%  → raise EXPANSION_THR (signals fire too early / in noise)")
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
    strategy_dir          = EMA_CACHE_DIR,
    backbone_path         = BACKBONE_PATH,
    strategy_feature_cols = EMA_FEATURE_COLS,
    num_strategy_features = NUM_EMA_FEATURES,
    micro_to_full         = MICRO_TO_FULL,
)

fold_results = run_walk_forward(
    folds                 = FOLDS,
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = EMA_CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    backbone_path         = BACKBONE_PATH,
    ffm_config            = ffm_config,
    training_cfg          = training_cfg,
    num_strategy_features = NUM_EMA_FEATURES,
    strategy_feature_cols = EMA_FEATURE_COLS,
    micro_to_full         = MICRO_TO_FULL,
)

# ── ONNX export (F4 — production deployment) ──
last_model = fold_results.get('_model')
if last_model is not None:
    print(f'\n{"="*60}\n  ONNX EXPORT\n{"="*60}')
    onnx_path = os.path.join(OUTPUT_DIR, 'ema_trend_hybrid.onnx')
    meta_path = onnx_path.replace('.onnx', '_metadata.json')
    try:
        export_onnx(last_model, onnx_path,
                    seq_len               = SEQ_LEN,
                    num_ffm_features      = len(get_model_feature_columns()),
                    num_strategy_features = NUM_EMA_FEATURES)

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
            'version':            'ema_trend_v1_0',
            'seq_len':            SEQ_LEN,
            'num_labels':         NUM_LABELS,
            'num_features':       len(get_model_feature_columns()),
            'num_ema_features':   NUM_EMA_FEATURES,
            'feature_cols':       get_model_feature_columns(),
            'ema_feature_cols':   EMA_FEATURE_COLS,
            'tickers':            TICKERS,
            'ema_params': {
                'fast_periods':        FAST_PERIODS,
                'med_periods':         MED_PERIODS,
                'compression_thr_fast': COMPRESSION_THR_FAST,
                'expansion_thr_fast':   EXPANSION_THR_FAST,
                'compression_thr_med':  COMPRESSION_THR_MED,
                'expansion_thr_med':    EXPANSION_THR_MED,
                'sl_atr_mult':          SL_ATR_MULT,
                'horizon_bars':         HORIZON_BARS,
            },
            'inference': {
                'output_0': '[B,2] signal_logits — softmax; index 1 = signal probability',
                'output_1': '[B,1] risk_predictions — predicted R:R (Softplus)',
                'output_2': '[B]   confidence — max(softmax), range [0.5, 1.0]',
                'direction': 'ema_feature_cols index 2 (fast_alignment): >0=BUY, <0=SELL',
                'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
            },
            'walk_forward_results': wf_results,
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'  ✅ ONNX:     {onnx_path}')
        print(f'  ✅ Metadata: {meta_path}')
    except Exception as e:
        print(f'  ❌ ONNX export failed: {e}')


# ==============================================================================
# CELL 5 — WALK-FORWARD EVALUATION SUMMARY
# ==============================================================================

from futures_foundation.finetune import print_eval_summary
print_eval_summary(fold_results, baseline_wr=BASELINE_WR, output_dir=OUTPUT_DIR)
