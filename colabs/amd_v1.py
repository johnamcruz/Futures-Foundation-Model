# ==============================================================================
# AMD (Accumulation / Manipulation / Distribution) — HYBRID FINE-TUNING v1.0
# ==============================================================================
# Strategy: session-based 3-phase setup (ICT AMD concept)
#   Accumulation  — Asian session (00:00–05:00 UTC) defines range
#   Manipulation  — London session (07:00–12:00 UTC) sweeps one side of range
#   Distribution  — NY session (13:00–18:00 UTC) delivers to opposite side
#
# Entry: first NY bar where price re-enters the Asian range after sweep
# SL:    1.5 × ATR from entry (data-validated; structural SL averaged 7–9× ATR)
# max_rr: scan forward 48 bars (4 hours)
#
# Data-validated mechanical edge (5min bars, 1.5× ATR SL):
#   NQ: 70.7% WR@2R, avgRR 5.78 over 5yr (270 signals)
#   GC: 70.5% WR@2R, avgRR 9.20 over 4yr (173 signals)
#
# Phase 1 (this script): signal classifier — noise vs AMD signal
# Phase 2 (risk head):   adapt cisd_ote_v71_riskhead.py with AMD_FEATURE_COLS
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
AMD_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/AMD_Labels_v1'
OUTPUT_DIR    = '/content/drive/MyDrive/AI_Models/AMD_v1'

# ── TICKERS ──
DATA_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC']
MICRO_TO_FULL = {
    'MES': 'ES', 'MNQ': 'NQ', 'MRTY': 'RTY', 'MYM': 'YM', 'MGC': 'GC',
}
TICKERS = DATA_TICKERS

# ── SESSION WINDOWS (UTC hours, covers both EST and EDT across DST) ──
ASIAN_START   = 0    # 00:00 UTC (~8pm ET)
ASIAN_END     = 5    # 05:00 UTC (~midnight ET)
LONDON_START  = 7    # 07:00 UTC (~2am ET)
LONDON_END    = 12   # 12:00 UTC (~7am ET)
NY_START      = 13   # 13:00 UTC (~8:30am ET)
NY_END        = 18   # 18:00 UTC (~1pm ET)

# ── AMD PARAMS ──
MIN_SWEEP_ATR     = 0.25  # minimum sweep depth beyond Asian level (ATR multiples)
MIN_ASIAN_BARS    = 3     # minimum bars required to define Asian range
MIN_LONDON_BARS   = 2     # minimum London bars required
MIN_NY_BARS       = 4     # minimum NY bars required for entry

# ── RISK / LABELING ──
SL_ATR_MULT   = 1.5   # data-validated: structural SL (7–9× ATR) hurts WR badly
HORIZON_BARS  = 48    # max forward scan for max_rr (48 × 5min = 4 hours)
MIN_SIGNAL_RR = 1.0   # discard signals that never reach 1R

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

# ── MECHANICAL BASELINE WR @ 2R (data-validated on NQ/GC; others conservative) ──
# NQ: 70.7% WR@2R over 5yr | GC: 70.5% WR@2R over 4yr
BASELINE_WR = {
    'ES': 0.65, 'NQ': 0.70, 'RTY': 0.65, 'YM': 0.65, 'GC': 0.70,
    'MES': 0.65, 'MNQ': 0.70, 'MRTY': 0.65, 'MYM': 0.65, 'MGC': 0.70,
}

# ── WALK-FORWARD FOLDS ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── AMD FEATURES (8) ──
# Session geometry unique to this strategy. FFM backbone provides all market
# context (momentum, volume, volatility, regime). These features give the model
# the AMD-specific price levels and sweep mechanics at the entry bar.
AMD_FEATURE_COLS = [
    'asian_high_dist',      # (close − asian_high) / ATR  — position vs Asian high
    'asian_low_dist',       # (close − asian_low)  / ATR  — position vs Asian low
    'asian_range_atr',      # (asian_high − asian_low) / ATR  — range size
    'sweep_direction',      # +1 swept low (LONG setup), −1 swept high (SHORT setup)
    'sweep_depth_atr',      # depth London swept beyond Asian level / ATR
    'price_vs_asian_mid',   # (close − asian_midpoint) / ATR
    'time_since_sweep_norm',# bars since London sweep / 48 (clipped 0–1)
    'london_range_atr',     # London session range / ATR  — manipulation strength
]
NUM_AMD_FEATURES = len(AMD_FEATURE_COLS)


# ==============================================================================
# CELL 3 — AMD LABELER
# ==============================================================================

import os, time
import numpy as np
import pandas as pd
from futures_foundation.finetune import StrategyLabeler, run_labeling


def label_amd_signals(df_5m, atr_arr):
    """
    Detect AMD (Accumulation/Manipulation/Distribution) session setups.

    Per-bar AMD features are populated for ALL bars during the NY session on days
    with a confirmed London sweep, giving the FFM backbone full session context
    even for non-signal bars. Features are 0 outside active AMD windows.

    Signal fires once per day at the first NY bar where price re-enters the
    Asian range after a London sweep:
      - London swept Asian LOW  → LONG  (distribution higher)
      - London swept Asian HIGH → SHORT (distribution lower)

    Data-validated: NQ 70.7% WR@2R, GC 70.5% WR@2R (1.5× ATR SL, 5yr data)

    Returns:
      signal_labels  int8[n]     0=noise, 1=LONG, 2=SHORT
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
    hours = df_5m.index.hour
    dates = df_5m.index.date

    # Per-bar feature arrays
    asian_high_dist       = np.zeros(n, dtype=np.float32)
    asian_low_dist        = np.zeros(n, dtype=np.float32)
    asian_range_atr       = np.zeros(n, dtype=np.float32)
    sweep_direction       = np.zeros(n, dtype=np.float32)
    sweep_depth_atr       = np.zeros(n, dtype=np.float32)
    price_vs_asian_mid    = np.zeros(n, dtype=np.float32)
    time_since_sweep_norm = np.zeros(n, dtype=np.float32)
    london_range_atr      = np.zeros(n, dtype=np.float32)

    signal_labels = np.zeros(n, dtype=np.int8)
    sl_dist_arr   = np.zeros(n, dtype=np.float32)
    stats = {'longs': 0, 'shorts': 0, 'no_sweep': 0, 'no_entry': 0}

    for date in pd.Series(dates).unique():
        day_mask = dates == date

        # ── Asian range ──
        asian_mask = day_mask & (hours >= ASIAN_START) & (hours < ASIAN_END)
        asian_idx  = np.where(asian_mask)[0]
        if len(asian_idx) < MIN_ASIAN_BARS:
            continue
        a_high   = high[asian_idx].max()
        a_low    = low[asian_idx].min()
        a_mid    = (a_high + a_low) / 2.0
        a_range  = a_high - a_low
        atr_ref  = float(atr[asian_idx[-1]])
        if atr_ref <= 0 or a_range <= 0:
            continue

        # ── London sweep ──
        lon_mask = day_mask & (hours >= LONDON_START) & (hours < LONDON_END)
        lon_idx  = np.where(lon_mask)[0]
        if len(lon_idx) < MIN_LONDON_BARS:
            stats['no_sweep'] += 1
            continue
        lon_high = high[lon_idx].max()
        lon_low  = low[lon_idx].min()
        lon_range = lon_high - lon_low

        swept_high = lon_high > a_high + MIN_SWEEP_ATR * atr_ref
        swept_low  = lon_low  < a_low  - MIN_SWEEP_ATR * atr_ref

        if swept_high == swept_low:   # both swept or neither
            stats['no_sweep'] += 1
            continue

        is_long     = swept_low
        sweep_dir   = 1.0 if is_long else -1.0
        sweep_ext   = lon_low if is_long else lon_high
        s_depth     = abs(sweep_ext - (a_low if is_long else a_high)) / atr_ref

        # Find the London sweep bar (first bar that crossed the level)
        sweep_bar = None
        for k in lon_idx:
            if is_long  and low[k]  < a_low  - MIN_SWEEP_ATR * atr_ref:
                sweep_bar = k; break
            if not is_long and high[k] > a_high + MIN_SWEEP_ATR * atr_ref:
                sweep_bar = k; break
        if sweep_bar is None:
            sweep_bar = lon_idx[0]

        # ── NY entry ──
        ny_mask = day_mask & (hours >= NY_START) & (hours < NY_END)
        ny_idx  = np.where(ny_mask)[0]
        if len(ny_idx) < MIN_NY_BARS:
            stats['no_entry'] += 1
            continue

        # Populate AMD features for all NY bars on this day (gives backbone context)
        for i in ny_idx:
            asian_high_dist[i]    = float((close[i] - a_high) / atr[i])
            asian_low_dist[i]     = float((close[i] - a_low)  / atr[i])
            asian_range_atr[i]    = float(a_range / atr[i])
            sweep_direction[i]    = sweep_dir
            sweep_depth_atr[i]    = float(s_depth)
            price_vs_asian_mid[i] = float((close[i] - a_mid) / atr[i])
            time_since_sweep_norm[i] = float(np.clip((i - sweep_bar) / 48.0, 0.0, 1.0))
            london_range_atr[i]   = float(lon_range / atr[i])

        # Entry: first NY bar where price re-enters the Asian range
        entry_bar = None
        for k in ny_idx:
            if is_long  and close[k] > a_low:
                entry_bar = k; break
            if not is_long and close[k] < a_high:
                entry_bar = k; break
        if entry_bar is None:
            entry_bar = ny_idx[0]   # fallback to first NY bar

        signal_labels[entry_bar] = 1 if is_long else 2
        sl_dist_arr[entry_bar]   = float(atr[entry_bar] * SL_ATR_MULT)
        stats['longs' if is_long else 'shorts'] += 1

    # ── Compute max_rr ──
    max_rr_arr = np.zeros(n, dtype=np.float32)
    for i in range(n):
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
        'asian_high_dist':       asian_high_dist,
        'asian_low_dist':        asian_low_dist,
        'asian_range_atr':       asian_range_atr,
        'sweep_direction':       sweep_direction,
        'sweep_depth_atr':       sweep_depth_atr,
        'price_vs_asian_mid':    price_vs_asian_mid,
        'time_since_sweep_norm': time_since_sweep_norm,
        'london_range_atr':      london_range_atr,
    }
    return signal_labels, max_rr_arr, sl_dist_arr, feature_arrays, stats


def _create_amd_feature_matrix(feature_arrays):
    mat = np.column_stack([feature_arrays[c] for c in AMD_FEATURE_COLS])
    return np.nan_to_num(np.clip(mat, -10.0, 10.0), nan=0.0).astype(np.float32)


class AMDLabeler(StrategyLabeler):
    """
    AMD session labeler: Asian accumulation → London manipulation sweep →
    NY distribution reversal.
    """

    @property
    def name(self):
        return 'amd'

    @property
    def feature_cols(self):
        return AMD_FEATURE_COLS

    def run(self, df_raw, ffm_df, ticker):
        # Use ATR from FFM parquet — same normalisation as backbone
        atr_arr = (ffm_df['vty_atr_raw']
                   .reindex(df_raw.index, fill_value=np.nan)
                   .values.astype(np.float64))
        atr_arr = pd.Series(atr_arr).ffill().bfill().values

        sig_arr, rr_arr, sl_arr, feat_arrs, stats = label_amd_signals(df_raw, atr_arr)
        after_rr = int((sig_arr > 0).sum())
        print(f'  Labels: {stats["longs"]}L + {stats["shorts"]}S '
              f'({stats["longs"]+stats["shorts"]} raw → {after_rr} after RR filter) '
              f'| no_sweep:{stats["no_sweep"]} no_entry:{stats["no_entry"]}')

        feat_mat = _create_amd_feature_matrix(feat_arrs)

        sig_s   = pd.Series(sig_arr.astype(np.int8),   index=df_raw.index)
        rr_s    = pd.Series(rr_arr.astype(np.float32), index=df_raw.index)
        sl_s    = pd.Series(sl_arr.astype(np.float32), index=df_raw.index)
        feats_s = pd.DataFrame(feat_mat, index=df_raw.index, columns=AMD_FEATURE_COLS)

        aligned_sig   = (sig_s.reindex(ffm_df.index).fillna(0).values > 0).astype(np.int8)
        aligned_rr    = rr_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_sl    = sl_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)
        aligned_feats = feats_s.reindex(ffm_df.index).fillna(0.0).values.astype(np.float32)

        labels_df = pd.DataFrame({
            'signal_label': aligned_sig,
            'max_rr':       aligned_rr,
            'sl_distance':  aligned_sl,
        })
        feats_df = pd.DataFrame(aligned_feats, columns=AMD_FEATURE_COLS)
        return feats_df, labels_df


labeler = AMDLabeler()
run_labeling(labeler, TICKERS, RAW_DATA_DIR, PREPARED_DIR, AMD_CACHE_DIR,
             micro_to_full=MICRO_TO_FULL)

# ── Label diagnostic ──
print(f"\n{'='*60}\n  📊 LABEL DIAGNOSTIC\n{'='*60}")
for ticker in TICKERS:
    label_path = os.path.join(AMD_CACHE_DIR, f'{ticker}_strategy_labels.parquet')
    feat_path  = os.path.join(AMD_CACHE_DIR, f'{ticker}_strategy_features.parquet')
    if not os.path.exists(label_path):
        continue
    ldf   = pd.read_parquet(label_path)
    total = int((ldf['signal_label'] > 0).sum())
    if total == 0:
        print(f'  ⚠️  {ticker}: 0 signals')
        continue
    sig      = ldf[ldf['signal_label'] > 0]
    wr_2r    = float((sig['max_rr'] >= 2.0).sum() / total)
    wr_3r    = float((sig['max_rr'] >= 3.0).sum() / total)
    baseline = BASELINE_WR.get(ticker, 0.65)
    status   = '✅' if wr_2r >= baseline - 0.05 else '⚠️ '
    long_cnt = short_cnt = 0
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        if len(fdf) == len(ldf):
            sd = fdf.loc[ldf['signal_label'] > 0, 'sweep_direction']
            long_cnt = int((sd > 0).sum()); short_cnt = int((sd < 0).sum())
    print(f'  {status} {ticker}: {total} signals | Long:{long_cnt} Short:{short_cnt} | '
          f'WR@2R:{wr_2r*100:.1f}% WR@3R:{wr_3r*100:.1f}% | baseline:{baseline*100:.0f}%')
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
    strategy_dir          = AMD_CACHE_DIR,
    backbone_path         = BACKBONE_PATH,
    strategy_feature_cols = AMD_FEATURE_COLS,
    num_strategy_features = NUM_AMD_FEATURES,
    micro_to_full         = MICRO_TO_FULL,
)

fold_results = run_walk_forward(
    folds                 = FOLDS,
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = AMD_CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    backbone_path         = BACKBONE_PATH,
    ffm_config            = ffm_config,
    training_cfg          = training_cfg,
    num_strategy_features = NUM_AMD_FEATURES,
    strategy_feature_cols = AMD_FEATURE_COLS,
    micro_to_full         = MICRO_TO_FULL,
)

# ── ONNX export (F4 — production deployment) ──
last_model = fold_results.get('_model')
if last_model is not None:
    print(f'\n{"="*60}\n  ONNX EXPORT\n{"="*60}')
    onnx_path = os.path.join(OUTPUT_DIR, 'amd_hybrid.onnx')
    meta_path = onnx_path.replace('.onnx', '_metadata.json')
    try:
        export_onnx(last_model, onnx_path,
                    seq_len               = SEQ_LEN,
                    num_ffm_features      = len(get_model_feature_columns()),
                    num_strategy_features = NUM_AMD_FEATURES)

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
            'version':           'amd_v1_0',
            'seq_len':           SEQ_LEN,
            'num_labels':        NUM_LABELS,
            'num_features':      len(get_model_feature_columns()),
            'num_amd_features':  NUM_AMD_FEATURES,
            'feature_cols':      get_model_feature_columns(),
            'amd_feature_cols':  AMD_FEATURE_COLS,
            'tickers':           TICKERS,
            'session_windows_utc': {
                'asian':  [ASIAN_START,  ASIAN_END],
                'london': [LONDON_START, LONDON_END],
                'ny':     [NY_START,     NY_END],
            },
            'amd_params': {
                'min_sweep_atr': MIN_SWEEP_ATR,
                'sl_atr_mult':   SL_ATR_MULT,
                'horizon_bars':  HORIZON_BARS,
            },
            'inference': {
                'output_0': '[B,2] signal_logits — softmax; index 1 = signal probability',
                'output_1': '[B,1] risk_predictions — predicted R:R (Softplus)',
                'output_2': '[B]   confidence — max(softmax), range [0.5, 1.0]',
                'direction': 'amd_feature_cols index 3 (sweep_direction): >0=LONG, <0=SHORT',
                'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
            },
            'mechanical_baseline': {'NQ': 0.707, 'GC': 0.705},
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
