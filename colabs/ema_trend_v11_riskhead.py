# ==============================================================================
# EMA TREND FOLLOW v1.1 — PHASE 2: RISK HEAD CALIBRATION
# ==============================================================================
# Loads v1.0 walk-forward checkpoints and fine-tunes ONLY the risk_head
# so predicted_rr can be used as a reliable TP target at trade entry.
#
# Requires: F1–F4 _done.pt checkpoints in OUTPUT_DIR (from ema_trend_v1.py)
# Requires: Labeled parquets in EMA_CACHE_DIR (from ema_trend_v1.py labeling)
#
# What is frozen:  backbone, strategy_projection, fusion, signal_head
# What is trained: risk_head only (~65K params vs 3.4M total)
# Loss:            Huber (robust to occasional 5R outlier signals)
# LR:              1e-5, epochs: 20 per fold, early stop patience: 5
#
# Output:  F{n}_{hash}_rr_done.pt per fold + ema_trend_hybrid_v11.onnx
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
    from futures_foundation import FFMConfig, get_model_feature_columns
    print(f'✅ FFM installed — {len(get_model_feature_columns())} features')
except (ImportError, ValueError) as e:
    print(f'⚠️  Import failed: {e}')
    print('🔄 Restarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ==============================================================================
# CELL 2 — CONFIGURATION
# ==============================================================================

import torch, copy, glob
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

# ── PATHS (must match ema_trend_v1.py) ──
PREPARED_DIR  = '/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared'
EMA_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/EMA_Trend_Labels_v1'
OUTPUT_DIR    = '/content/drive/MyDrive/AI_Models/EMA_Trend_v1'

# ── TICKERS (must match ema_trend_v1.py) ──
DATA_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC']
TICKERS      = DATA_TICKERS
MICRO_TO_FULL = {
    'MES': 'ES', 'MNQ': 'NQ', 'MRTY': 'RTY', 'MYM': 'YM', 'MGC': 'GC',
}

# ── MODEL (must match v1.0) ──
SEQ_LEN          = 96
NUM_LABELS       = 2
RISK_WEIGHT      = 0.1
FOCAL_SMOOTHING  = 0.10
NUM_FFM_FEATURES = len(get_model_feature_columns())
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

# ── WALK-FORWARD FOLDS (must match ema_trend_v1.py) ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── PHASE 2 TRAINING ──
RR_LR       = 1e-5
RR_EPOCHS   = 20
RR_PATIENCE = 5
RR_BATCH    = 64
HUBER_DELTA = 1.0   # transitions from L2→L1 at 1.0R — robust to 5R outliers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Folds:  {[f["name"] for f in FOLDS]}')
print(f'Phase 2 — risk_head only, Huber loss, LR={RR_LR}, epochs={RR_EPOCHS}')


# ==============================================================================
# CELL 3 — DATA LOADING
# ==============================================================================

import pandas as pd
import numpy as np
from futures_foundation.finetune import HybridStrategyDataset
from futures_foundation.finetune.trainer import _load_fold_data, _concat_with_meta

ffm_config = FFMConfig(
    num_features    = NUM_FFM_FEATURES,
    label_smoothing = FOCAL_SMOOTHING,
)

# Pre-load fold data for all folds (no labeling — reuse v1.0 cache)
print(f'\n{"="*60}')
print('  Loading fold datasets from cache...')
print(f'{"="*60}')

fold_data = {}
for fold in FOLDS:
    fold_cfg = {
        'train_end': fold['train_end'],
        'val_end':   fold['val_end'],
        'test_end':  fold['test_end'],
    }
    train_dsets, val_dsets, test_dsets = _load_fold_data(
        fold_cfg, TICKERS, PREPARED_DIR, EMA_CACHE_DIR,
        EMA_FEATURE_COLS, SEQ_LEN, MICRO_TO_FULL,
    )
    if not train_dsets or not val_dsets:
        print(f'  ⚠ {fold["name"]}: insufficient data — will skip')
        fold_data[fold['name']] = None
        continue

    train_ds = _concat_with_meta(train_dsets, SEQ_LEN)
    val_ds   = ConcatDataset(val_dsets)

    # Signal-only subsets — Phase 2 trains exclusively on confirmed signal windows
    train_sig = Subset(train_ds, train_ds.signal_indices)
    val_sig_indices = []
    for i, d in enumerate(val_dsets):
        offset = sum(len(val_dsets[j]) for j in range(i))
        val_sig_indices.extend(offset + s for s in d.signal_indices)
    val_sig = Subset(val_ds, val_sig_indices)

    n_train_sig = len(train_sig)
    n_val_sig   = len(val_sig)
    print(f'  {fold["name"]}: {n_train_sig} train signals, {n_val_sig} val signals')

    fold_data[fold['name']] = {
        'train_sig': train_sig,
        'val_sig':   val_sig,
    }

print('\n✅ Data loaded')


# ==============================================================================
# CELL 4 — PHASE 2 FINE-TUNING
# ==============================================================================

from futures_foundation.finetune import HybridStrategyModel, export_onnx

def _run_rr_epoch(model, loader, optimizer, training):
    model.train() if training else model.eval()
    total_loss = 0.0; n = 0
    all_pred = []; all_true = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            feats   = batch['features'].to(DEVICE)
            strat   = batch['strategy_features'].to(DEVICE)
            candles = batch['candle_types'].to(DEVICE)
            inst    = batch['instrument_ids'].to(DEVICE)
            sess    = batch['session_ids'].to(DEVICE)
            tod     = batch['time_of_day'].to(DEVICE)
            dow     = batch['day_of_week'].to(DEVICE)
            max_rr  = batch['max_rr'].to(DEVICE)

            out  = model(features=feats, strategy_features=strat,
                         candle_types=candles, time_of_day=tod,
                         day_of_week=dow, instrument_ids=inst,
                         session_ids=sess)
            pred = out['risk_predictions'].squeeze(-1)
            loss = F.huber_loss(pred, max_rr, delta=HUBER_DELTA)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item(); n += 1
            all_pred.extend(pred.detach().cpu().tolist())
            all_true.extend(max_rr.cpu().tolist())

    mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))
    return total_loss / max(n, 1), mae, all_pred, all_true


def train_risk_head(fold_name, model, train_sig, val_sig):
    # Freeze everything except risk_head
    for name, param in model.named_parameters():
        param.requires_grad = 'risk_head' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params: {trainable:,} (risk_head only)')

    optimizer = torch.optim.Adam(model.risk_head.parameters(), lr=RR_LR)

    train_loader = DataLoader(train_sig, batch_size=RR_BATCH, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_sig,   batch_size=RR_BATCH, shuffle=False,
                              num_workers=2, pin_memory=True)

    best_val_loss = float('inf')
    best_state    = None
    patience_left = RR_PATIENCE

    for epoch in range(1, RR_EPOCHS + 1):
        tr_loss, tr_mae, _, _ = _run_rr_epoch(model, train_loader, optimizer, training=True)
        vl_loss, vl_mae, _, _ = _run_rr_epoch(model, val_loader,   optimizer, training=False)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = RR_PATIENCE
            marker = ' ✅'
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'  Early stop at epoch {epoch}')
                break

        print(f'  E{epoch:02d}  train_huber={tr_loss:.4f} mae={tr_mae:.3f}R  '
              f'val_huber={vl_loss:.4f} mae={vl_mae:.3f}R{marker}')

    model.load_state_dict(best_state)

    # Final val metrics with best weights
    _, _, val_pred, val_true = _run_rr_epoch(model, val_loader, optimizer, training=False)
    return model, np.array(val_pred), np.array(val_true)


def print_calibration(fold_name, pred, true):
    print(f'\n  Calibration — {fold_name} val signals ({len(pred)} total)')
    print(f'  {"Predict ≥":>10}  {"Signals":>8}  {"Pct":>6}  {"2R hit":>8}  {"3R hit":>8}  {"4R hit":>8}')
    print(f'  {"-"*58}')
    for thr in [1.0, 1.5, 2.0, 3.0, 4.0]:
        mask = pred >= thr
        n = mask.sum()
        if n == 0:
            continue
        pct    = n / len(pred) * 100
        hit_2r = (true[mask] >= 2.0).mean() * 100
        hit_3r = (true[mask] >= 3.0).mean() * 100
        hit_4r = (true[mask] >= 4.0).mean() * 100
        print(f'  {thr:>10.1f}  {n:>8}  {pct:>5.1f}%  {hit_2r:>7.1f}%  {hit_3r:>7.1f}%  {hit_4r:>7.1f}%')

    print(f'\n  Actual max_rr distribution:')
    for pct, label in [(25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
        print(f'    {label}: {np.percentile(true, pct):.2f}R')
    print(f'  Predicted max_rr distribution:')
    for pct, label in [(25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
        print(f'    {label}: {np.percentile(pred, pct):.2f}R')


print(f'\n{"="*60}')
print('  PHASE 2 — RISK HEAD FINE-TUNING')
print(f'{"="*60}')

rr_done_paths = {}

for fold in FOLDS:
    fold_name = fold['name']

    if fold_data.get(fold_name) is None:
        print(f'\n  ⚠ {fold_name}: no data — skipping')
        continue

    # Find Phase 1 checkpoint for this fold
    done_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f'{fold_name}_*_done.pt')))
    done_files = [f for f in done_files if '_rr_done' not in f]
    if not done_files:
        print(f'\n  ⚠ {fold_name}: no Phase 1 _done.pt found — skipping')
        continue
    p1_path = done_files[-1]
    p1_ckpt = torch.load(p1_path, map_location='cpu', weights_only=False)
    config_hash = p1_ckpt.get('config_hash', 'unknown')

    rr_ckpt_path = os.path.join(OUTPUT_DIR, f'{fold_name}_{config_hash}_rr_done.pt')

    print(f'\n{"="*60}')
    print(f'  {fold_name} | hash={config_hash} | from {os.path.basename(p1_path)}')
    print(f'{"="*60}')

    if os.path.exists(rr_ckpt_path):
        print(f'  ✅ Phase 2 checkpoint already exists — skipping re-training')
        rr_done_paths[fold_name] = rr_ckpt_path
        continue

    # Reconstruct model from Phase 1 checkpoint
    model = HybridStrategyModel(
        ffm_config            = ffm_config,
        num_strategy_features = NUM_EMA_FEATURES,
        num_labels            = NUM_LABELS,
        risk_weight           = RISK_WEIGHT,
    ).to(DEVICE)
    model.load_state_dict(
        {k: v.to(DEVICE) for k, v in p1_ckpt['next_fold_state'].items()}
    )

    data   = fold_data[fold_name]
    model, val_pred, val_true = train_risk_head(
        fold_name, model, data['train_sig'], data['val_sig']
    )
    print_calibration(fold_name, val_pred, val_true)

    # Save Phase 2 checkpoint (same structure as Phase 1 _done.pt for ONNX compat)
    torch.save({
        'config_hash':     config_hash,
        'phase':           'v1.1_risk_head',
        'next_fold_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'rr_metrics': {
            'val_pred': val_pred.tolist(),
            'val_true': val_true.tolist(),
            'val_mae':  float(np.mean(np.abs(val_pred - val_true))),
        },
    }, rr_ckpt_path)
    print(f'\n  💾 Saved: {rr_ckpt_path}')
    rr_done_paths[fold_name] = rr_ckpt_path

print(f'\n✅ Phase 2 complete')


# ==============================================================================
# CELL 5 — ONNX EXPORT (F4 Phase 2 checkpoint)
# ==============================================================================

import json

f4_rr_path = rr_done_paths.get('F4')
if not f4_rr_path:
    raise RuntimeError('F4 Phase 2 checkpoint not found — check Cell 4 output')

print(f'\nLoading F4 Phase 2 checkpoint: {f4_rr_path}')
f4_ckpt     = torch.load(f4_rr_path, map_location='cpu', weights_only=False)
config_hash = f4_ckpt.get('config_hash', 'unknown')

model = HybridStrategyModel(
    ffm_config            = ffm_config,
    num_strategy_features = NUM_EMA_FEATURES,
    num_labels            = NUM_LABELS,
    risk_weight           = RISK_WEIGHT,
)
model.load_state_dict(f4_ckpt['next_fold_state'])
model.eval()

onnx_path = os.path.join(OUTPUT_DIR, 'ema_trend_hybrid_v11.onnx')
meta_path  = onnx_path.replace('.onnx', '_metadata.json')

export_onnx(
    model,
    onnx_path,
    seq_len               = SEQ_LEN,
    num_ffm_features      = NUM_FFM_FEATURES,
    num_strategy_features = NUM_EMA_FEATURES,
)

rr_metrics = f4_ckpt.get('rr_metrics', {})
metadata = {
    'version':            'ema_trend_v1_1',
    'phase':              'risk_head_calibrated',
    'config_hash':        config_hash,
    'seq_len':            SEQ_LEN,
    'num_labels':         NUM_LABELS,
    'num_features':       NUM_FFM_FEATURES,
    'num_ema_features':   NUM_EMA_FEATURES,
    'feature_cols':       get_model_feature_columns(),
    'ema_feature_cols':   EMA_FEATURE_COLS,
    'f4_rr_val_mae':      rr_metrics.get('val_mae'),
    'inference': {
        'output_0':   '[B,2] signal_logits — softmax; index 1 = signal probability',
        'output_1':   '[B,1] risk_predictions — predicted max R:R (Softplus, calibrated)',
        'output_2':   '[B]   confidence — max(softmax), range [0.5, 1.0]',
        'direction':  'ema_feature_cols index 0 (fast_spread_norm): >0=LONG, <=0=SHORT',
        'sl_sizing':  'sl_distance = ATR × 1.5; contracts = risk_dollars / (ATR × 1.5 × point_value)',
        'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
        'tp_usage':   'if predicted_rr >= 3.0, set TP at 2R (conservative); >= 4.0 → 3R',
        'warning':    'max_rr label is an optimistic ceiling — use predicted_rr conservatively',
    },
}

with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Metadata written: {meta_path}')
print('\n✅ Done — ema_trend_hybrid_v11.onnx ready for algotrader integration')
