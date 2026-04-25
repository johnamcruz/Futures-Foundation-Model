# ==============================================================================
# CISD+OTE v7.1 — PHASE 2: RISK HEAD CALIBRATION
# ==============================================================================
# Loads v7.0 walk-forward checkpoints and fine-tunes ONLY the risk_head
# so predicted_rr can be used as a reliable TP target at trade entry.
#
# Requires: F1–F4 _done.pt checkpoints in OUTPUT_DIR (from cisd_ote.py v7.0)
# Requires: Labeled parquets in CISD_CACHE_DIR (from cisd_ote.py v7.0 labeling)
#
# What is frozen:  backbone, strategy_projection, fusion, signal_head
# What is trained: risk_head only (~65K params vs 3.4M total)
# Loss:            Huber (robust to occasional 5R outlier signals)
# LR:              1e-5, epochs: 20 per fold, early stop patience: 5
#
# Output:  F{n}_{hash}_rr_done.pt per fold + cisd_ote_hybrid_v71.onnx
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

import torch

# ── PATHS (must match cisd_ote.py) ──
PREPARED_DIR   = '/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared'
CISD_CACHE_DIR = '/content/drive/MyDrive/AI_Cache/CISD_OTE_Labels_v7'
OUTPUT_DIR     = '/content/drive/MyDrive/AI_Models/CISD_OTE_Hybrid_v7'

# ── TICKERS (must match cisd_ote.py) ──
TICKERS      = ['ES', 'NQ', 'RTY', 'YM', 'GC']
MICRO_TO_FULL = {
    'MES': 'ES', 'MNQ': 'NQ', 'MRTY': 'RTY', 'MYM': 'YM', 'MGC': 'GC',
}

# ── MODEL (must match v7.0) ──
SEQ_LEN         = 96
NUM_LABELS      = 2
RISK_WEIGHT     = 0.1
FOCAL_SMOOTHING = 0.10
CISD_FEATURE_COLS = [
    'zone_height_vs_atr', 'price_vs_zone_top', 'price_vs_zone_bot',
    'zone_age_bars', 'zone_is_bullish', 'cisd_displacement_str',
    'had_liquidity_sweep', 'entry_distance_pct', 'risk_dollars_norm',
    'in_optimal_session',
]

# ── WALK-FORWARD FOLDS (must match cisd_ote.py) ──
FOLDS = [
    {'name': 'F1', 'train_end': '2022-01-01', 'val_end': '2022-07-01', 'test_end': '2023-01-01'},
    {'name': 'F2', 'train_end': '2023-01-01', 'val_end': '2023-07-01', 'test_end': '2024-01-01'},
    {'name': 'F3', 'train_end': '2024-01-01', 'val_end': '2024-07-01', 'test_end': '2025-01-01'},
    {'name': 'F4', 'train_end': '2025-01-01', 'val_end': '2025-06-01', 'test_end': '2025-12-01'},
]

# ── PHASE 2 HYPERPARAMS ──
RR_LR       = 1e-5
RR_EPOCHS   = 20
RR_PATIENCE = 5
RR_BATCH    = 64
HUBER_DELTA = 1.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Phase 2 — risk_head only, Huber loss, LR={RR_LR}, epochs={RR_EPOCHS}')


# ==============================================================================
# CELL 3 — PHASE 2 FINE-TUNING
# ==============================================================================

from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune import run_risk_head_calibration

ffm_config = FFMConfig(
    num_features    = len(get_model_feature_columns()),
    label_smoothing = FOCAL_SMOOTHING,
)

rr_done_paths = run_risk_head_calibration(
    folds                 = FOLDS,
    tickers               = TICKERS,
    ffm_dir               = PREPARED_DIR,
    strategy_dir          = CISD_CACHE_DIR,
    output_dir            = OUTPUT_DIR,
    strategy_feature_cols = CISD_FEATURE_COLS,
    ffm_config            = ffm_config,
    num_labels            = NUM_LABELS,
    risk_weight           = RISK_WEIGHT,
    focal_smoothing       = FOCAL_SMOOTHING,
    micro_to_full         = MICRO_TO_FULL,
    seq_len               = SEQ_LEN,
    rr_lr                 = RR_LR,
    rr_epochs             = RR_EPOCHS,
    rr_patience           = RR_PATIENCE,
    rr_batch              = RR_BATCH,
    huber_delta           = HUBER_DELTA,
    device                = DEVICE,
)


# ==============================================================================
# CELL 4 — ONNX EXPORT (F4 Phase 2 checkpoint)
# ==============================================================================

import json
from futures_foundation.finetune import HybridStrategyModel, export_onnx

f4_rr_path = rr_done_paths.get('F4')
if not f4_rr_path:
    raise RuntimeError('F4 Phase 2 checkpoint not found — check Cell 3 output')

print(f'\nLoading F4 Phase 2 checkpoint: {f4_rr_path}')
f4_ckpt     = torch.load(f4_rr_path, map_location='cpu', weights_only=False)
config_hash = f4_ckpt.get('config_hash', 'unknown')

model = HybridStrategyModel(
    ffm_config            = ffm_config,
    num_strategy_features = len(CISD_FEATURE_COLS),
    num_labels            = NUM_LABELS,
    risk_weight           = RISK_WEIGHT,
)
model.load_state_dict(f4_ckpt['next_fold_state'])
model.eval()

onnx_path = os.path.join(OUTPUT_DIR, 'cisd_ote_hybrid_v71.onnx')
meta_path  = onnx_path.replace('.onnx', '_metadata.json')

export_onnx(
    model,
    onnx_path,
    seq_len               = SEQ_LEN,
    num_ffm_features      = len(get_model_feature_columns()),
    num_strategy_features = len(CISD_FEATURE_COLS),
)

rr_metrics = f4_ckpt.get('rr_metrics', {})
metadata = {
    'version':           'cisd_ote_v7_1',
    'phase':             'risk_head_calibrated',
    'config_hash':       config_hash,
    'seq_len':           SEQ_LEN,
    'num_labels':        NUM_LABELS,
    'num_features':      len(get_model_feature_columns()),
    'num_cisd_features': len(CISD_FEATURE_COLS),
    'feature_cols':      get_model_feature_columns(),
    'cisd_feature_cols': CISD_FEATURE_COLS,
    'f4_rr_val_mae':     rr_metrics.get('val_mae'),
    'inference': {
        'output_0':   '[B,2] signal_logits — softmax; index 1 = signal probability',
        'output_1':   '[B,1] risk_predictions — predicted max R:R (Softplus, calibrated)',
        'output_2':   '[B]   confidence — max(softmax), range [0.5, 1.0]',
        'direction':  'cisd_feature_cols index 4 (zone_is_bullish): >0=BUY, <=0=SELL',
        'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
        'tp_usage':   'if predicted_rr >= 3.0, set TP at 2R (conservative); >= 4.0 → 3R',
        'warning':    'max_rr label is an optimistic ceiling — use predicted_rr conservatively',
    },
}

with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Metadata written: {meta_path}')
print('\n✅ Done — cisd_ote_hybrid_v71.onnx ready for algotrader integration')
