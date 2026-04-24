"""
Standalone ONNX export for CISD+OTE v7.0.

Run this in Colab instead of the full cisd_ote.py when all 4 fold checkpoints
already exist and you only want to regenerate cisd_ote_hybrid.onnx.

Requirements: F4_*_done.pt must exist in OUTPUT_DIR.
No labeling data, raw CSVs, or training required.

Usage (Colab):
    !python /content/Futures-Foundation-Model/scripts/export_cisd_ote_v7.py
"""

import os, glob, json, subprocess

# ── Setup ─────────────────────────────────────────────────────────────────────

os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive')

subprocess.run(['git', 'clone', '--depth=1',
                'https://github.com/johnamcruz/Futures-Foundation-Model.git',
                '/content/Futures-Foundation-Model'],
               check=True)
os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')

# ── Config (must match training) ───────────────────────────────────────────────

OUTPUT_DIR    = '/content/drive/MyDrive/AI_Models/CISD_OTE_Hybrid_v7'
SEQ_LEN       = 96
NUM_LABELS    = 2
RISK_WEIGHT   = 0.1
FOCAL_SMOOTHING = 0.10
CISD_FEATURE_COLS = [
    'zone_height_vs_atr', 'price_vs_zone_top', 'price_vs_zone_bot',
    'zone_age_bars', 'zone_is_bullish', 'cisd_displacement_str',
    'had_liquidity_sweep', 'entry_distance_pct', 'risk_dollars_norm',
    'in_optimal_session',
]
NUM_CISD_FEATURES = len(CISD_FEATURE_COLS)

# ── Load F4 checkpoint ─────────────────────────────────────────────────────────

import torch
from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune.model import HybridStrategyModel
from futures_foundation.finetune.trainer import export_onnx

done_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'F4_*_done.pt')))
if not done_files:
    raise FileNotFoundError(
        f'No F4_*_done.pt found in {OUTPUT_DIR}.\n'
        'Run the full cisd_ote.py training first.'
    )
ckpt_path = done_files[-1]
print(f'Loading checkpoint: {ckpt_path}')

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
config_hash = ckpt.get('config_hash', 'unknown')
print(f'Config hash: {config_hash}')

# ── Reconstruct model ──────────────────────────────────────────────────────────

ffm_config = FFMConfig(
    num_features    = len(get_model_feature_columns()),
    label_smoothing = FOCAL_SMOOTHING,
)

model = HybridStrategyModel(
    ffm_config           = ffm_config,
    num_strategy_features = NUM_CISD_FEATURES,
    num_labels           = NUM_LABELS,
    risk_weight          = RISK_WEIGHT,
)
model.load_state_dict(ckpt['next_fold_state'])
print(f'Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters')

# ── Export ─────────────────────────────────────────────────────────────────────

onnx_path = os.path.join(OUTPUT_DIR, 'cisd_ote_hybrid.onnx')
meta_path = onnx_path.replace('.onnx', '_metadata.json')

export_onnx(
    model,
    onnx_path,
    seq_len               = SEQ_LEN,
    num_ffm_features      = len(get_model_feature_columns()),
    num_strategy_features = NUM_CISD_FEATURES,
)

# ── Metadata ───────────────────────────────────────────────────────────────────

test_metrics = ckpt.get('test_metrics', {})
metadata = {
    'version':           'cisd_ote_v7_0',
    'config_hash':       config_hash,
    'seq_len':           SEQ_LEN,
    'num_labels':        NUM_LABELS,
    'num_features':      len(get_model_feature_columns()),
    'num_cisd_features': NUM_CISD_FEATURES,
    'feature_cols':      get_model_feature_columns(),
    'cisd_feature_cols': CISD_FEATURE_COLS,
    'inference': {
        'output_0': '[B,2] signal_logits — apply softmax; index 1 = signal probability',
        'output_1': '[B,1] risk_predictions — predicted R:R (Softplus)',
        'output_2': '[B]   confidence — max(softmax), range [0.5, 1.0]',
        'direction': 'cisd_feature_cols index 4 (zone_is_bullish): >0=BUY, <=0=SELL',
        'thresholds': {'conservative': 0.90, 'moderate': 0.80, 'aggressive': 0.70},
    },
}

# Add per-fold precision from saved test_metrics if available
if test_metrics and 'all_conf' in test_metrics:
    confs  = torch.tensor(test_metrics['all_conf'])
    labels = torch.tensor(test_metrics['all_labels'])
    entry  = {'signals': int((labels > 0).sum())}
    for key, thr in [('prec_at_70', 0.70), ('prec_at_80', 0.80), ('prec_at_90', 0.90)]:
        mask = confs >= thr
        entry[key] = round(float((labels[mask] > 0).float().mean()), 3) if mask.sum() > 0 else None
    metadata['f4_test_results'] = entry

with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'Metadata written: {meta_path}')
print('\nDone.')
