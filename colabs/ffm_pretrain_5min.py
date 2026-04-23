# # 🚀 Futures Foundation Model — Pretraining Pipeline
#
# **Step 1:** Prepare data (derive 66 features + generate 4 labels from raw OHLCV)
# **Step 2:** Pretrain backbone (4-task self-supervised learning)
#
# ## Prerequisites
# - Raw 5-min OHLCV CSVs on Google Drive
# - Columns: `datetime, open, high, low, close, volume`
# - Files named like: `ES_5min.csv`, `NQ_5min.csv`, `GC_5min.csv`, etc.
# - GPU runtime (T4 is fine)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — 5min Timeframe | Foundation Model Pretraining
# ═══════════════════════════════════════════════════════════════

RAW_DATA_DIR      = "/content/drive/MyDrive/Futures Data/5min/"
PREPARED_DATA_DIR = "/content/drive/MyDrive/AI_Cache/5min_FFM_Prepared/"
CHECKPOINT_DIR    = "/content/drive/MyDrive/AI_Cache/5min_FFM_Checkpoints/"

# ─── Training Hyperparameters ───
EPOCHS        = 50
BATCH_SIZE    = 256
LEARNING_RATE = 1e-4
SEQ_LEN       = 96         # 96 bars × 5min = ~8.0 hours — full trading day + overnight context
                            # AMP halves activation memory, making 96 as cheap as 64 was before
TRAIN_STRIDE  = 4          # stride=4 reduces ~96× overlap between training sequences to ~24×
                            # — cuts epoch time ~4× while exposing more diverse market windows
VAL_RATIO     = 0.20       # interleaved split samples val from every 5th time block
                            # so all market regimes appear in validation, not just the last 20%
PATIENCE      = 10
WARMUP_STEPS  = 8000
GRAD_CLIP     = 1.0

# ─── Overfitting / Stability Thresholds ───
MAX_RATIO      = 1.25
RATIO_PATIENCE = 12

# ─── Model Architecture ───
HIDDEN_SIZE       = 256
NUM_LAYERS        = 6
NUM_HEADS         = 8
INTERMEDIATE_SIZE = 512
LABEL_SMOOTHING   = 0.1    # market labels are threshold-computed from noisy price data;
                            # soft targets prevent over-confidence on borderline bars

SEED = 42

print("✅ Configuration set — 5min Foundation Model Pretraining")
print(f"   Raw data:    {RAW_DATA_DIR}")
print(f"   Prepared:    {PREPARED_DATA_DIR}")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   SEQ_LEN:     {SEQ_LEN} bars × 5min = ~{SEQ_LEN * 5 / 60:.1f} hours context")
print(f"   Stride:      {TRAIN_STRIDE} (train) / 1 (val)")
print(f"   LabelSmooth: {LABEL_SMOOTHING}")
print(f"   Val ratio:   {VAL_RATIO:.0%} (interleaved across all time blocks)")

# ---
# ## Cell 1: Setup — Clone repo, install dependencies, mount Drive
# ---
import os, subprocess

os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive')

print("📥 Cloning fresh repo...")
os.system('rm -rf /content/Futures-Foundation-Model')
result = subprocess.run(
    ['git', 'clone', 'https://github.com/johnamcruz/Futures-Foundation-Model.git', '/content/Futures-Foundation-Model'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"❌ Clone failed:\n{result.stderr}")
    raise RuntimeError("Git clone failed — check repo URL and network")

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q')

print("\n✅ Setup complete")
print(f"📁 Working directory: {os.getcwd()}")

import torch
if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   AMP: ✅ enabled — float16 forward passes")
else:
    print("⚠️  No GPU detected! Go to Runtime → Change runtime type → T4 GPU")

# ---
# ## Cell 2: Verify Data Files
# ---
from pathlib import Path
import pandas as pd

raw_dir = Path(RAW_DATA_DIR)
data_files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet")))

if not data_files:
    print(f"❌ No data files found in {RAW_DATA_DIR}")
    print(f"   Expected: ES_5min.csv, NQ_5min.csv, GC_5min.csv, etc.")
    print(f"   Columns:  datetime, open, high, low, close, volume")
else:
    print(f"📊 Found {len(data_files)} data files:\n")
    for f in data_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name:30s} ({size_mb:.1f} MB)")

    df_peek = pd.read_csv(data_files[0], nrows=5) if data_files[0].suffix == ".csv" \
              else pd.read_parquet(data_files[0]).head(5)
    print(f"\n📋 Columns in {data_files[0].name}: {list(df_peek.columns)}")
    print(f"\n📋 First 3 rows:\n{df_peek.head(3).to_string()}")

# ---
# ## Cell 3: STEP 1 — Prepare Data
# Derives 66 features + generates 4 pretraining labels from raw OHLCV.
# Saves as parquet. Run once (or when label logic changes).
# ---
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from futures_foundation import (
    derive_features,
    generate_all_labels,
    get_model_feature_columns,
    print_label_distribution,
    INSTRUMENT_MAP,
)

raw_dir = Path(RAW_DATA_DIR)
out_dir = Path(PREPARED_DATA_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

data_files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet")))

print(f"\n{'='*60}")
print(f"  STEP 1: DATA PREPARATION")
print(f"  Input:  {raw_dir} ({len(data_files)} files)")
print(f"  Output: {out_dir}")
print(f"{'='*60}")

summary = {}
total_start = time.time()

for data_path in data_files:
    instrument = data_path.stem.split("_")[0].upper()

    if instrument not in INSTRUMENT_MAP:
        print(f"\n  ⚠ Skipping {data_path.name} — '{instrument}' not in INSTRUMENT_MAP")
        print(f"    Known: {list(INSTRUMENT_MAP.keys())}")
        continue

    print(f"\n{'─'*60}")
    print(f"  {instrument} — {data_path.name}")
    print(f"{'─'*60}")

    t0 = time.time()

    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" \
         else pd.read_csv(data_path)

    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})

    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ❌ Missing columns: {missing} — skipping")
        continue

    print(f"  Loaded {len(df):,} bars")
    print(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

    features_df = derive_features(df, instrument=instrument)

    feature_cols = get_model_feature_columns()
    valid_count = features_df[feature_cols].notna().all(axis=1).sum()
    nan_count = len(features_df) - valid_count
    print(f"  Derived {len(feature_cols)} features")
    print(f"  Valid rows: {valid_count:,} / {len(features_df):,} ({valid_count/len(features_df)*100:.1f}%)")
    if nan_count > 0:
        print(f"  NaN rows (warmup): {nan_count:,} — expected from rolling windows")

    labels_df = generate_all_labels(features_df)
    print_label_distribution(labels_df)

    features_path = out_dir / f"{instrument}_features.parquet"
    labels_path = out_dir / f"{instrument}_labels.parquet"
    features_df.to_parquet(features_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    elapsed = time.time() - t0
    feat_size = features_path.stat().st_size / 1024 / 1024
    label_size = labels_path.stat().st_size / 1024 / 1024

    print(f"\n  ✓ Saved: {features_path.name} ({feat_size:.1f} MB) + {labels_path.name} ({label_size:.1f} MB)")
    print(f"    Time: {elapsed:.1f}s")

    summary[instrument] = {
        "raw_bars": len(df),
        "valid_bars": int(valid_count),
        "date_start": str(df["datetime"].iloc[0]),
        "date_end": str(df["datetime"].iloc[-1]),
    }

config_path = out_dir / "prep_config.json"
with open(config_path, "w") as f:
    json.dump({
        "num_features": len(get_model_feature_columns()),
        "feature_columns": get_model_feature_columns(),
        "instruments": summary,
    }, f, indent=2)

total_elapsed = time.time() - total_start
total_bars = sum(s["raw_bars"] for s in summary.values())
total_valid = sum(s["valid_bars"] for s in summary.values())

print(f"\n{'='*60}")
print(f"  ✅ STEP 1 COMPLETE")
print(f"  Instruments: {len(summary)}")
print(f"  Total bars:  {total_bars:,} ({total_valid:,} valid)")
print(f"  Time: {total_elapsed:.1f}s")
print(f"  Output: {out_dir}")
print(f"{'='*60}")

# ## Cell 5: STEP 2 — Pretrain Backbone
#
# Improvements over baseline:
#   ✅ FIX 1 — candle_types passed to model (was silently dropped; embedding now trains)
#   ✅ FIX 2 — interleaved_train_val_split: val covers all market regimes, not just recent 20%
#   ✅ FIX 3 — stride_train=4: 4× less overlap between sequences → more diverse training
#   ✅ FIX 4 — label_smoothing=0.1: soft targets for noisy threshold-based market labels
#   ✅ FIX 5 — AMP (float16): ~1.5-2× faster on T4, enabling SEQ_LEN=96 context window
#   ✅ FIX 6 — per-task loss logging: diagnose which head is struggling each epoch

import time
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

from futures_foundation import (
    FFMConfig,
    FFMForPretraining,
    get_model_feature_columns,
    FFMMultiInstrumentDataset,
    interleaved_train_val_split,
    create_dataloaders,
    REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}  |  AMP: {'✅ float16' if USE_AMP else '❌ disabled (no GPU)'}")

# ─── Training Config ───
STABLE_EPOCHS   = 3
MIN_TASK_ACC    = 0.10
MAX_MAJORITY    = 0.95
LABEL_SENTINEL  = -100

TASK_NAMES = ["regime", "volatility", "structure", "range"]
TASK_LABEL_MAPS = {
    "regime": REGIME_LABELS, "volatility": VOLATILITY_LABELS,
    "structure": STRUCTURE_LABELS, "range": RANGE_LABELS,
}
TASK_NUM_CLASSES = {"regime": 4, "volatility": 4, "structure": 2, "range": 5}


# =========================================================
# HELPERS
# =========================================================
def gen_status(train_loss, val_loss):
    r = val_loss / train_loss if train_loss > 0 else 1.0
    if   r > 1.20: return f"🚨 CRIT ({r:.2f})", "crit"
    elif r > 1.15: return f"⚠️  SEV  ({r:.2f})",  "sev"
    elif r > 1.12: return f"⚠️  MOD  ({r:.2f})",  "mod"
    elif r > 1.08: return f"ℹ️  SLT  ({r:.2f})",  "slt"
    elif r < 0.85: return f"ℹ️  UND  ({r:.2f})",  "und"
    else:          return f"✅ OK   ({r:.2f})",    "ok"


def check_collapse(preds_counter, num_classes, task_name):
    total = sum(preds_counter.values())
    if total == 0:
        return True, "no predictions"
    for cls, count in preds_counter.items():
        if count / total > MAX_MAJORITY:
            return True, f"class {cls} = {count/total:.0%}"
    return False, ""


def task_accuracy(preds, labels, sentinel=None):
    if sentinel is not None:
        mask = labels != sentinel
        if mask.sum() == 0:
            return 0, 0
        return (preds[mask] == labels[mask]).sum().item(), mask.sum().item()
    return (preds == labels).sum().item(), labels.size(0)


# =========================================================
# LOAD DATA — interleaved split across all market regimes
# =========================================================
prepared_dir = Path(PREPARED_DATA_DIR)
feature_files = sorted(prepared_dir.glob("*_features.parquet"))

if not feature_files:
    raise FileNotFoundError(f"No prepared data in {prepared_dir}. Run Cell 3 first.")

print(f"\n{'='*60}")
print(f"  LOADING PREPARED DATA")
print(f"  Split: interleaved 80/20 across 20 time blocks (val in every 5th block)")
print(f"  Stride: train={TRAIN_STRIDE}, val=1")
print(f"{'='*60}")

train_datasets, val_datasets = [], []

for feat_path in feature_files:
    instrument = feat_path.stem.replace("_features", "")
    label_path = prepared_dir / f"{instrument}_labels.parquet"
    if not label_path.exists():
        print(f"  ⚠ Skipping {instrument} — no labels file")
        continue

    t0 = time.time()
    features_df = pd.read_parquet(feat_path)
    labels_df = pd.read_parquet(label_path)
    load_time = time.time() - t0

    # FIX #2 + #3: interleaved val across all regimes, stride=4 for train
    tr_dsets, va_dsets = interleaved_train_val_split(
        features_df, labels_df,
        val_ratio=VAL_RATIO,
        seq_len=SEQ_LEN,
        n_blocks=20,
        stride_train=TRAIN_STRIDE,
    )
    tr_seqs = sum(len(d) for d in tr_dsets)
    va_seqs = sum(len(d) for d in va_dsets)
    print(f"  {instrument}: {len(features_df):,} bars → "
          f"{tr_seqs:,} train / {va_seqs:,} val sequences ({load_time:.1f}s)")

    train_datasets.extend(tr_dsets)
    val_datasets.extend(va_dsets)

combined_train = FFMMultiInstrumentDataset(train_datasets)
combined_val   = FFMMultiInstrumentDataset(val_datasets)

total_seqs = len(combined_train) + len(combined_val)
actual_val_pct = len(combined_val) / total_seqs if total_seqs > 0 else 0
print(f"\n  Total: {len(combined_train):,} train / {len(combined_val):,} val "
      f"({actual_val_pct:.1%} val — expected ~{VAL_RATIO:.0%})")

if actual_val_pct < 0.05:
    raise RuntimeError(
        f"Val set is only {actual_val_pct:.1%} — interleaved split may have failed."
    )
print(f"  ✅ Split looks correct\n")

train_loader, val_loader = create_dataloaders(
    combined_train, combined_val, batch_size=BATCH_SIZE, num_workers=2,
)

# =========================================================
# MODEL
# =========================================================
num_features = len(get_model_feature_columns())
config = FFMConfig(
    num_features=num_features,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    label_smoothing=LABEL_SMOOTHING,   # FIX #4: soft targets for noisy market labels
)

model = FFMForPretraining(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} parameters")
print(f"  Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}, FF: {INTERMEDIATE_SIZE}")
print(f"  Label smoothing: {LABEL_SMOOTHING}")

# =========================================================
# OPTIMIZER, SCHEDULER, AMP SCALER
# =========================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
total_steps = len(train_loader) * EPOCHS

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# FIX #5: AMP GradScaler — only active when running on GPU
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

# ─── Save config ───
ckpt_dir = Path(CHECKPOINT_DIR)
ckpt_dir.mkdir(parents=True, exist_ok=True)
config.save_pretrained(str(ckpt_dir))

with open(ckpt_dir / "train_args.json", "w") as f:
    json.dump({
        "timeframe": "5min",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        "seq_len": SEQ_LEN, "train_stride": TRAIN_STRIDE,
        "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS, "intermediate_size": INTERMEDIATE_SIZE,
        "label_smoothing": LABEL_SMOOTHING,
        "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP,
        "val_ratio": VAL_RATIO, "patience": PATIENCE, "seed": SEED,
        "max_ratio": MAX_RATIO, "ratio_patience": RATIO_PATIENCE,
        "amp": USE_AMP,
        "val_split": "interleaved_train_val_split (n_blocks=20)",
        "fixes": [
            "candle_types passed to model",
            "interleaved val split",
            f"stride_train={TRAIN_STRIDE}",
            f"label_smoothing={LABEL_SMOOTHING}",
            "AMP float16",
            "per-task loss logging",
        ],
    }, f, indent=2)


# =========================================================
# TRAINING FUNCTIONS
# =========================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, num_batches = 0, 0
    task_correct = {t: 0 for t in TASK_NAMES}
    task_total   = {t: 0 for t in TASK_NAMES}
    task_loss_sum   = {t: 0.0 for t in TASK_NAMES}
    task_loss_count = {t: 0   for t in TASK_NAMES}

    for batch in loader:
        kwargs = {
            "features":          batch["features"].to(device),
            "candle_types":      batch["candle_types"].to(device),   # FIX #1
            "time_of_day":       batch["time_of_day"].to(device),
            "day_of_week":       batch["day_of_week"].to(device),
            "instrument_ids":    batch["instrument_ids"].to(device),
            "session_ids":       batch["session_ids"].to(device),
            "regime_labels":     batch["regime_label"].to(device),
            "volatility_labels": batch["volatility_label"].to(device),
            "structure_labels":  batch["structure_label"].to(device),
            "range_labels":      batch["range_label"].to(device),
        }
        optimizer.zero_grad()

        with torch.autocast("cuda", enabled=USE_AMP):  # FIX #5
            outputs = model(**kwargs)

        loss = outputs["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # FIX #6: per-task loss tracking
        for task in TASK_NAMES:
            key = f"{task}_loss"
            if key in outputs:
                task_loss_sum[task]   += outputs[key].item()
                task_loss_count[task] += 1

        for task in TASK_NAMES:
            preds = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            sentinel = LABEL_SENTINEL if task == "structure" else None
            correct, total = task_accuracy(preds, labels, sentinel)
            task_correct[task] += correct
            task_total[task]   += total

    avg_loss = total_loss / max(1, num_batches)
    task_acc  = {k: task_correct[k] / max(1, task_total[k]) for k in TASK_NAMES}
    task_loss = {k: task_loss_sum[k] / max(1, task_loss_count[k]) for k in TASK_NAMES}
    return avg_loss, task_acc, task_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, num_batches = 0, 0
    task_correct = {t: 0 for t in TASK_NAMES}
    task_total   = {t: 0 for t in TASK_NAMES}
    task_preds   = {t: [] for t in TASK_NAMES}
    task_loss_sum   = {t: 0.0 for t in TASK_NAMES}
    task_loss_count = {t: 0   for t in TASK_NAMES}

    for batch in loader:
        kwargs = {
            "features":          batch["features"].to(device),
            "candle_types":      batch["candle_types"].to(device),   # FIX #1
            "time_of_day":       batch["time_of_day"].to(device),
            "day_of_week":       batch["day_of_week"].to(device),
            "instrument_ids":    batch["instrument_ids"].to(device),
            "session_ids":       batch["session_ids"].to(device),
            "regime_labels":     batch["regime_label"].to(device),
            "volatility_labels": batch["volatility_label"].to(device),
            "structure_labels":  batch["structure_label"].to(device),
            "range_labels":      batch["range_label"].to(device),
        }
        with torch.autocast("cuda", enabled=USE_AMP):  # FIX #5
            outputs = model(**kwargs)

        total_loss += outputs["loss"].item()
        num_batches += 1

        # FIX #6: per-task loss tracking
        for task in TASK_NAMES:
            key = f"{task}_loss"
            if key in outputs:
                task_loss_sum[task]   += outputs[key].item()
                task_loss_count[task] += 1

        for task in TASK_NAMES:
            preds = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            sentinel = LABEL_SENTINEL if task == "structure" else None
            correct, total = task_accuracy(preds, labels, sentinel)
            task_correct[task] += correct
            task_total[task]   += total
            mask_for_preds = (labels != sentinel) if sentinel is not None else torch.ones_like(labels, dtype=torch.bool)
            task_preds[task].extend(preds[mask_for_preds].cpu().numpy().tolist())

    avg_loss  = total_loss / max(1, num_batches)
    task_acc  = {k: task_correct[k] / max(1, task_total[k]) for k in TASK_NAMES}
    task_loss = {k: task_loss_sum[k] / max(1, task_loss_count[k]) for k in TASK_NAMES}
    task_pred_counts = {k: Counter(task_preds[k]) for k in TASK_NAMES}
    return avg_loss, task_acc, task_pred_counts, task_loss


# =========================================================
# TRAINING LOOP
# =========================================================
print(f"\n{'='*60}")
print(f"  PRETRAINING — 5min | {EPOCHS} epochs | {len(train_loader)} batches/epoch")
print(f"  Context: {SEQ_LEN} bars × 5min = {SEQ_LEN*5/60:.1f}h")
print(f"  Overfitting:  max_ratio={MAX_RATIO}  ratio_patience={RATIO_PATIENCE}")
print(f"  Collapse:     acc<{MIN_TASK_ACC}  or  majority>{MAX_MAJORITY:.0%}")
print(f"{'='*60}\n")

best_val_loss    = float("inf")
patience_counter = 0
bad_ratio_counter = 0
stable_counter   = 0
history          = []
val_loss_history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc, train_task_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, device
    )
    val_loss, val_acc, val_pred_counts, val_task_loss = evaluate(model, val_loader, device)

    elapsed = time.time() - t0
    lr      = optimizer.param_groups[0]["lr"]
    ratio   = val_loss / train_loss if train_loss > 0 else 1.0

    status_str, status_level = gen_status(train_loss, val_loss)

    bad_ratio_counter = bad_ratio_counter + 1 if ratio > MAX_RATIO else 0

    val_loss_history.append(val_loss)
    if len(val_loss_history) >= STABLE_EPOCHS:
        recent    = val_loss_history[-STABLE_EPOCHS:]
        is_stable = np.std(recent) < 0.01 and status_level in ("ok", "slt")
    else:
        is_stable = False

    stable_counter = stable_counter + 1 if (is_stable and ratio <= MAX_RATIO) else 0

    collapse_warnings = []
    for task in TASK_NAMES:
        if val_acc[task] < MIN_TASK_ACC:
            collapse_warnings.append(f"{task[0].upper()}:{val_acc[task]:.0%}↓")
        collapsed, reason = check_collapse(val_pred_counts[task], TASK_NUM_CLASSES[task], task)
        if collapsed:
            collapse_warnings.append(f"{task[0].upper()}:⚠{reason}")

    saved = ""
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(),           ckpt_dir / "best_pretrained.pt")
        torch.save(model.backbone.state_dict(),  ckpt_dir / "best_backbone.pt")
        model.save_pretrained(str(ckpt_dir / "hf_model"))
        saved = "✅ SAVE"
    else:
        patience_counter += 1
        saved = f"⏳ {patience_counter}/{PATIENCE}"

    tags = [saved]
    if stable_counter >= STABLE_EPOCHS:
        tags.append(f"🔒×{stable_counter}")
    if collapse_warnings:
        tags.append(" ".join(collapse_warnings))

    # ── Line 1: combined loss + val accuracy + save status ──
    print(
        f"E{epoch:>3}/{EPOCHS} ({elapsed:.0f}s) lr={lr:.1e} | "
        f"TrL:{train_loss:.4f} VL:{val_loss:.4f} | "
        f"R:{val_acc['regime']:.3f} V:{val_acc['volatility']:.3f} "
        f"S:{val_acc['structure']:.3f} P:{val_acc['range']:.3f} | "
        f"{' '.join(tags)}"
    )
    # ── Line 2: overfitting status + train accuracy ──
    train_acc_str = " ".join(f"{t[0].upper()}:{train_acc[t]:.3f}" for t in TASK_NAMES)
    print(f"     {status_str} | TrAcc: {train_acc_str}")
    # ── Line 3: per-task val losses (FIX #6) ──
    tl_str = " ".join(f"{t[0].upper()}:{val_task_loss[t]:.3f}" for t in TASK_NAMES)
    tr_tl_str = " ".join(f"{t[0].upper()}:{train_task_loss[t]:.3f}" for t in TASK_NAMES)
    print(f"     VTaskL: {tl_str} | TrTaskL: {tr_tl_str}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss, "val_loss": val_loss,
        "ratio": ratio, "status": status_level,
        "train_acc": train_acc, "val_acc": val_acc,
        "train_task_loss": train_task_loss, "val_task_loss": val_task_loss,
        "lr": lr, "time": elapsed,
        "stable_counter": stable_counter,
        "collapse_warnings": collapse_warnings,
    })

    if bad_ratio_counter >= RATIO_PATIENCE:
        print(f"\n🛑 Ratio exceeded {MAX_RATIO} for {RATIO_PATIENCE} consecutive epochs — stopping")
        break

    if patience_counter >= PATIENCE:
        print(f"\n⏹ Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
        break

# =========================================================
# SAVE & SUMMARY
# =========================================================
with open(ckpt_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"  FINAL VALIDATION — PREDICTION DISTRIBUTIONS")
print(f"{'='*60}")

_, final_acc, final_preds, final_task_loss = evaluate(model, val_loader, device)
for task in TASK_NAMES:
    counts    = final_preds[task]
    total     = sum(counts.values())
    label_map = TASK_LABEL_MAPS[task]
    collapsed, reason = check_collapse(counts, TASK_NUM_CLASSES[task], task)
    flag = " 🚨 COLLAPSED" if collapsed else ""

    print(f"\n  {task} (acc={final_acc[task]:.3f}  loss={final_task_loss[task]:.3f}){flag}:")
    for cls in sorted(counts.keys()):
        name = label_map.get(cls, f"class_{cls}")
        pct  = counts[cls] / total * 100 if total > 0 else 0
        print(f"    {cls} ({name:>20s}): {counts[cls]:>8,d} ({pct:5.1f}%)")

print(f"\n{'='*60}")
print(f"  ✅ PRETRAINING COMPLETE — 5min")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Final ratio:   {history[-1]['ratio']:.3f} ({history[-1]['status']})")
print(f"  Stable epochs: {stable_counter}")
print(f"  Backbone:      {ckpt_dir / 'best_backbone.pt'}")
print(f"  Full model:    {ckpt_dir / 'best_pretrained.pt'}")
print(f"  HF model:      {ckpt_dir / 'hf_model'}")
print(f"{'='*60}")

# ---
# ## Cell 6: Training Curves
# ---
import json
import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path(CHECKPOINT_DIR) / "training_history.json"
with open(history_path) as f:
    history = json.load(f)

epochs     = [h["epoch"]      for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss   = [h["val_loss"]   for h in history]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Combined loss
axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=1.5)
axes[0].plot(epochs, val_loss,   'r-', label='Val',   linewidth=1.5)
best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Combined Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Per-task val accuracy
tasks  = ["regime", "volatility", "structure", "range"]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
for task, color in zip(tasks, colors):
    acc = [h["val_acc"][task] for h in history]
    axes[1].plot(epochs, acc, color=color, label=task.capitalize(), linewidth=1.5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Val Accuracy by Task")
axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)

# Per-task val losses (FIX #6)
for task, color in zip(tasks, colors):
    tl = [h["val_task_loss"][task] for h in history]
    axes[2].plot(epochs, tl, color=color, label=task.capitalize(), linewidth=1.5)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")
axes[2].set_title("Val Loss by Task")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best = min(history, key=lambda h: h["val_loss"])
print(f"\n📊 Best Epoch: {best['epoch']}")
print(f"   Val Loss: {best['val_loss']:.4f}")
for task in tasks:
    print(f"   {task:>12s}: acc={best['val_acc'][task]:.1%}  loss={best['val_task_loss'][task]:.3f}")

# ---
# ## Cell 7: Verify Backbone
# ---
import torch
from pathlib import Path
from futures_foundation import FFMConfig, FFMBackbone

ckpt_dir = Path(CHECKPOINT_DIR)

config   = FFMConfig.from_pretrained(str(ckpt_dir))
backbone = FFMBackbone(config)
backbone.load_state_dict(torch.load(ckpt_dir / "best_backbone.pt", map_location="cpu"))
backbone.eval()

batch_size, seq_len, num_feat = 4, SEQ_LEN, config.num_features

with torch.no_grad():
    embedding = backbone(
        features=torch.randn(batch_size, seq_len, num_feat),
        candle_types=torch.randint(0, 6, (batch_size, seq_len)),
        time_of_day=torch.rand(batch_size, seq_len),
        day_of_week=torch.randint(0, 5, (batch_size, seq_len)),
        instrument_ids=torch.zeros(batch_size, dtype=torch.long),
        session_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
    )

print(f"✅ Backbone loaded successfully!")
print(f"   Input:  ({batch_size}, {seq_len}, {num_feat}) + candle_types ({batch_size}, {seq_len})")
print(f"   Output: {embedding.shape} — {config.hidden_size}-dim embedding")
print(f"   Stats:  mean={embedding.mean():.4f}, std={embedding.std():.4f}")
print(f"\n🎯 Ready for fine-tuning!")
print(f"   Backbone: {ckpt_dir / 'best_backbone.pt'}")
