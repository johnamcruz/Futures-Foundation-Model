# # 🚀 Futures Foundation Model — Pretraining Pipeline
#
# **Step 1:** Prepare data (derive 52 features + generate 4 labels from raw OHLCV)
# **Step 2:** Pretrain backbone (4-task self-supervised learning)
#
# ## Prerequisites
# - Raw 3-min OHLCV CSVs on Google Drive
# - Columns: `datetime, open, high, low, close, volume`
# - Files named like: `ES_3min.csv`, `NQ_3min.csv`, `GC_3min.csv`, etc.
# - GPU runtime (T4 is fine)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — 3min Timeframe | Foundation Model Pretraining
# ═══════════════════════════════════════════════════════════════

RAW_DATA_DIR      = "/content/drive/MyDrive/Futures Data/3min/"
PREPARED_DATA_DIR = "/content/drive/MyDrive/AI_Cache/3min_FFM_Prepared/"
CHECKPOINT_DIR    = "/content/drive/MyDrive/AI_Cache/3min_FFM_Checkpoints/"

# ─── Training Hyperparameters ───
EPOCHS        = 50
BATCH_SIZE    = 256
LEARNING_RATE = 1e-4    # halved again from 2e-4 — the gap is widening too fast
                        # (was 1e-3 → too aggressive, caused CRIT overfitting by epoch 7)
SEQ_LEN       = 106     # 106 bars × 3min = ~5.3 hours of market context
                        # (was 64 × 5min — keeping same real-time window)
VAL_RATIO     = 0.20    # 20% val — critical for reliable early stopping signal
                        # (was 0.15 → produced only ~2,700 val sequences vs 488k train)
PATIENCE      = 10      # Foundation model: more forgiving, maximize regime exposure
WARMUP_STEPS  = 8000    # Longer warmup for stable pretraining
GRAD_CLIP     = 1.0

# ─── Overfitting / Stability Thresholds ───
MAX_RATIO      = 1.25   # Slightly more tolerant for pretraining
RATIO_PATIENCE = 12     # More forgiving before killing run

# ─── Model Architecture ───
HIDDEN_SIZE       = 256
NUM_LAYERS        = 6
NUM_HEADS         = 8
INTERMEDIATE_SIZE = 512

SEED = 42

print("✅ Configuration set — 3min Foundation Model Pretraining")
print(f"   Raw data:    {RAW_DATA_DIR}")
print(f"   Prepared:    {PREPARED_DATA_DIR}")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   SEQ_LEN:     {SEQ_LEN} bars × 3min = ~{SEQ_LEN * 3 / 60:.1f} hours context")
print(f"   LR:          {LEARNING_RATE}")
print(f"   Val ratio:   {VAL_RATIO:.0%}")

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
    ['git', 'clone', 'https://github.com/johnamcruz/Futures-Foundation-Model.git',
     '/content/Futures-Foundation-Model'],
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
    print(f"   Expected format: ES_3min.csv, NQ_3min.csv, GC_3min.csv, etc.")
    print(f"   Expected columns: datetime, open, high, low, close, volume")
else:
    print(f"📊 Found {len(data_files)} data files:\n")
    for f in data_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name:30s} ({size_mb:.1f} MB)")

    if data_files[0].suffix == ".csv":
        df_peek = pd.read_csv(data_files[0], nrows=5)
    else:
        df_peek = pd.read_parquet(data_files[0]).head(5)

    print(f"\n📋 Columns in {data_files[0].name}:")
    print(f"   {list(df_peek.columns)}")
    print(f"\n📋 First 3 rows:")
    print(df_peek.head(3).to_string())

# ---
# ## Cell 3: STEP 1 — Prepare Data
#
# Derives 52 features + generates 4 pretraining labels from raw OHLCV.
# Saves as parquet for fast loading. **Run once** (or when you change feature/label logic).
#
# Feature groups (52 total):
#   Group 1 — Bar Anatomy (8):          Body/wick ratios, range in ATR units
#   Group 2 — Returns & Momentum (8):   Multi-horizon returns, acceleration
#   Group 3 — Volume Dynamics (6):      Relative volume, delta proxy
#   Group 4 — Volatility Measures (6):  ATR z-score, range ratios, realized vol
#   Group 5 — Session Context (5):      Distance from session OHLC + VWAP
#   Group 6 — Market Structure (9):     Swing distances, range position
#   Group 7 — CRT Sweep State (10):     1H/4H prior-candle liquidity sweep events
#                                        (bar-frequency-agnostic expiry windows)
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

NUM_FEATURES = len(get_model_feature_columns())

print(f"\n{'='*60}")
print(f"  STEP 1: DATA PREPARATION")
print(f"  Input:  {raw_dir} ({len(data_files)} files)")
print(f"  Output: {out_dir}")
print(f"  Features: {NUM_FEATURES} (incl. 10 CRT sweep state features)")
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

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

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

    # derive_features auto-detects bar frequency for CRT expiry windows
    features_df = derive_features(df, instrument=instrument)

    feature_cols = get_model_feature_columns()
    valid_count = features_df[feature_cols].notna().all(axis=1).sum()
    nan_count = len(features_df) - valid_count

    # Report per-group NaN budget
    swp_cols = [c for c in feature_cols if c.startswith("swp_")]
    swp_valid = features_df[swp_cols].notna().all(axis=1).sum()

    print(f"  Derived {len(feature_cols)} features ({len(swp_cols)} CRT sweep features)")
    print(f"  Valid rows: {valid_count:,} / {len(features_df):,} ({valid_count/len(features_df)*100:.1f}%)")
    print(f"  CRT sweep valid: {swp_valid:,} / {len(features_df):,} ({swp_valid/len(features_df)*100:.1f}%)")
    if nan_count > 0:
        print(f"  NaN rows (warmup): {nan_count:,} — expected from rolling windows")

    labels_df = generate_all_labels(features_df)
    print_label_distribution(labels_df)

    features_path = out_dir / f"{instrument}_features.parquet"
    labels_path   = out_dir / f"{instrument}_labels.parquet"
    features_df.to_parquet(features_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    elapsed = time.time() - t0
    feat_size  = features_path.stat().st_size / 1024 / 1024
    label_size = labels_path.stat().st_size / 1024 / 1024

    print(f"\n  ✓ Saved: {features_path.name} ({feat_size:.1f} MB) + {labels_path.name} ({label_size:.1f} MB)")
    print(f"    Time: {elapsed:.1f}s")

    summary[instrument] = {
        "raw_bars": len(df),
        "valid_bars": int(valid_count),
        "date_start": str(df["datetime"].iloc[0]),
        "date_end":   str(df["datetime"].iloc[-1]),
    }

config_path = out_dir / "prep_config.json"
with open(config_path, "w") as f:
    json.dump({
        "num_features": NUM_FEATURES,
        "feature_columns": get_model_feature_columns(),
        "instruments": summary,
    }, f, indent=2)

total_elapsed = time.time() - total_start
total_bars  = sum(s["raw_bars"]   for s in summary.values())
total_valid = sum(s["valid_bars"] for s in summary.values())

print(f"\n{'='*60}")
print(f"  ✅ STEP 1 COMPLETE")
print(f"  Instruments: {len(summary)}")
print(f"  Total bars:  {total_bars:,} ({total_valid:,} valid)")
print(f"  Time: {total_elapsed:.1f}s")
print(f"  Output: {out_dir}")
print(f"{'='*60}")

# ---
# ## Cell 4: STEP 2 — Pretrain Backbone
#
# Trains the FFM transformer on 4 self-supervised tasks.
# The backbone encodes 52-feature sequences into 256-dim market context embeddings.
#
# Training infrastructure:
#   ✅ Overfitting detection: train/val ratio with severity levels
#   ✅ Per-task accuracy + collapse detection every epoch
#   ✅ Stability tracking: consecutive stable epochs counter
#   ✅ Ratio patience: kill training if overfitting persists N epochs
#   ✅ Majority-class collapse detection per task
# ---

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
    FFMDataset,
    FFMMultiInstrumentDataset,
    create_dataloaders,
    REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

MAX_MAJORITY = 0.95
MIN_TASK_ACC = 0.10
STABLE_EPOCHS = 3

TASK_NAMES = ["regime", "volatility", "structure", "range"]
TASK_LABEL_MAPS = {
    "regime":     REGIME_LABELS,
    "volatility": VOLATILITY_LABELS,
    "structure":  STRUCTURE_LABELS,
    "range":      RANGE_LABELS,
}
TASK_NUM_CLASSES = {"regime": 4, "volatility": 4, "structure": 3, "range": 5}


# =========================================================
# Temporal train/val split
# Splits on bars at the cutoff, creates FFMDataset from each
# half independently → val sequences ≈ VAL_RATIO × total.
# =========================================================
def fixed_temporal_split(features_df, labels_df, val_ratio, seq_len):
    n_bars = len(features_df)
    cutoff = int(n_bars * (1 - val_ratio))
    cutoff = min(cutoff, n_bars - seq_len - 1)

    train_ds = FFMDataset(
        features_df.iloc[:cutoff].reset_index(drop=True),
        labels_df.iloc[:cutoff].reset_index(drop=True),
        seq_len=seq_len,
    )
    val_ds = FFMDataset(
        features_df.iloc[cutoff:].reset_index(drop=True),
        labels_df.iloc[cutoff:].reset_index(drop=True),
        seq_len=seq_len,
    )
    return train_ds, val_ds


# =========================================================
# Helpers
# =========================================================
def gen_status(train_loss, val_loss):
    r = val_loss / train_loss if train_loss > 0 else 1.0
    if   r > 1.20: return f"🚨 CRIT ({r:.2f})", "crit"
    elif r > 1.15: return f"⚠️  SEV ({r:.2f})",  "sev"
    elif r > 1.12: return f"⚠️  MOD ({r:.2f})",  "mod"
    elif r > 1.08: return f"ℹ️  SLT ({r:.2f})",  "slt"
    elif r < 0.85: return f"ℹ️  UND ({r:.2f})",  "und"
    else:          return f"✅ OK ({r:.2f})",     "ok"


def check_collapse(preds_counter, num_classes, task_name):
    total = sum(preds_counter.values())
    if total == 0:
        return True, "no predictions"
    for cls, count in preds_counter.items():
        if count / total > MAX_MAJORITY:
            return True, f"class {cls} = {count/total:.0%}"
    return False, ""


# =========================================================
# Load data
# =========================================================
prepared_dir = Path(PREPARED_DATA_DIR)
feature_files = sorted(prepared_dir.glob("*_features.parquet"))

if not feature_files:
    raise FileNotFoundError(f"No prepared data in {prepared_dir}. Run Cell 3 first.")

print(f"\n{'='*60}")
print(f"  LOADING PREPARED DATA")
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
    labels_df   = pd.read_parquet(label_path)
    load_time   = time.time() - t0

    train_ds, val_ds = fixed_temporal_split(
        features_df, labels_df, val_ratio=VAL_RATIO, seq_len=SEQ_LEN,
    )
    print(f"  {instrument}: {len(features_df):,} bars → "
          f"{len(train_ds):,} train / {len(val_ds):,} val ({load_time:.1f}s)")
    train_datasets.append(train_ds)
    val_datasets.append(val_ds)

combined_train = FFMMultiInstrumentDataset(train_datasets)
combined_val   = FFMMultiInstrumentDataset(val_datasets)
print(f"\n  Total: {len(combined_train):,} train / {len(combined_val):,} val")

actual_val_pct = len(combined_val) / (len(combined_train) + len(combined_val))
print(f"\n  Val split check: {actual_val_pct:.1%} (expected ~{VAL_RATIO:.0%})")
if actual_val_pct < 0.05:
    raise RuntimeError(
        f"Val set is only {actual_val_pct:.1%} — fixed_temporal_split may have failed."
    )
print(f"  ✅ Val split looks correct\n")

train_loader, val_loader = create_dataloaders(
    combined_train, combined_val, batch_size=BATCH_SIZE, num_workers=2,
)

# =========================================================
# Model
# =========================================================
num_features = len(get_model_feature_columns())  # 52 (incl. CRT sweep features)

config = FFMConfig(
    num_features=num_features,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
)

model = FFMForPretraining(config).to(device)
total_params = sum(p.numel() for p in model.parameters())

print(f"Model: {total_params:,} parameters")
print(f"  Features:    {num_features} (42 base + 10 CRT sweep state)")
print(f"  Hidden:      {HIDDEN_SIZE}")
print(f"  Layers:      {NUM_LAYERS}")
print(f"  Heads:       {NUM_HEADS}")
print(f"  FF dim:      {INTERMEDIATE_SIZE}")

# =========================================================
# Optimizer & scheduler
# =========================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.05,
)
total_steps = len(train_loader) * EPOCHS

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

ckpt_dir = Path(CHECKPOINT_DIR)
ckpt_dir.mkdir(parents=True, exist_ok=True)
config.save_pretrained(str(ckpt_dir))

with open(ckpt_dir / "train_args.json", "w") as f:
    json.dump({
        "timeframe":        "3min",
        "num_features":     num_features,
        "epochs":           EPOCHS,
        "batch_size":       BATCH_SIZE,
        "lr":               LEARNING_RATE,
        "seq_len":          SEQ_LEN,
        "hidden_size":      HIDDEN_SIZE,
        "num_layers":       NUM_LAYERS,
        "num_heads":        NUM_HEADS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "warmup_steps":     WARMUP_STEPS,
        "grad_clip":        GRAD_CLIP,
        "val_ratio":        VAL_RATIO,
        "patience":         PATIENCE,
        "seed":             SEED,
        "max_ratio":        MAX_RATIO,
        "ratio_patience":   RATIO_PATIENCE,
        "stable_epochs":    STABLE_EPOCHS,
        "features": {
            "total": num_features,
            "base_ohlcv": 42,
            "crt_sweep_state": 10,
            "crt_timeframes": ["1H", "4H"],
        },
    }, f, indent=2)


# =========================================================
# Training functions
# =========================================================
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, num_batches = 0, 0
    task_correct = {t: 0 for t in TASK_NAMES}
    task_total   = {t: 0 for t in TASK_NAMES}

    for batch in loader:
        kwargs = {
            "features":          batch["features"].to(device),
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
        outputs = model(**kwargs)
        outputs["loss"].backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += outputs["loss"].item()
        num_batches += 1
        for task in TASK_NAMES:
            preds  = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (preds == labels).sum().item()
            task_total[task]   += labels.size(0)

    avg_loss = total_loss / max(1, num_batches)
    task_acc = {k: task_correct[k] / max(1, task_total[k]) for k in TASK_NAMES}
    return avg_loss, task_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, num_batches = 0, 0
    task_correct  = {t: 0  for t in TASK_NAMES}
    task_total    = {t: 0  for t in TASK_NAMES}
    task_preds    = {t: [] for t in TASK_NAMES}

    for batch in loader:
        kwargs = {
            "features":          batch["features"].to(device),
            "time_of_day":       batch["time_of_day"].to(device),
            "day_of_week":       batch["day_of_week"].to(device),
            "instrument_ids":    batch["instrument_ids"].to(device),
            "session_ids":       batch["session_ids"].to(device),
            "regime_labels":     batch["regime_label"].to(device),
            "volatility_labels": batch["volatility_label"].to(device),
            "structure_labels":  batch["structure_label"].to(device),
            "range_labels":      batch["range_label"].to(device),
        }
        outputs = model(**kwargs)
        total_loss  += outputs["loss"].item()
        num_batches += 1
        for task in TASK_NAMES:
            preds  = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (preds == labels).sum().item()
            task_total[task]   += labels.size(0)
            task_preds[task].extend(preds.cpu().numpy().tolist())

    avg_loss       = total_loss / max(1, num_batches)
    task_acc       = {k: task_correct[k] / max(1, task_total[k]) for k in TASK_NAMES}
    task_pred_counts = {k: Counter(task_preds[k]) for k in TASK_NAMES}
    return avg_loss, task_acc, task_pred_counts


# =========================================================
# Training loop
# =========================================================
print(f"\n{'='*60}")
print(f"  PRETRAINING — 3min | {EPOCHS} epochs | {len(train_loader)} batches/epoch")
print(f"  Features:    {num_features} (42 base + 10 CRT sweep state)")
print(f"  Overfitting: max_ratio={MAX_RATIO}  ratio_patience={RATIO_PATIENCE}")
print(f"  Stability:   {STABLE_EPOCHS} consecutive stable epochs")
print(f"  Collapse:    task_acc<{MIN_TASK_ACC} or majority>{MAX_MAJORITY:.0%}")
print(f"{'='*60}\n")

best_val_loss     = float("inf")
patience_counter  = 0
bad_ratio_counter = 0
stable_counter    = 0
history           = []
val_loss_history  = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc               = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc, val_pred_counts  = evaluate(model, val_loader, device)

    elapsed      = time.time() - t0
    lr           = optimizer.param_groups[0]["lr"]
    ratio        = val_loss / train_loss if train_loss > 0 else 1.0
    status_str, status_level = gen_status(train_loss, val_loss)

    bad_ratio_counter = bad_ratio_counter + 1 if ratio > MAX_RATIO else 0

    val_loss_history.append(val_loss)
    if len(val_loss_history) >= STABLE_EPOCHS:
        is_stable = np.std(val_loss_history[-STABLE_EPOCHS:]) < 0.01 and status_level in ("ok", "slt")
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

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), ckpt_dir / "best_pretrained.pt")
        torch.save(model.backbone.state_dict(), ckpt_dir / "best_backbone.pt")
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

    print(
        f"E{epoch:>3}/{EPOCHS} ({elapsed:.0f}s) lr={lr:.1e} | "
        f"TrL:{train_loss:.4f} VL:{val_loss:.4f} | "
        f"R:{val_acc['regime']:.3f} V:{val_acc['volatility']:.3f} "
        f"S:{val_acc['structure']:.3f} P:{val_acc['range']:.3f} | "
        f"{' '.join(tags)}"
    )
    train_acc_str = " ".join(f"{t[0].upper()}:{train_acc[t]:.3f}" for t in TASK_NAMES)
    print(f"     {status_str} | Train acc: {train_acc_str}")

    history.append({
        "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
        "ratio": ratio, "status": status_level,
        "train_acc": train_acc, "val_acc": val_acc,
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
# Save & summary
# =========================================================
with open(ckpt_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"  FINAL VALIDATION — PREDICTION DISTRIBUTIONS")
print(f"{'='*60}")

_, final_acc, final_preds = evaluate(model, val_loader, device)
for task in TASK_NAMES:
    counts    = final_preds[task]
    total     = sum(counts.values())
    label_map = TASK_LABEL_MAPS[task]
    collapsed, reason = check_collapse(counts, TASK_NUM_CLASSES[task], task)
    flag = " 🚨 COLLAPSED" if collapsed else ""

    print(f"\n  {task} (acc={final_acc[task]:.3f}){flag}:")
    for cls in sorted(counts.keys()):
        name = label_map.get(cls, f"class_{cls}")
        pct  = counts[cls] / total * 100 if total > 0 else 0
        print(f"    {cls} ({name:>20s}): {counts[cls]:>8,d} ({pct:5.1f}%)")

print(f"\n{'='*60}")
print(f"  ✅ PRETRAINING COMPLETE — 3min | {num_features} features")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Final ratio:   {history[-1]['ratio']:.3f} ({history[-1]['status']})")
print(f"  Stable epochs: {stable_counter}")
print(f"  Backbone:      {ckpt_dir / 'best_backbone.pt'}")
print(f"  Full model:    {ckpt_dir / 'best_pretrained.pt'}")
print(f"  HF model:      {ckpt_dir / 'hf_model'}")
print(f"{'='*60}")

# ---
# ## Cell 5: Training Curves
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

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=1.5)
axes[0].plot(epochs, val_loss,   'r-', label='Val',   linewidth=1.5)
best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Per-task val accuracy
tasks  = ["regime", "volatility", "structure", "range"]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
for task, color in zip(tasks, colors):
    acc = [h["val_acc"][task] for h in history]
    axes[1].plot(epochs, acc, color=color, label=task.capitalize(), linewidth=1.5)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy by Task")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

best = min(history, key=lambda h: h["val_loss"])
print(f"\n📊 Best Epoch: {best['epoch']}")
print(f"   Val Loss: {best['val_loss']:.4f}")
for task in tasks:
    print(f"   {task:>12s}: {best['val_acc'][task]:.1%}")

# ---
# ## Cell 6: Verify Backbone
# Quick test that the trained backbone loads and produces valid embeddings.
# ---
import torch
from pathlib import Path
from futures_foundation import FFMConfig, FFMBackbone, get_model_feature_columns

ckpt_dir = Path(CHECKPOINT_DIR)

config   = FFMConfig.from_pretrained(str(ckpt_dir))
backbone = FFMBackbone(config)
backbone.load_state_dict(torch.load(ckpt_dir / "best_backbone.pt", map_location="cpu"))
backbone.eval()

batch_size = 4
num_feat   = config.num_features  # 52

with torch.no_grad():
    embedding = backbone(
        features=torch.randn(batch_size, SEQ_LEN, num_feat),
        time_of_day=torch.rand(batch_size, SEQ_LEN),
        day_of_week=torch.randint(0, 5, (batch_size, SEQ_LEN)),
        instrument_ids=torch.zeros(batch_size, dtype=torch.long),
        session_ids=torch.ones(batch_size, SEQ_LEN, dtype=torch.long),
    )

print(f"✅ Backbone loaded successfully!")
print(f"   Input:    ({batch_size}, {SEQ_LEN}, {num_feat})  — 52 features incl. CRT sweeps")
print(f"   Output:   {embedding.shape}  — {config.hidden_size}-dim context embedding")
print(f"   Stats:    mean={embedding.mean():.4f}  std={embedding.std():.4f}")
print(f"\n🎯 Ready for fine-tuning!")
print(f"   Backbone: {ckpt_dir / 'best_backbone.pt'}")
