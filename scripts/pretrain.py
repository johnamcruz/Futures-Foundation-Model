"""
Step 2: Pretraining — Train the FFM backbone.

Two modes:
    A) From prepared parquet (fast, recommended):
        python scripts/pretrain.py --data-dir data/prepared/ --from-prepared

    B) From raw OHLCV (derives features on the fly, slower):
        python scripts/pretrain.py --data-dir data/raw/
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from futures_foundation import (
    FFMConfig,
    FFMForPretraining,
    derive_features,
    generate_all_labels,
    get_model_feature_columns,
    print_label_distribution,
    FFMDataset,
    FFMMultiInstrumentDataset,
    temporal_train_val_split,
    create_dataloaders,
)


def parse_args():
    p = argparse.ArgumentParser(description="Step 2: Pretrain FFM backbone")
    # Data
    p.add_argument("--data-dir", type=str, required=True, help="Directory with data files")
    p.add_argument("--from-prepared", action="store_true", help="Load from prepared parquet (output of prepare_data.py)")
    # Model
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=512)
    # Training
    p.add_argument("--output-dir", type=str, default="checkpoints/pretrained/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# =============================================================================
# Data Loading
# =============================================================================


def load_prepared_data(data_dir, seq_len, val_ratio):
    """Load pre-computed features and labels from parquet (fast)."""
    data_dir = Path(data_dir)
    feature_files = sorted(data_dir.glob("*_features.parquet"))

    if not feature_files:
        raise FileNotFoundError(
            f"No *_features.parquet files in {data_dir}. "
            f"Run prepare_data.py first, or remove --from-prepared flag."
        )

    print(f"\nLoading prepared data from {data_dir}")
    train_datasets, val_datasets = [], []

    for feat_path in feature_files:
        instrument = feat_path.stem.replace("_features", "")
        label_path = data_dir / f"{instrument}_labels.parquet"

        if not label_path.exists():
            print(f"  ⚠ Skipping {instrument} — no labels file")
            continue

        t0 = time.time()
        features_df = pd.read_parquet(feat_path)
        labels_df = pd.read_parquet(label_path)
        load_time = time.time() - t0

        train_ds, val_ds = temporal_train_val_split(
            features_df, labels_df, val_ratio=val_ratio, seq_len=seq_len,
        )
        print(f"  {instrument}: {len(features_df):,} bars → "
              f"{len(train_ds):,} train / {len(val_ds):,} val "
              f"({load_time:.1f}s)")

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    combined_train = FFMMultiInstrumentDataset(train_datasets)
    combined_val = FFMMultiInstrumentDataset(val_datasets)
    print(f"\n  Total: {len(combined_train):,} train / {len(combined_val):,} val")
    return combined_train, combined_val


def load_raw_data(data_dir, seq_len, val_ratio):
    """Load raw OHLCV and derive features on the fly (slower)."""
    data_dir = Path(data_dir)
    data_files = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
    )
    data_files = [f for f in data_files if "_features" not in f.stem and "_labels" not in f.stem]

    if not data_files:
        raise FileNotFoundError(f"No data files in {data_dir}")

    print(f"\nLoading raw data from {data_dir}")
    train_datasets, val_datasets = [], []

    for data_path in data_files:
        instrument = data_path.stem.split("_")[0].upper()
        print(f"\n  Processing {instrument}...")

        df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
        df.columns = df.columns.str.strip().str.lower()
        if "date" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"date": "datetime"})
        print(f"    {len(df):,} bars")

        features_df = derive_features(df, instrument=instrument)
        labels_df = generate_all_labels(features_df)
        print_label_distribution(labels_df)

        train_ds, val_ds = temporal_train_val_split(
            features_df, labels_df, val_ratio=val_ratio, seq_len=seq_len,
        )
        print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,}")
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    combined_train = FFMMultiInstrumentDataset(train_datasets)
    combined_val = FFMMultiInstrumentDataset(val_datasets)
    print(f"\n  Total: {len(combined_train):,} train / {len(combined_val):,} val")
    return combined_train, combined_val


# =============================================================================
# Training
# =============================================================================


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss, num_batches = 0, 0
    task_correct = {t: 0 for t in ["regime", "volatility", "structure", "range"]}
    task_total = {t: 0 for t in task_correct}

    for batch in loader:
        kwargs = {
            "features": batch["features"].to(device),
            "time_of_day": batch["time_of_day"].to(device),
            "day_of_week": batch["day_of_week"].to(device),
            "instrument_ids": batch["instrument_ids"].to(device),
            "session_ids": batch["session_ids"].to(device),
            "regime_labels": batch["regime_label"].to(device),
            "volatility_labels": batch["volatility_label"].to(device),
            "structure_labels": batch["structure_label"].to(device),
            "range_labels": batch["range_label"].to(device),
        }
        optimizer.zero_grad()
        outputs = model(**kwargs)
        outputs["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += outputs["loss"].item()
        num_batches += 1
        for task in task_correct:
            preds = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (preds == labels).sum().item()
            task_total[task] += labels.size(0)

    return total_loss / max(1, num_batches), {k: task_correct[k] / max(1, task_total[k]) for k in task_correct}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, num_batches = 0, 0
    task_correct = {t: 0 for t in ["regime", "volatility", "structure", "range"]}
    task_total = {t: 0 for t in task_correct}

    for batch in loader:
        kwargs = {
            "features": batch["features"].to(device),
            "time_of_day": batch["time_of_day"].to(device),
            "day_of_week": batch["day_of_week"].to(device),
            "instrument_ids": batch["instrument_ids"].to(device),
            "session_ids": batch["session_ids"].to(device),
            "regime_labels": batch["regime_label"].to(device),
            "volatility_labels": batch["volatility_label"].to(device),
            "structure_labels": batch["structure_label"].to(device),
            "range_labels": batch["range_label"].to(device),
        }
        outputs = model(**kwargs)
        total_loss += outputs["loss"].item()
        num_batches += 1
        for task in task_correct:
            preds = outputs[f"{task}_logits"].argmax(-1)
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (preds == labels).sum().item()
            task_total[task] += labels.size(0)

    return total_loss / max(1, num_batches), {k: task_correct[k] / max(1, task_total[k]) for k in task_correct}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    # Load data
    print(f"\n{'='*60}")
    print(f"  LOADING DATA {'(prepared parquet)' if args.from_prepared else '(raw OHLCV)'}")
    print(f"{'='*60}")

    if args.from_prepared:
        train_dataset, val_dataset = load_prepared_data(args.data_dir, args.seq_len, args.val_ratio)
    else:
        train_dataset, val_dataset = load_raw_data(args.data_dir, args.seq_len, args.val_ratio)

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    config = FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
    )
    model = FFMForPretraining(config).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, len(train_loader) * args.epochs)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(output_dir))
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training
    print(f"\n{'='*60}")
    print(f"  PRETRAINING — {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{args.epochs} ({elapsed:.0f}s) | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Acc — R:{val_acc['regime']:.3f} V:{val_acc['volatility']:.3f} "
            f"S:{val_acc['structure']:.3f} P:{val_acc['range']:.3f}",
            end="",
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "time": elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_pretrained.pt")
            torch.save(model.backbone.state_dict(), output_dir / "best_backbone.pt")
            model.save_pretrained(str(output_dir / "hf_model"))
            print(" ✓", end="")
        else:
            patience_counter += 1

        print()
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  DONE — Best val loss: {best_val_loss:.4f}")
    print(f"  Backbone: {output_dir / 'best_backbone.pt'}")
    print(f"{'='*60}")
    print(f"\n  Next: python scripts/finetune.py --backbone {output_dir / 'best_backbone.pt'} --strategy orb --data-dir data/orb_labeled/")


if __name__ == "__main__":
    main()