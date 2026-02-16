"""
Stage 1: Pretraining Script — Trains FFM backbone on all instruments.

Usage:
    python scripts/pretrain.py \
        --data-dir data/raw/ --output-dir checkpoints/pretrained/ \
        --epochs 50 --batch-size 256 --lr 1e-3 --seq-len 64
"""

import argparse, os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from futures_foundation import (
    FFMConfig, FFMForPretraining, derive_features, generate_all_labels,
    get_model_feature_columns, print_label_distribution,
    FFMDataset, FFMMultiInstrumentDataset, temporal_train_val_split, create_dataloaders,
)


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain FFM backbone")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/pretrained/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_and_prepare_data(data_dir, seq_len, val_ratio):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"\nFound {len(csv_files)} data files:")
    train_datasets, val_datasets = [], []

    for csv_path in sorted(csv_files):
        instrument = csv_path.stem.split("_")[0].upper()
        print(f"\n  Processing {instrument} from {csv_path.name}...")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        if "date" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"date": "datetime"})
        print(f"    Loaded {len(df):,} bars")

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
    print(f"\n  Combined: {len(combined_train):,} train | {len(combined_val):,} val")
    return combined_train, combined_val


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
        features = batch["features"].to(device)
        kwargs = {
            "features": features,
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
        loss = outputs["loss"]
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        for task in task_correct:
            logits = outputs[f"{task}_logits"]
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (logits.argmax(-1) == labels).sum().item()
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
            logits = outputs[f"{task}_logits"]
            labels = batch[f"{task}_label"].to(device)
            task_correct[task] += (logits.argmax(-1) == labels).sum().item()
            task_total[task] += labels.size(0)

    return total_loss / max(1, num_batches), {k: task_correct[k] / max(1, task_total[k]) for k in task_correct}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    print(f"Using device: {device}")

    train_dataset, val_dataset = load_and_prepare_data(args.data_dir, args.seq_len, args.val_ratio)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size, num_workers=0)

    config = FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
    )
    model = FFMForPretraining(config).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, len(train_loader) * args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(output_dir))

    best_val_loss, patience_counter = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs} ({time.time()-t0:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc — Regime: {train_acc['regime']:.3f} Vol: {train_acc['volatility']:.3f} Struct: {train_acc['structure']:.3f} Range: {train_acc['range']:.3f}")
        print(f"  Val   Acc — Regime: {val_acc['regime']:.3f} Vol: {val_acc['volatility']:.3f} Struct: {val_acc['structure']:.3f} Range: {val_acc['range']:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_pretrained.pt")
            torch.save(model.backbone.state_dict(), output_dir / "best_backbone.pt")
            model.save_pretrained(str(output_dir / "hf_model"))
            print(f"  ✓ Saved best (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nPretraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Backbone: {output_dir / 'best_backbone.pt'}")

if __name__ == "__main__":
    main()