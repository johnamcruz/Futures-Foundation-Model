"""
Stage 2: Fine-Tuning Script — Fine-tune FFM backbone for a specific strategy.

Usage:
    python scripts/finetune.py \
        --backbone checkpoints/pretrained/best_backbone.pt \
        --strategy orb --data-dir data/orb_labeled/ \
        --output-dir checkpoints/orb/ --freeze-ratio 0.66
"""

import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from futures_foundation import (
    FFMConfig, FFMForClassification, derive_features,
    get_model_feature_columns, FFMDataset, create_dataloaders,
)

STRATEGY_CONFIGS = {
    "orb": {"num_labels": 3, "label_names": ["BUY", "SELL", "HOLD"]},
    "ict_cisd": {"num_labels": 3, "label_names": ["BULLISH", "BEARISH", "NONE"]},
    "mean_reversion": {"num_labels": 3, "label_names": ["LONG", "SHORT", "FLAT"]},
    "regime": {"num_labels": 4, "label_names": ["TREND_UP", "TREND_DOWN", "ROTATIONAL", "VOLATILE"]},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=None)
    p.add_argument("--strategy", type=str, required=True, choices=STRATEGY_CONFIGS.keys())
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--label-column", type=str, default="label")
    p.add_argument("--output-dir", type=str, default="checkpoints/finetune/")
    p.add_argument("--freeze-ratio", type=float, default=0.66)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    strategy = STRATEGY_CONFIGS[args.strategy]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")

    config_dir = args.config_dir or str(Path(args.backbone).parent)
    try:
        config = FFMConfig.from_pretrained(config_dir)
    except Exception:
        config = FFMConfig()

    # Load & prepare data
    data_dir = Path(args.data_dir)
    all_features, all_labels = [], []
    for csv_path in sorted(data_dir.glob("*.csv")):
        instrument = csv_path.stem.split("_")[0].upper()
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        features_df = derive_features(df, instrument=instrument)
        if args.label_column in df.columns:
            all_features.append(features_df)
            all_labels.append(df[args.label_column].astype(int).reindex(features_df.index))

    features_combined = pd.concat(all_features, ignore_index=True)
    labels_combined = pd.concat(all_labels, ignore_index=True)
    labels_df = pd.DataFrame({"label": labels_combined})

    split_idx = int(len(features_combined) * (1 - args.val_ratio))
    train_ds = FFMDataset(features_combined.iloc[:split_idx], labels_df.iloc[:split_idx], seq_len=args.seq_len)
    val_ds = FFMDataset(features_combined.iloc[split_idx:], labels_df.iloc[split_idx:], seq_len=args.seq_len, stride=args.seq_len // 2)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=args.batch_size, num_workers=0)

    # Model
    model = FFMForClassification(config, num_labels=strategy["num_labels"])
    model.load_backbone(args.backbone)
    model.freeze_backbone(freeze_ratio=args.freeze_ratio)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss, patience_counter = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(features=features, time_of_day=batch["time_of_day"].to(device),
                        day_of_week=batch["day_of_week"].to(device), instrument_ids=batch["instrument_ids"].to(device),
                        session_ids=batch["session_ids"].to(device))
            loss = loss_fn(out["logits"], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            correct += (out["logits"].argmax(-1) == labels).sum().item()
            total += labels.size(0)

        # Validate
        model.eval()
        val_loss, all_preds, all_labels_list = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                out = model(features=features, time_of_day=batch["time_of_day"].to(device),
                            day_of_week=batch["day_of_week"].to(device), instrument_ids=batch["instrument_ids"].to(device),
                            session_ids=batch["session_ids"].to(device))
                val_loss += loss_fn(out["logits"], labels).item()
                all_preds.extend(out["logits"].argmax(-1).cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels_list))
        print(f"Epoch {epoch} — Train: {train_loss/len(train_loader):.4f} ({correct/total:.3f}) | Val: {val_loss:.4f} ({val_acc:.3f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"best_{args.strategy}.pt")
            model.save_pretrained(str(output_dir / "hf_model"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(classification_report(all_labels_list, all_preds, target_names=strategy["label_names"], zero_division=0))

if __name__ == "__main__":
    main()