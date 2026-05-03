"""
diagnostic_pretrain.py — Check pretrain backbone task head health.

Loads best_pretrained.pt from a checkpoint directory and runs the 4 pretraining
heads on prepared data to check for prediction collapse.

Usage:
    python3 utility/diagnostic_pretrain.py <checkpoint_dir> <prepared_data_dir>

Example:
    python3 utility/diagnostic_pretrain.py \
        /path/to/AI_Cache/5min_FFM_Checkpoints_v6 \
        /path/to/AI_Cache/5min_FFM_Prepared_v4

Healthy signs:
    - Regime   > 40%  (baseline 25%)
    - Vol      > 35%  (baseline 25%)
    - Structure > 55% (baseline 50%)
    - Range    > 30%  (baseline 20%) AND no class dominating >60%
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to checkpoint dir (contains best_pretrained.pt)")
    parser.add_argument("prepared_data_dir", help="Path to prepared data dir (contains *_features.parquet)")
    parser.add_argument("--max-batches", type=int, default=50, help="Batches to evaluate (default 50)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=96)
    args = parser.parse_args()

    ckpt_dir  = Path(args.checkpoint_dir)
    prep_dir  = Path(args.prepared_data_dir)

    try:
        import pandas as pd
        from futures_foundation import (
            FFMConfig, FFMForPretraining,
            get_model_feature_columns,
            FFMMultiInstrumentDataset,
            interleaved_train_val_split,
            REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run from the Futures-Foundation-Model root with the venv active.")
        sys.exit(1)

    TASK_NAMES = ["regime", "volatility", "structure", "range"]
    TASK_LABEL_MAPS = {
        "regime": REGIME_LABELS, "volatility": VOLATILITY_LABELS,
        "structure": STRUCTURE_LABELS, "range": RANGE_LABELS,
    }
    TASK_NUM_CLASSES = {"regime": 4, "volatility": 4, "structure": 2, "range": 5}
    BASELINES        = {"regime": 0.25, "volatility": 0.25, "structure": 0.50, "range": 0.20}
    LABEL_SENTINEL   = -100

    # ── Load model ──
    pt_path = ckpt_dir / "best_pretrained.pt"
    if not pt_path.exists():
        print(f"❌ best_pretrained.pt not found in {ckpt_dir}")
        print("   Only best_backbone.pt exists — cannot check task heads.")
        sys.exit(1)

    config = FFMConfig.from_pretrained(str(ckpt_dir))
    model  = FFMForPretraining(config)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    print(f"✅ Loaded best_pretrained.pt from {ckpt_dir}")

    # ── Load one instrument's val data ──
    feature_files = sorted(prep_dir.glob("*_features.parquet"))
    if not feature_files:
        print(f"❌ No feature parquets in {prep_dir}")
        sys.exit(1)

    import pandas as pd
    feat_path  = feature_files[0]
    instrument = feat_path.stem.replace("_features", "")
    label_path = prep_dir / f"{instrument}_labels.parquet"
    print(f"   Using {instrument} for evaluation ({feat_path.name})")

    features_df = pd.read_parquet(feat_path)
    labels_df   = pd.read_parquet(label_path)

    _, va_dsets = interleaved_train_val_split(
        features_df, labels_df, val_ratio=0.20,
        seq_len=args.seq_len, n_blocks=20, stride_train=4,
    )
    val_ds = FFMMultiInstrumentDataset(va_dsets)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Evaluate ──
    task_correct = {t: 0 for t in TASK_NAMES}
    task_total   = {t: 0 for t in TASK_NAMES}
    task_preds   = {t: [] for t in TASK_NAMES}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            outputs = model(
                features=batch["features"],
                candle_types=batch["candle_types"],
                time_of_day=batch["time_of_day"],
                day_of_week=batch["day_of_week"],
                instrument_ids=batch["instrument_ids"],
                session_ids=batch["session_ids"],
                regime_labels=batch["regime_label"],
                volatility_labels=batch["volatility_label"],
                structure_labels=batch["structure_label"],
                range_labels=batch["range_label"],
            )
            for task in TASK_NAMES:
                preds  = outputs[f"{task}_logits"].argmax(-1).reshape(-1)
                labels = batch[f"{task}_label"].reshape(-1)
                sentinel = LABEL_SENTINEL if task in ("regime", "structure") else None
                if sentinel is not None:
                    mask = labels != sentinel
                    task_correct[task] += (preds[mask] == labels[mask]).sum().item()
                    task_total[task]   += mask.sum().item()
                    task_preds[task].extend(preds[mask].numpy().tolist())
                else:
                    task_correct[task] += (preds == labels).sum().item()
                    task_total[task]   += labels.numel()
                    task_preds[task].extend(preds.numpy().tolist())

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  PRETRAIN TASK HEAD HEALTH — {ckpt_dir.name}")
    print(f"  Instrument: {instrument}  |  Batches: {min(args.max_batches, i+1)}")
    print(f"{'='*60}")

    all_ok = True
    for task in TASK_NAMES:
        acc      = task_correct[task] / max(1, task_total[task])
        baseline = BASELINES[task]
        counts   = Counter(task_preds[task])
        total    = sum(counts.values())
        max_cls  = max(counts.values()) / total if total > 0 else 0
        label_map = TASK_LABEL_MAPS[task]

        above_baseline = acc > baseline + 0.05
        not_collapsed  = max_cls < 0.70
        ok = above_baseline and not_collapsed

        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False

        print(f"\n  {status} {task.upper()} — acc={acc:.1%}  baseline={baseline:.0%}  max_class={max_cls:.0%}")
        for cls in sorted(counts.keys()):
            name = label_map.get(cls, f"class_{cls}")
            pct  = counts[cls] / total * 100 if total > 0 else 0
            flag = " ⚠️  DOMINANT" if counts[cls] / total > 0.60 else ""
            print(f"      {cls} ({name:>20s}): {pct:5.1f}%{flag}")

    print(f"\n{'='*60}")
    if all_ok:
        print(f"  ✅ All task heads healthy — backbone is safe to use")
    else:
        print(f"  ❌ One or more heads show collapse — backbone representations may be degraded")
        print(f"     Consider retraining with adjusted class weights or label strategy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
