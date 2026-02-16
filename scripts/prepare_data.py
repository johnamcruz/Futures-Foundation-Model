"""
Step 1: Data Preparation — Run ONCE before training.

Reads raw OHLCV data (CSV or Parquet), derives 42 features, generates
4 pretraining labels, and saves everything as parquet for fast loading.

CSV format expected:
    datetime, open, high, low, close, volume

Usage:
    python scripts/prepare_data.py \
        --input data/raw/ \
        --output data/prepared/

Output:
    data/prepared/
    ├── ES_features.parquet
    ├── ES_labels.parquet
    ├── NQ_features.parquet
    ├── NQ_labels.parquet
    ├── GC_features.parquet
    ├── GC_labels.parquet
    └── prep_config.json       # Settings used (for reproducibility)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from futures_foundation import (
    derive_features,
    generate_all_labels,
    get_model_feature_columns,
    print_label_distribution,
    INSTRUMENT_MAP,
)


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: Derive features & generate labels")
    p.add_argument("--input", type=str, required=True, help="Directory with raw OHLCV files (CSV or Parquet)")
    p.add_argument("--output", type=str, default="data/prepared/", help="Output directory for prepared parquet files")
    p.add_argument("--bar-size", type=str, default="5min", help="Bar size label (metadata only)")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--structure-lookback", type=int, default=10)
    p.add_argument("--regime-atr-threshold", type=float, default=1.5, help="ATR z-score threshold for volatile regime")
    p.add_argument("--regime-momentum-threshold", type=float, default=0.3, help="Momentum threshold for trend vs rotational")
    p.add_argument("--volatility-lookback", type=int, default=50, help="Rolling window for volatility percentile")
    return p.parse_args()


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load a single OHLCV file (CSV or Parquet)."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})

    # Validate required columns
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    return df


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files
    data_files = sorted(
        list(input_dir.glob("*.csv")) + list(input_dir.glob("*.parquet"))
    )
    if not data_files:
        raise FileNotFoundError(f"No CSV or Parquet files found in {input_dir}")

    print(f"\n{'='*60}")
    print(f"  STEP 1: DATA PREPARATION")
    print(f"  Input:  {input_dir} ({len(data_files)} files)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    summary = {}
    total_start = time.time()

    for data_path in data_files:
        instrument = data_path.stem.split("_")[0].upper()

        if instrument not in INSTRUMENT_MAP:
            print(f"\n  ⚠ Skipping {data_path.name} — '{instrument}' not in INSTRUMENT_MAP")
            print(f"    Known instruments: {list(INSTRUMENT_MAP.keys())}")
            print(f"    Add it to INSTRUMENT_MAP in features.py if needed")
            continue

        print(f"\n{'─'*60}")
        print(f"  {instrument} — {data_path.name}")
        print(f"{'─'*60}")

        t0 = time.time()

        # Load raw data
        df = load_raw_data(data_path)
        print(f"  Loaded {len(df):,} bars")
        print(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

        # Derive features
        features_df = derive_features(
            df,
            instrument=instrument,
            atr_period=args.atr_period,
            structure_lookback=args.structure_lookback,
        )

        feature_cols = get_model_feature_columns()
        valid_count = features_df[feature_cols].notna().all(axis=1).sum()
        nan_count = len(features_df) - valid_count
        print(f"  Derived {len(feature_cols)} features")
        print(f"  Valid rows: {valid_count:,} / {len(features_df):,} ({valid_count/len(features_df)*100:.1f}%)")
        if nan_count > 0:
            print(f"  NaN rows (warmup period): {nan_count:,} — expected from rolling windows")

        # Generate labels
        labels_df = generate_all_labels(features_df)
        print_label_distribution(labels_df)

        # Save as parquet
        features_path = output_dir / f"{instrument}_features.parquet"
        labels_path = output_dir / f"{instrument}_labels.parquet"

        features_df.to_parquet(features_path, index=False)
        labels_df.to_parquet(labels_path, index=False)

        elapsed = time.time() - t0
        feat_size = features_path.stat().st_size / 1024 / 1024
        label_size = labels_path.stat().st_size / 1024 / 1024

        print(f"\n  ✓ Saved:")
        print(f"    {features_path.name} ({feat_size:.1f} MB)")
        print(f"    {labels_path.name} ({label_size:.1f} MB)")
        print(f"    Time: {elapsed:.1f}s")

        summary[instrument] = {
            "raw_bars": len(df),
            "valid_bars": int(valid_count),
            "date_start": str(df["datetime"].iloc[0]),
            "date_end": str(df["datetime"].iloc[-1]),
            "features_file": str(features_path.name),
            "labels_file": str(labels_path.name),
            "features_mb": round(feat_size, 1),
            "labels_mb": round(label_size, 1),
        }

    # Save preparation config (for reproducibility)
    prep_config = {
        "bar_size": args.bar_size,
        "atr_period": args.atr_period,
        "structure_lookback": args.structure_lookback,
        "regime_atr_threshold": args.regime_atr_threshold,
        "regime_momentum_threshold": args.regime_momentum_threshold,
        "volatility_lookback": args.volatility_lookback,
        "num_features": len(get_model_feature_columns()),
        "feature_columns": get_model_feature_columns(),
        "instruments": summary,
    }

    config_path = output_dir / "prep_config.json"
    with open(config_path, "w") as f:
        json.dump(prep_config, f, indent=2)

    total_elapsed = time.time() - total_start
    total_bars = sum(s["raw_bars"] for s in summary.values())
    total_valid = sum(s["valid_bars"] for s in summary.values())

    print(f"\n{'='*60}")
    print(f"  PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Instruments: {len(summary)}")
    print(f"  Total bars:  {total_bars:,} ({total_valid:,} valid)")
    print(f"  Output dir:  {output_dir}")
    print(f"  Config:      {config_path}")
    print(f"  Time:        {total_elapsed:.1f}s")
    print(f"\n  Next step:")
    print(f"    python scripts/pretrain.py --data-dir {output_dir} --from-prepared")
    print()


if __name__ == "__main__":
    main()