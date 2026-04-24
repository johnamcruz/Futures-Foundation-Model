# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A pretrained transformer backbone for futures market structure and regime classification — with a plug-and-play fine-tuning framework for any trading strategy.**

---

## Overview

Futures Foundation Model (FFM) is an open-source pretrained transformer designed to learn **market structure** and **regime dynamics** from raw OHLCV futures data. The backbone learns general representations of market behavior that can be fine-tuned for any downstream trading strategy.

### Philosophy

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure before being fine-tuned for sentiment or Q&A, FFM learns market structure before being fine-tuned for ORB entries, ICT setups, mean reversion signals, or any other strategy. The backbone handles all market context — trend, volatility, session, HTF structure, order flow. Your strategy adds only the setup-specific features it uniquely knows.

---

## Fine-Tuning Framework

**v0.3 introduced `futures_foundation.finetune` — a reusable, fully-tested walk-forward training framework.**

Adding a new strategy now requires implementing one class. Everything else — training loop, walk-forward splits, warm start between folds, dual checkpointing, evaluation tables, ONNX export — is handled by the framework.

### Add a new strategy in ~30 lines

```python
from futures_foundation.finetune import StrategyLabeler, TrainingConfig, run_labeling, run_walk_forward

class MyStrategyLabeler(StrategyLabeler):
    @property
    def name(self): return 'my_strategy'

    @property
    def feature_cols(self): return ['zone_height', 'entry_depth', 'risk_norm']

    def run(self, df_raw, ffm_df, ticker):
        # df_raw:  raw 5-min OHLCV (tz-aware NY index)
        # ffm_df:  FFM-prepared features — use htf_1h_structure, vty_atr_raw, etc.
        #          directly rather than recomputing from raw bars
        features_df, labels_df = my_signal_logic(df_raw, ffm_df)
        return features_df, labels_df  # both aligned to ffm_df.index
```

```python
# Cell 3 — Label all tickers (cached to parquet on first run)
labeler = MyStrategyLabeler()
run_labeling(labeler, TICKERS, RAW_DATA_DIR, PREPARED_DIR, CACHE_DIR)

# Cell 4 — Walk-forward fine-tuning (4 folds, warm start, dual checkpoint)
fold_results = run_walk_forward(
    folds=FOLDS, tickers=TICKERS, ffm_dir=PREPARED_DIR,
    strategy_dir=CACHE_DIR, output_dir=OUTPUT_DIR,
    backbone_path=BACKBONE_PATH, ffm_config=ffm_config,
    training_cfg=TrainingConfig(), num_strategy_features=3,
    strategy_feature_cols=labeler.feature_cols,
)

# Cell 5 — Evaluation (confidence thresholds, per-fold, vs baseline)
from futures_foundation.finetune import print_eval_summary
print_eval_summary(fold_results, baseline_wr=BASELINE_WR)
```

### What the framework provides

| Component | Description |
|---|---|
| `StrategyLabeler` | ABC — implement `name`, `feature_cols`, `run()` to define any strategy |
| `TrainingConfig` | Dataclass holding all training hyperparameters |
| `HybridStrategyModel` | FFM backbone + strategy feature projection + signal/risk/confidence heads |
| `HybridStrategyDataset` | Sliding-window dataset parameterised by your strategy feature columns |
| `run_labeling()` | CSV I/O, timezone normalization, parquet caching per ticker |
| `run_walk_forward()` | 4-fold walk-forward, warm start F1→F2→F3→F4, dual checkpoint (val_loss + signal_F1) |
| `print_eval_summary()` | Confidence threshold table, per-fold breakdown, vs-baseline comparison |
| `export_onnx()` | Production ONNX export of the final fold model |

### Model architecture

```
FFM Backbone (frozen lower layers)
     │  → CLS embedding (256-dim)
     │
┌────┴─────────────────────────────────────┐
│                                           │
│   Strategy features (N strategy-specific) │
│        → Linear(64) → GELU → Linear(64)  │
│                           │               │
└────── cat ────────────────┘               │
          │ (256 + 64)                       │
      fusion: Linear → GELU → LayerNorm     │
          │ (256)                            │
   ┌──────┼──────────┬───────────┐
   │      │          │           │
signal   risk   confidence
 head    head     head
```

The backbone handles **all market context** — HTF trend, volatility regime, session structure, CRT sweeps, order flow. Strategy features cover only what the backbone cannot derive: setup geometry, zone age, entry distance, risk sizing.

### CISD+OTE: first concrete implementation

`colabs/cisd_ote.py` implements the Change in State of Delivery + Optimal Trade Entry strategy using the framework. It reduces to a `CISDOTELabeler(StrategyLabeler)` subclass with 10 strategy-specific features (zone geometry and trade mechanics). The 400-line training cell is replaced by a single `run_walk_forward()` call.

Baseline performance (v5.1 reference, 5 instruments):
- 2,729 signals across ES, NQ, RTY, YM, GC
- 68.2% precision @ 0.90 confidence threshold
- Profit factor 8.71 · +35.7pp above mechanical baseline

---

## Architecture

```
Input: OHLCV Bars (sequence of N bars × 66 continuous features + candle_type embedding)
         │
    [Instrument Embedding + Session Embedding + Temporal Encoding]
         │
    [Transformer Encoder × 6 layers]
      • Multi-head self-attention (8 heads, optional causal mask)
      • Feed-forward network (512-dim)
      • Pre-norm LayerNorm + residual connections
      • Dropout regularization
         │
    [CLS Token Pooling]  or  [Per-Bar Hidden States (output_sequence=True)]
         │
    BACKBONE OUTPUT: Market Context Embedding (256-dim)
         │
    ┌────┴────────┴──────────┴────────┴───┐
 [Regime]  [Volatility]  [Structure]  [Range]    ← Pretraining heads
    │
    └──→ Fine-tune: [Classification] [Regression] [Strategy+Risk] [HybridStrategy]
```

### Pretraining Objectives (Forward-Looking, Self-Supervised)

All labels are **forward-looking** — the model must predict what happens in the **next N bars**, not read the current state. Labels are derived automatically from price data with no manual annotation:

| Task | Classes | Horizon | Description |
|------|---------|---------|-------------|
| **Regime** | Trending Up, Trending Down, Rotational, Volatile | 20 bars | Future return direction + volatility expansion |
| **Volatility State** | Low, Normal, Elevated, Extreme | 10 bars | Forward realized vol ranked vs recent history |
| **Market Structure** | Bullish, Bearish | 20 bars | Two-factor confirmation: 1H causal structure AND forward expansion asymmetry must agree; conflicting signals → sentinel (skipped in loss) |
| **Range Position** | 5 quintiles (0-20%, ..., 80-100%) | 10 bars | Where future close lands in current range |

> **Structure labels use two-factor confirmation.** A bar is labeled bullish only when the 1H higher-timeframe structure is already bullish (majority of last 3 completed 1H closes moving higher) AND forward price shows upside > 1.5× downside over the next 20 bars. Both factors must agree — ambiguous bars are skipped via `ignore_index=-100`. This prevents the model from confusing counter-trend bounces with genuine structural trends.

---

## Quick Start

### Installation

```bash
git clone https://github.com/johnamcruz/Futures-Foundation-Model.git
cd Futures-Foundation-Model
pip install -e .
```

### Using the Pretrained Backbone

```python
from futures_foundation import FFMConfig, FFMBackbone

config   = FFMConfig()
backbone = FFMBackbone(config)
backbone.load_pretrained("path/to/checkpoint")

embeddings = backbone(features_tensor)  # (batch, 256)
```

### Fine-Tuning with the Framework

```python
from futures_foundation.finetune import (
    StrategyLabeler, TrainingConfig,
    run_labeling, run_walk_forward, print_eval_summary,
)
```

See the [Fine-Tuning Framework](#fine-tuning-framework) section above and `colabs/cisd_ote.py` for a complete working example.

### Causal Attention Mask (Per-Bar Predictions)

All model classes support a `causal=True` parameter that applies a strict lower-triangular mask so bar *i* cannot attend to any bar *j > i*. Use this when fine-tuning with `output_sequence=True` for per-bar predictions where lookahead must be eliminated:

```python
# Per-bar volatility prediction — no lookahead allowed
logits = model(features, output_sequence=True, causal=True)

# Global summary inference — use full bidirectional attention (default)
embedding = backbone(features, causal=False)
```

---

## Data Preparation

### Supported Instruments

Currently pretrained on 5 instruments (~1.7M bars total):

| Instrument | Symbol | Description |
|-----------|--------|-------------|
| **ES** | E-mini S&P 500 | US large cap index |
| **NQ** | E-mini Nasdaq 100 | US tech index |
| **RTY** | E-mini Russell 2000 | US small cap index |
| **YM** | E-mini Dow | US blue chip index |
| **GC** | Gold Futures | Precious metals |

Extensible to: SI (Silver), CL (Crude Oil), NKD (Nikkei), and more.

### Input Format

```
data/raw/
├── ES_5min.csv
├── NQ_5min.csv
├── RTY_5min.csv
├── YM_5min.csv
└── GC_5min.csv
```

Each CSV should have columns: `datetime, open, high, low, close, volume`

### Feature Derivation (67 Inputs: 66 Continuous + 1 Embedding)

Features are instrument-agnostic via ATR normalization:

| Group | Count | Examples |
|-------|-------|---------|
| 1 — Bar Anatomy | 8 | Body/wick ratios, range in ATR |
| 2 — Returns & Momentum | 8 | Multi-horizon returns, acceleration |
| 3 — Volume Dynamics | 6 | Relative volume, delta proxy |
| 4 — Volatility Measures | 6 | ATR z-score, realized vol |
| 5 — Session Context | 5 | Distance from session OHLC + VWAP |
| 6 — Market Structure | 9 | Swing distances, range position |
| 7 — CRT Sweep State | 10 | 1H/4H prior-candle liquidity sweep events |
| 8 — Candle Psychology | 5 + 1 emb | engulf count, momentum speed, wick rejection, dir consistency, bar size vs session; candle_type → dedicated model embedding |
| 9 — HTF Price Context | 5 | Daily + weekly OHLC distances, higher-timeframe range position |
| 10 — Volume Absorption & Order Flow | 4 | Cumulative signed delta, absorption ratio, volume-momentum alignment |

#### CRT Sweep State Features

Candle Range Theory (CRT) sweeps occur when a bar wicks beyond the prior candle's high or low and closes back inside it — a liquidity sweep that often precedes directional expansion. These features capture sweep activity on the 1-hour and 4-hour timeframes and align it to each base bar:

| Feature | Description |
|---------|-------------|
| `swp_1h_bull_active` | 1H bull sweep active (wicked below prior low, closed above it) |
| `swp_1h_bear_active` | 1H bear sweep active (wicked above prior high, closed below it) |
| `swp_1h_age_norm` | Normalized age of the most recent 1H sweep (0 = fresh, 1 = expired) |
| `swp_1h_magnitude` | ATR-normalized wick penetration depth of the 1H sweep, clipped to [0, 3] |
| `swp_4h_bull_active` | 4H bull sweep active |
| `swp_4h_bear_active` | 4H bear sweep active |
| `swp_4h_age_norm` | Normalized age of the most recent 4H sweep |
| `swp_4h_magnitude` | ATR-normalized wick penetration depth of the 4H sweep, clipped to [0, 3] |
| `swp_tf_alignment` | Timeframe alignment: +1 (both bullish), -1 (both bearish), 0 (mixed) |
| `swp_dominant_dir` | Dominant sweep direction across timeframes (same as `swp_tf_alignment`) |

Sweep state is forward-filled for a frequency-agnostic expiry window (1 hour = `round(60 / bar_minutes)` bars) so the features work correctly on 3-min, 5-min, or any other base timeframe.

#### Candle Psychology Features

Strategy-agnostic price action descriptors computed from raw OHLCV:

| Feature | Description |
|---------|-------------|
| `candle_type` | Categorical candle class (0=doji, 1=bull strong, 2=bear strong, 3=bull pin, 4=bear pin, 5=neutral) — routed through a dedicated `nn.Embedding(6, 256)` |
| `engulf_count` | Count of prior N bars whose bodies are fully engulfed by the current bar |
| `momentum_speed_ratio` | Ratio of impulse speed to retrace speed; >1 = impulse dominant |
| `wick_rejection` | Signed wick asymmetry: `(lower_wick − upper_wick) / range`, range [−1, 1] |
| `dir_consistency` | Fraction of last N bars whose direction matches the current bar |
| `bar_size_vs_session` | Current bar range relative to running session average |

#### HTF Price Context Features (Group 9)

| Feature | Description |
|---------|-------------|
| `htf_dist_daily_high` | `(close − daily_high) / daily_high` |
| `htf_dist_daily_low` | `(close − daily_low) / daily_low` |
| `htf_dist_weekly_high` | `(close − weekly_high) / weekly_high` |
| `htf_dist_weekly_low` | `(close − weekly_low) / weekly_low` |
| `htf_range_position` | `(close − weekly_low) / (weekly_high − weekly_low)` |

#### Volume Absorption & Order Flow Features (Group 10)

| Feature | Description |
|---------|-------------|
| `vol_cum_signed_5` | Rolling 5-bar net buying/selling pressure |
| `vol_cum_signed_20` | Same over 20 bars |
| `vol_absorption` | High volume + small body = price being absorbed |
| `vol_momentum_align` | Elevated volume confirming or diverging from trend direction |

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/          # Core library
│   ├── __init__.py
│   ├── config.py               # FFMConfig (HuggingFace compatible)
│   ├── model.py                # Backbone + Classification/Regression/Strategy heads
│   ├── features.py             # OHLCV → 66 derived features (10 groups)
│   ├── candle_psychology.py    # Candle psychology features
│   ├── labels.py               # Forward-looking label generation
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   └── finetune/               # ★ Strategy fine-tuning framework
│       ├── __init__.py
│       ├── base.py             # StrategyLabeler ABC
│       ├── config.py           # TrainingConfig dataclass
│       ├── model.py            # HybridStrategyModel
│       ├── dataset.py          # HybridStrategyDataset
│       ├── losses.py           # FocalLoss
│       └── trainer.py          # run_labeling, run_walk_forward, print_eval_summary
├── colabs/
│   ├── ffm_pretrain_5min.py    # Colab pretraining script
│   └── cisd_ote.py             # CISD+OTE strategy (example fine-tune implementation)
├── tests/                      # Unit tests (217 total)
│   ├── test_model.py           # Backbone + heads (32 tests)
│   ├── test_finetune.py        # Fine-tuning framework (42 tests, incl. FFM field coverage)
│   ├── test_features_crt.py    # CRT sweep features (24 tests)
│   ├── test_features_core.py   # Core feature groups (30 tests)
│   ├── test_labels.py          # Label generation (25 tests)
│   └── test_candle_psychology.py  # Candle psychology (33 tests)
├── .githooks/
│   └── pre-commit              # Runs all unit tests before every commit
├── setup.py
├── requirements.txt
└── README.md
```

---

## Releases

| Version | Description |
|---------|-------------|
| **v0.3** | `futures_foundation.finetune` framework — plug-and-play walk-forward fine-tuning; CISD+OTE migrated as first concrete strategy |
| **v0.2** | FFM backbone + CISD+OTE fine-tuning pipeline (v7); 58 backbone features |
| **v0.1** | Last stable backbone checkpoint reference |

---

## Contributing

We welcome contributions! Key areas:

- **New strategy implementations**: Add a `StrategyLabeler` subclass for ORB, ICT breaker blocks, mean reversion, etc.
- **New instruments**: Add support for crypto, forex, additional commodities
- **Additional pretraining tasks**: Order flow proxies, session pattern recognition
- **Feature engineering**: Novel OHLCV-derived features

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

- [x] Core transformer backbone with HuggingFace compatibility
- [x] OHLCV feature derivation pipeline (66 ATR-normalized continuous features)
- [x] CRT sweep state features — 1H/4H prior-candle liquidity sweeps (10 features)
- [x] Candle psychology features — 5 continuous + candle_type embedding
- [x] HTF price context features — daily/weekly OHLC distances + range position
- [x] Volume absorption & order flow features
- [x] Forward-looking self-supervised label generation (4 tasks)
- [x] Two-factor structure labels with confidence sentinel
- [x] Causal attention mask for per-bar predictions
- [x] 5-instrument pretraining (ES, NQ, RTY, YM, GC)
- [x] **`futures_foundation.finetune` — reusable walk-forward fine-tuning framework**
- [x] **`StrategyLabeler` ABC — implement one class, get everything else for free**
- [x] **CISD+OTE strategy as first concrete fine-tune implementation**
- [x] Unit test suite (217 tests) with per-column FFM field coverage checks
- [ ] Pretrained weights release on HuggingFace Hub
- [ ] Additional strategy implementations (ORB, ICT breaker blocks)
- [ ] Multi-timeframe input support
- [ ] Additional instruments (SI, CL, NKD)
- [ ] ONNX export for production inference

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
