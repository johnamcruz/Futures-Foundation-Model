# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A pretrained transformer backbone for futures market structure and regime classification.**

---

## Overview

Futures Foundation Model (FFM) is an open-source pretrained transformer designed to learn **market structure** and **regime dynamics** from raw OHLCV futures data. The backbone learns general representations of market behavior that can be fine-tuned for any downstream trading strategy.

### Philosophy

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure before being fine-tuned for sentiment or Q&A, FFM learns market structure before being fine-tuned for ORB entries, ICT setups, mean reversion signals, or any other strategy.

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
    └──→ Fine-tune: [Classification] [Regression] [Strategy+Risk]
```

### Model Variants

| Model | Purpose | Output |
|-------|---------|--------|
| `FFMForPretraining` | Multi-task self-supervised pretraining | 4 classification heads |
| `FFMForClassification` | Strategy signal prediction (e.g., BUY/SELL/HOLD) | N-class logits |
| `FFMForRegression` | Continuous value prediction (e.g., dynamic SL/TP) | N positive targets |
| `FFMForStrategyWithRisk` | Combined signal + risk management | Logits + SL/TP distances |

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

# Load pretrained backbone
config = FFMConfig()
backbone = FFMBackbone(config)
backbone.load_pretrained("path/to/checkpoint")

# Get market context embeddings
embeddings = backbone(features_tensor)  # (batch, 256)
```

### Fine-Tuning for a Strategy

```python
from futures_foundation import FFMForClassification

# ORB strategy: BUY / SELL / HOLD
model = FFMForClassification(config, num_labels=3)
model.load_backbone("path/to/pretrained/backbone")
model.freeze_backbone(freeze_ratio=0.66)  # Freeze bottom 2/3

# Train only the top layers + classification head
optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
```

### Combined Strategy + Risk Management

```python
from futures_foundation import FFMForStrategyWithRisk

# Signal head (BUY/SELL/HOLD) + Risk head (SL/TP in ATR units)
model = FFMForStrategyWithRisk(config, num_labels=3, num_risk_targets=2)
model.load_backbone("path/to/pretrained/backbone")
model.freeze_backbone(freeze_ratio=0.66)

outputs = model(features)
# outputs["signal_logits"]    → (batch, 3)  BUY/SELL/HOLD
# outputs["risk_predictions"] → (batch, 2)  [sl_atr, tp_atr]
```

### Causal Attention Mask (Per-Bar Predictions)

All model classes support a `causal=True` parameter that applies a strict lower-triangular mask so bar *i* cannot attend to any bar *j > i*. Use this when fine-tuning with `output_sequence=True` for per-bar predictions where lookahead must be eliminated:

```python
# Per-bar volatility prediction — no lookahead allowed
logits = model(features, output_sequence=True, causal=True)
# logits → (batch, seq_len, num_labels); each position only sees its own history

# Global summary inference — use full bidirectional attention (default)
embedding = backbone(features, causal=False)  # default; CLS aggregates all bars
```

> In causal mode the CLS token (position 0) is also restricted — it processes only itself, not the full sequence. For global sequence summary use `causal=False` (default).

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

Place your OHLCV CSV files in your data directory:

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

Strategy-agnostic price action descriptors computed from raw OHLCV. These capture candle structure, sequential momentum, and session context without embedding any specific setup logic:

| Feature | Description |
|---------|-------------|
| `candle_type` | Categorical candle class (0=doji, 1=bull strong, 2=bear strong, 3=bull pin, 4=bear pin, 5=neutral) — routed through a dedicated `nn.Embedding(6, 256)` rather than the continuous feature matrix to avoid implying a false ordinal relationship |
| `engulf_count` | Count of prior N bars (default 5) whose bodies are fully engulfed by the current bar |
| `momentum_speed_ratio` | Ratio of impulse speed to retrace speed over a rolling window; >1 = impulse leg dominant, <1 = retrace dominant |
| `wick_rejection` | Signed wick asymmetry: `(lower_wick − upper_wick) / range`, range [−1, 1]; positive = bullish rejection, negative = bearish rejection |
| `dir_consistency` | Fraction of the last N bars (default 5) whose close-open direction matches the current bar; range [0, 1] |
| `bar_size_vs_session` | Current bar range relative to the running session average range (resets at session open); >1 = larger than session average |

#### HTF Price Context Features (Group 9)

Where the current bar sits relative to the broader daily and weekly reference frames:

| Feature | Description |
|---------|-------------|
| `htf_dist_daily_high` | `(close − daily_high) / daily_high`, range ≤ 0 |
| `htf_dist_daily_low` | `(close − daily_low) / daily_low`, range ≥ 0 |
| `htf_dist_weekly_high` | `(close − weekly_high) / weekly_high`, range ≤ 0 |
| `htf_dist_weekly_low` | `(close − weekly_low) / weekly_low`, range ≥ 0 |
| `htf_range_position` | `(close − weekly_low) / (weekly_high − weekly_low)`, range [0, 1]; where in the week's range is price right now? |

#### Volume Absorption & Order Flow Features (Group 10)

Order-flow proxies that reveal whether volume is confirming or absorbing price movement:

| Feature | Description |
|---------|-------------|
| `vol_cum_signed_5` | Rolling 5-bar net buying/selling pressure: `Σ(vol_delta_proxy) / Σ(volume)`, range [−0.5, 0.5]; positive = buyers dominant |
| `vol_cum_signed_20` | Same as above over 20 bars |
| `vol_absorption` | `vol_ratio_5 × (1 − |body_pct|)`, range [0, 5]; high volume + small body = price being absorbed by opposing interest |
| `vol_momentum_align` | `sign(momentum_5) × (vol_ratio_5 − 1)`, range [−3, 3]; positive = elevated volume confirming the trend direction |

---

## Training

### Two-Stage Pipeline

**Stage 1: Pretrain** the backbone on all instruments with self-supervised tasks:

```bash
python scripts/pretrain.py \
    --data-dir data/raw/ \
    --output-dir checkpoints/pretrained/ \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --seq-len 64
```

**Stage 2: Fine-tune** for a specific strategy:

```bash
python scripts/finetune.py \
    --backbone checkpoints/pretrained/best_backbone.pt \
    --strategy orb \
    --data-dir data/orb_labeled/ \
    --output-dir checkpoints/orb/ \
    --freeze-ratio 0.66 \
    --epochs 30 \
    --lr 1e-4
```

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/          # Core library
│   ├── __init__.py
│   ├── config.py               # FFMConfig (HuggingFace compatible)
│   ├── model.py                # Backbone + Classification/Regression/Strategy heads
│   ├── features.py             # OHLCV → 66 derived features (10 groups incl. HTF context + volume absorption)
│   ├── candle_psychology.py    # Candle psychology feature derivation (6 features)
│   ├── labels.py               # Forward-looking label generation
│   └── dataset.py              # PyTorch Dataset + DataLoader
├── scripts/                    # Training & data prep scripts
│   ├── pretrain.py
│   └── finetune.py
├── tests/                      # Unit tests
│   ├── test_model.py
│   ├── test_features_crt.py    # CRT sweep feature tests (24 tests)
│   ├── test_features_core.py   # Core feature group tests (30 tests)
│   ├── test_labels.py          # Label generation tests (25 tests)
│   └── test_candle_psychology.py  # Candle psychology tests (33 tests)
├── .githooks/                  # Git hooks (activate with: git config core.hooksPath .githooks)
│   └── pre-commit              # Runs all unit tests before every commit
├── setup.py
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Contributing

We welcome contributions! Key areas:

- **New instruments**: Add support for crypto, forex, additional commodities
- **Additional pretraining tasks**: Order flow proxies, session pattern recognition
- **Fine-tuning recipes**: Share configs for specific strategies (ORB, ICT, mean reversion)
- **Feature engineering**: Novel OHLCV-derived features
- **Evaluation benchmarks**: Standardized regime classification benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

- [x] Core transformer backbone with HuggingFace compatibility
- [x] OHLCV feature derivation pipeline (66 ATR-normalized continuous features)
- [x] CRT sweep state features — 1H/4H prior-candle liquidity sweeps (10 features)
- [x] Candle psychology features — 5 continuous features + candle_type via dedicated model embedding
- [x] HTF price context features — daily/weekly OHLC distances + range position (5 features, Group 9)
- [x] Volume absorption & order flow features — cumulative signed delta, absorption ratio, vol-momentum alignment (4 features, Group 10)
- [x] Forward-looking self-supervised label generation (4 tasks)
- [x] Two-factor structure labels — 1H causal structure + forward expansion asymmetry must agree
- [x] Causal attention mask — `causal=True` on all forward() for per-bar lookahead-free predictions
- [x] Pretraining with overfitting detection + collapse monitoring
- [x] Fine-tuning framework: Classification, Regression, Strategy+Risk
- [x] Backbone freezing with differential layer groups
- [x] 5-instrument pretraining (ES, NQ, RTY, YM, GC)
- [x] Unit test suite (172 tests) with pre-commit hook enforcement
- [ ] Pretrained weights release on HuggingFace Hub
- [ ] Multi-timeframe input support
- [ ] Additional instruments (SI, CL, NKD)
- [ ] Evaluation suite and benchmarks
- [ ] ONNX export for production inference

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.