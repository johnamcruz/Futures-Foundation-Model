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
Input: OHLCV Bars (sequence of N bars × 42 derived features)
         │
    [Instrument Embedding + Session Embedding + Temporal Encoding]
         │
    [Transformer Encoder × 6 layers]
      • Multi-head self-attention (8 heads)
      • Feed-forward network (512-dim)
      • Pre-norm LayerNorm + residual connections
      • Dropout regularization
         │
    [CLS Token Pooling]
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
| **Market Structure** | Bullish, Bearish, Mixed | 20 bars | Upside vs downside expansion asymmetry |
| **Range Position** | 5 quintiles (0-20%, ..., 80-100%) | 10 bars | Where future close lands in current range |

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

### Feature Derivation (42 Features)

Features are instrument-agnostic via ATR normalization:

| Group | Count | Examples |
|-------|-------|---------|
| Bar Anatomy | 8 | Body/wick ratios, range in ATR |
| Returns & Momentum | 8 | Multi-horizon returns, acceleration |
| Volume Dynamics | 6 | Relative volume, delta proxy |
| Volatility Measures | 6 | ATR z-score, realized vol |
| Session Context | 5 | Distance from session OHLC + VWAP |
| Market Structure | 9 | Swing distances, range position |

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
│   ├── features.py             # OHLCV → 42 derived features
│   ├── labels.py               # Forward-looking label generation
│   └── dataset.py              # PyTorch Dataset + DataLoader
├── scripts/                    # Training & data prep scripts
│   ├── pretrain.py
│   └── finetune.py
├── tests/                      # Unit tests
│   └── test_model.py
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
- [x] OHLCV feature derivation pipeline (42 ATR-normalized features)
- [x] Forward-looking self-supervised label generation (4 tasks)
- [x] Pretraining with overfitting detection + collapse monitoring
- [x] Fine-tuning framework: Classification, Regression, Strategy+Risk
- [x] Backbone freezing with differential layer groups
- [x] 5-instrument pretraining (ES, NQ, RTY, YM, GC)
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