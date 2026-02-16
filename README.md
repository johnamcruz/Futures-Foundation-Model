# 🏛️ Futures Foundation Model (FFM)

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
Input: OHLCV Bars (sequence of N bars × F derived features)
         │
    [Instrument Embedding + Session Embedding + Temporal Encoding]
         │
    [Transformer Encoder × 6 layers]
      • Multi-head self-attention (8 heads)
      • Feed-forward network
      • LayerNorm + residual connections
      • Dropout regularization
         │
    [Sequence Pooling] ← CLS token aggregation
         │
    BACKBONE OUTPUT: Market Context Embedding (256-dim)
         │
    ┌────┴────────┴──────────┴────────┴───┐
 [Regime]  [Volatility]  [Structure]  [Range]    ← Pretraining heads
    │
    └──→ Fine-tune: [ORB Head] [ICT Head] [Custom Head]
```

### Pretraining Objectives (Self-Supervised from OHLCV)

All labels are **derived automatically** from price data — no manual annotation required:

| Task | Classes | Description |
|------|---------|-------------|
| **Regime** | Trending Up, Trending Down, Rotational, Volatile Expansion | Market regime at sequence end |
| **Volatility State** | Low, Normal, Elevated, Extreme | ATR percentile vs rolling history |
| **Market Structure** | HH+HL (Bullish), LH+LL (Bearish), Mixed | Swing point structure |
| **Range Position** | 5 quintiles (0-20%, 20-40%, ..., 80-100%) | Price position in recent range |

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/futures-foundation-model.git
cd futures-foundation-model
pip install -e .
```

### Using the Pretrained Backbone

```python
from futures_foundation import FFMConfig, FFMForPretraining, FFMBackbone

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

---

## Data Preparation

### Supported Instruments
- **ES** (E-mini S&P 500)
- **NQ** (E-mini Nasdaq 100)
- **RTY** (E-mini Russell 2000)
- **YM** (E-mini Dow)
- Extensible to: GC (Gold), SI (Silver), CL (Crude Oil), and more

### Input Format

Place your OHLCV CSV files in `data/raw/`:

```
data/raw/
├── ES_5min.csv
├── NQ_5min.csv
├── RTY_5min.csv
└── YM_5min.csv
```

Each CSV should have columns: `datetime, open, high, low, close, volume`

### Feature Derivation & Label Generation

```bash
# Derive features, generate labels, and create sequences are handled
# automatically by the pretrain.py script. Just point it at your raw data:
python scripts/pretrain.py --data-dir data/raw/ --output-dir checkpoints/pretrained/
```

---

## Training

### Stage 1: Pretraining

```bash
python scripts/pretrain.py \
    --data-dir data/raw/ \
    --output-dir checkpoints/pretrained/ \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --seq-len 64
```

### Stage 2: Fine-Tuning (Example: ORB)

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
futures-foundation-model/
├── futures_foundation/          # Core library
│   ├── __init__.py
│   ├── config.py               # FFMConfig (HuggingFace compatible)
│   ├── model.py                # Transformer backbone + heads
│   ├── features.py             # OHLCV → derived features
│   ├── labels.py               # Auto-label generation
│   └── dataset.py              # PyTorch Dataset + DataLoader
├── scripts/                    # Training & data prep scripts
│   ├── pretrain.py
│   └── finetune.py
├── configs/                    # Model & training configs
│   └── default.yaml
├── tests/                      # Unit tests
│   └── test_model.py
├── examples/                   # Usage examples
│   └── finetune_orb.py
├── setup.py
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Contributing

We welcome contributions! Key areas:

- **New instruments**: Add support for crypto, forex, commodities
- **Additional pretraining tasks**: Order flow proxies, session pattern recognition
- **Fine-tuning recipes**: Share configs for specific strategies
- **Feature engineering**: Novel OHLCV-derived features
- **Evaluation benchmarks**: Standardized regime classification benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

- [x] Core transformer backbone with HuggingFace compatibility
- [x] OHLCV feature derivation pipeline (42 features)
- [x] Self-supervised label generation (4 tasks)
- [x] Pretraining script with multi-task uncertainty weighting
- [x] Fine-tuning framework with backbone freezing
- [ ] Pretrained weights release (ES, NQ, RTY, YM — 5 years)
- [ ] HuggingFace Hub integration (`from_pretrained`)
- [ ] Multi-timeframe input support
- [ ] Additional instruments (GC, SI, CL)
- [ ] Evaluation suite and benchmarks
- [ ] ONNX export for production inference

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
