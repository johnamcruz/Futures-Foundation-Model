# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A model-agnostic classification foundation for futures markets — any pretrained time-series classification backbone learns market structure from raw OHLCV, then thin per-strategy heads finetune on top, all held to an honest-ruler walk-forward.**

**Contents:** [Quick Start](#quick-start) · [Philosophy](#philosophy--bert-for-futures) · [Overview](#overview) · [Self-Supervised Pretraining](#self-supervised-pretraining-the-bert-stage) · [The Classifier Seam](#the-classifier-seam--model-agnostic) · [Finetuning Pipeline](#finetuning-pipeline--walk-forward--produce) · [The Training Loop](#the-training-loop--overfit-driven) · [Add a Strategy](#add-a-strategy) · [Data](#data) · [Project Structure](#project-structure)

---

## Quick Start

```bash
pip install -e .
# + the package for your chosen classification backbone
```

FFM separates **learning the market** from **deciding a trade**. Stage 1 pretrains a backbone on raw OHLCV (self-supervised); stage 2 finetunes a thin per-strategy classifier on top, validated on the honest ruler.

```python
from futures_foundation.finetune import ssl, wf, produce

# 1) PRETRAIN (BERT stage) — self-supervised on raw OHLCV, runs on a GPU (Colab).
#    Output: an adapted backbone checkpoint that downstream classifiers start from.
ssl.loop_ssl(data_dir='…', out_path='ssl_ohlcv.pt')            # see colab/ runner

# 2) VALIDATE — walk-forward honest ruler with overfit→Optuna; classifier-agnostic.
verdict = wf.loop_streamed(make_labeler, streams,
                           clf_kwargs={'backbone_ckpt': 'ssl_ohlcv.pt'})

# 3) PRODUCE — only if it generalizes; trains on all data minus a holdout → ONNX bundle.
if verdict['generalizes']:
    produce.train_final_streamed(make_labeler, streams)
```

→ Labeler contract: [Add a Strategy](#add-a-strategy) · How validation catches & fixes overfitting: [The Training Loop](#the-training-loop--overfit-driven)

---

## Philosophy — BERT for futures

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure from unlabeled text before being finetuned for sentiment or Q&A, the FFM backbone learns **regime, structure, and volatility** from unlabeled futures OHLCV before any strategy logic runs. A strategy then adds only what the backbone cannot derive — setup geometry, entry distance, risk sizing — and finetunes a light classification head. Market-context knowledge is learned once and shared across every strategy.

Two principles shape everything below:

1. **The backbone is a pretrained foundation model designed for *classification*** — and ingests **multivariate raw OHLCV** (price + volume + range), so it can encode participation and volatility, the raw material of momentum compression → expansion.
2. **The backbone is swappable.** FFM commits to an *interface*, not a model. Any pretrained classification foundation model plugs in behind the same seam without touching a single strategy.

---

## Overview

The flow is two stages over one shared backbone:

```
raw OHLCV (9 tickers × 1/3/5/15min)
        │
        ▼  Stage 1 — SELF-SUPERVISED PRETRAIN  (finetune/ssl.py, GPU/Colab)
   temporal contrastive learning  ──►  adapted backbone checkpoint
        │                               (learns regime / structure / volatility)
        ▼  Stage 2 — FINETUNE  (finetune/wf.py → produce.py)
   strategy labeler + light classifier head  ──►  ONNX bundle the bot loads
```

- **Honest by construction.** Every result passes the honest ruler: walk-forward × {REAL, SHUFFLE, RANDOM} with an **overfit→Optuna** loop and a pre-registered PASS/FAIL auto-verdict. A number is believed only if REAL clearly beats every control, fold after fold.
- **2026 is a reserved out-of-sample year** — excluded from *both* pretraining and the rolling walk-forward, so the final OOS is never contaminated.
- **Causal by contract.** Every feature/window is strictly causal (streaming == batch, per bar); the leak audit is mandatory.

---

## Self-Supervised Pretraining (the BERT stage)

**`futures_foundation/finetune/ssl.py` — generic temporal contrastive learning that adapts the backbone to our markets, with the same generalization discipline as the strategy pipeline.**

Two augmented views of the *same* OHLCV window (slight time shifts, different window sizes, jitter, scale, magnitude-warp) are pulled together with an **NT-Xent** contrastive loss; all other windows in the (large) batch are pushed apart. To tell windows apart, the encoder must encode their temporal shape — so it learns volatility, regime, and compression→expansion structure. It runs GPU-maximized (data resident on GPU, vectorized GPU augmentations, large batch, AMP).

It is **overfit-gated and Optuna-tuned**, mirroring the strategy pipeline:

| Gate | What it checks |
|---|---|
| Time-split val early-stop | generalizes forward in time (2026 excluded) |
| REAL vs SHUFFLE vs RANDOM | REAL must reach a lower val loss than time-shuffled and noise windows → it learned *real* temporal structure |
| Collapse guard | embedding std / alignment / uniformity — the contrastive failure mode |
| Optuna | if it doesn't generalize, tune lr / temperature / regularization / augmentation strength until it does |
| **Final probe** | a linear probe shows the frozen embedding predicts **regime / volatility / structure** better than the un-adapted backbone — the "useful for downstream" check |

**Output:** an adapted backbone checkpoint (saved to Drive on Colab). Downstream finetuning initializes from it via `backbone_ckpt`. A Colab runner under [`colab/`](colab/) handles clone → install → Drive data path → run.

---

## The Classifier Seam — model-agnostic

`futures_foundation.finetune.classifier` is the swap point: a `Classifier` ABC + a `get_classifier(name, **cfg)` registry. A strategy pipeline references a classifier **by name**; the backbone behind it can change with no strategy edits.

```python
from futures_foundation.finetune.classifier import get_classifier

clf = get_classifier(BACKBONE, backbone_ckpt='ssl_ohlcv.pt', ft_mode='partial')
```

- **Pretrained classification backbone** — a foundation model + a per-strategy channel adapter + a light head, finetuned end-to-end and initialized from the SSL checkpoint via `backbone_ckpt`. It runs in an **isolated torch subprocess** (the parent stays torch-free) so torch never collides with other native libraries in one process. **Currently supported:** one such backbone (installed via its own package); **additional pretrained classification foundation models are planned behind the same interface.**
- **`logistic`** — a torch-free baseline / test vehicle for the whole pipeline.
- **Add your own backbone** by implementing `featurize()` + `fit_predict()` and registering it — the walk-forward, produce, and ONNX paths are all classifier-agnostic.

---

## Finetuning Pipeline — walk-forward → produce

**`futures_foundation/finetune/` — the strategy-pluggable harness: streamed walk-forward evaluator with honest-ruler controls, production trainer, ONNX export.** Every strategy goes through it; nothing about it is tied to a specific backbone.

**What it does:** a strategy labeler defines event candidates (e.g., a trend-pivot); for each event a multivariate context window → the classifier predicts `P(take)`, scored on **realized R** via the strategy's own evaluator. Validation runs the **overfit-driven training loop** on a rolling **train / validate / test** walk-forward with **REAL / SHUFFLE / RANDOM** controls and a pre-registered PASS/FAIL auto-verdict. The production trainer then fits one head on the full corpus minus the holdout and saves a single bundle + ONNX the bot loads.

| Component | Role |
|---|---|
| `wf.py` | Streamed walk-forward (`run_streamed`, `loop_streamed`) — featurize once across all streams (bounded RAM), rolling folds, VAL-selected operating point + **VAL→TEST generalization gate**, REAL/SHUFFLE/RANDOM, overfit→Optuna loop, PASS/FAIL verdict. 2026 excluded as OOS. |
| `produce.py` | Production training: one fit on the full corpus minus an N-month holdout; scores the 2026 OOS; emits the deployment bundle + signal contract + ONNX. |
| `tune.py` | Optuna search with a generalization-robust objective + held-out guard, auto-falling back to defaults unless the tuned config beats them. |
| `loop.py` | The overfit-driven loop: default WF → generalize check → Optuna only if it overfits → rerun → repeat → final full WF. |
| `_memmap.py` | Featurize-to-disk + streaming so full multi-timeframe, all-ticker runs fit in bounded RAM. |
| `classifier.py` / `classifiers/` | The model-agnostic seam (above) + backbone implementations. |

### The training loop — overfit-driven

`loop_streamed(...)` runs the whole process as one self-correcting loop. **Optuna fires only when overfitting is detected** — a config that already generalizes is left untouched:

1. Walk-forward with the **default** classifier config.
2. **Generalizes?** (VAL→TEST gap within tolerance, REAL beats controls fold-after-fold) → **keep defaults, done.**
3. **Overfit?** → **Optuna** for a config that generalizes (objective rewards cross-fold stability; auto-falls back to defaults unless the tuned config beats them on a held-out guard).
4. **Rerun**; repeat until it passes (capped — if nothing generalizes, the model is **flagged**).

Two guardrails keep it honest: the **VAL→TEST gate** (operating point chosen on *validation*, reported on *test*; an edge that decays is rejected) and tuning/selection that sees train+validation only — **test is never consulted**.

---

## Add a strategy

```python
class MyLabeler:
    n_classes = 2                               # binary selection (take / skip)
    def calendar(self): ...                     # ticker × timestamp
    def build(self, lo, hi, test_start):
        # → (contexts, labels, keys)  — keys carry realized-R per target
        ...
    def mv_contexts(self, keys):                # → [N, C, seq] multivariate windows
        ...
    def evaluate(self, keys, preds):            # → per-trade realized-R array
        ...
```

```python
from futures_foundation.finetune import wf, produce

verdict = wf.loop_streamed(make_labeler, streams,
                           clf_kwargs={'backbone_ckpt': 'ssl_ohlcv.pt'})
if verdict['generalizes']:
    produce.train_final_streamed(make_labeler, streams, export_onnx=True)
```

The labeler's `final run()` (in `finetune.base.StrategyLabeler`) applies a session-calibrated TP≥SL triple barrier (entry = next-bar open) and emits `signal_label` / `max_rr` / `sl_distance` / `direction`, centralizing the entry-after-signal / orientation bug class once for every strategy. `FoldHealthMonitor` flags per-fold pathologies (val/test gap, N-collapse, confidence-flat, zero-signal-fold); realized-R economics report PF / WR / mean-R / maxDD under a trailing exit (not optimistic MFE).

---

## Data

### Supported instruments

9 instruments: **ES, NQ, RTY, YM** (equity indices), **GC, SI** (metals), **CL** (energy), **ZB, ZN** (rates) — each at **1 / 3 / 5 / 15min**.

### Input format

```
data/
├── ES_3min.csv      # datetime, open, high, low, close, volume
├── ES_5min.csv
└── ...
```

`databento/build_continuous.py` resamples raw 1-min bars to any timeframe; `databento/append_update.py` splices new exports into `data/` continuously. A configurable `data_dir` (e.g. a Google-Drive mount on Colab) lets pretraining and finetuning read the same CSVs anywhere.

### Features

Raw OHLCV is the backbone's input. For strategies that fuse hand-crafted geometry, `futures_foundation.features.derive_features` produces instrument-agnostic (ATR-normalized), strictly causal features (bar anatomy, returns/momentum, volume dynamics, volatility, session context, market structure, HTF context) — every feature held to the no-look-ahead causal-parity rule (streaming == batch, per bar).

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/                # Foundation package (torch-free to import)
│   ├── finetune/                      # ★ The model-agnostic classification pipeline
│   │   ├── ssl.py / ssl_data.py       #   SSL temporal-contrastive pretraining (BERT stage)
│   │   ├── _ssl_torch.py              #   GPU-max contrastive trainer (subprocess/Colab)
│   │   ├── ssl_probe.py               #   linear probe: regime / vol / structure
│   │   ├── classifier.py              #   Classifier ABC + get_classifier registry (the seam)
│   │   ├── classifiers/               #   pluggable pretrained backbones + logistic baseline
│   │   ├── wf.py                      #   streamed walk-forward honest ruler + overfit→Optuna
│   │   ├── produce.py                 #   production trainer + 2026 OOS + ONNX + contract
│   │   ├── tune.py / loop.py          #   Optuna search + overfit-driven loop
│   │   ├── _memmap.py                 #   featurize-to-disk streaming (bounded RAM)
│   │   └── base.py / health.py        #   StrategyLabeler + FoldHealthMonitor
│   ├── extractors/                    #   Pluggable backbone interface (FeatureExtractor)
│   ├── features.py / primitives/      #   OHLCV → causal features; indicators/barriers
│   └── prepare.py                     #   raw CSVs → features parquet
├── colab/                             # ★ Colab runners: clone → install → Drive paths → run
├── databento/                         # Continuous-contract build + incremental update
├── docs/                              # Build specs + runbooks
├── tests/                             # Unit tests (pre-commit gated; torch-free by contract)
└── data/                              # Raw OHLCV CSVs (gitignored)
```

---

## Roadmap

- [x] Model-agnostic classifier seam (`finetune.classifier`) — backbone swappable behind one interface
- [x] Self-supervised temporal-contrastive pretraining on raw OHLCV (BERT stage), overfit-gated + Optuna + probe
- [ ] Pretrain the backbone on the full corpus (GPU/Colab) → downstream strategy walk-forward A/B vs the un-adapted backbone
- [ ] Additional pretrained classification foundation models behind the same seam
- [ ] Multivariate context beyond OHLCV (order-flow) as the next information rung

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
