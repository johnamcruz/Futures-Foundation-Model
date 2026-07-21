# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A leakage-aware foundation-model pipeline for futures bars: adapt a pretrained time-series encoder to raw OHLCV, validate it through causal controls and walk-forward tests, then export lightweight downstream heads for deployment.**

**Contents:** [Quick Start](#quick-start) · [Design](#design) · [Overview](#overview) · [Self-Supervised Pretraining](#self-supervised-pretraining) · [Checkpoint Contract](#checkpoint-contract) · [Classifier Seam](#classifier-seam) · [Walk-forward and Production](#walk-forward-and-production) · [Data](#data) · [Project Structure](#project-structure)

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install mantis-tsfm
```

FFM separates **learning reusable market representations** from **training a downstream decision head**. The production pretraining lineage is:

```text
Mask → Temporal Contrastive → Multi-horizon Seq2seq → NextLeg
```

Before a long run, validate the complete 9×4 data contract and immutable holdout:

```bash
./.venv/bin/python scripts/mantis_ssl_clean_pipeline.py --preflight-only --device mps
```

The core API can also run any registered objective independently:

```python
from futures_foundation.finetune import ssl

mask = ssl.loop_ssl(
    data_dir="data",
    out_path="checkpoints/mantis_ssl_ohlcv.pt",
    pretext="mask",
    holdout_start="2026-01-01",
    controls=("shuffle", "random"),
    lora_r=8,
    lora_alpha=16,
)
```

Each later stage receives the previous checkpoint through `backbone_ckpt` and writes to a different output path. See [Self-Supervised Pretraining](#self-supervised-pretraining) and [Checkpoint Contract](#checkpoint-contract).

## Design

> Separate **"understanding market context"** from **"making downstream decisions."**

Like masked-language pretraining, FFM first learns structure from unlabeled sequences. The backbone learns **regime, temporal geometry, volatility, and forward dynamics** from futures OHLCV before a downstream task is introduced. A lightweight classifier or forecasting head can then reuse the same encoder.

Four principles shape everything below:

1. **Raw bars are the source of truth.** The foundation path consumes multivariate OHLCV rather than future-derived trading labels.
2. **Backbones and heads are separated.** Downstream code depends on a small classifier interface, not on one model implementation.
3. **Time is a hard boundary.** Training, validation, and held-out periods are separated before windows and targets are constructed.
4. **Artifacts are traceable.** Every stage records its data provenance, configuration, parent-checkpoint hash, and validation report.

---

## Overview

The flow is a self-supervised pretraining pipeline over one shared backbone, followed by a lightweight downstream head:

```
continuous-contract OHLCV (9 tickers × 4 timeframes)
        │
        ▼  SELF-SUPERVISED PRETRAINING
   1) masked reconstruction       → local candle and volatility structure
   2) temporal contrastive        → smooth multi-scale market-state geometry
   3) multi-horizon seq2seq       → forward OHLCV dynamics
   4) next-leg forecasting        → direction-agnostic leg development in bars
        │   warm-started checkpoints; LoRA or full adaptation; crash-safe best saves
        ▼  DOWNSTREAM VALIDATION  (finetune/wf.py)
   task labeler + light classifier head
        ▼  PRODUCTION  (finetune/produce.py)
   calibrated bundle + signal contract + ONNX artifacts
```

- **Honest by construction.** Downstream claims pass rolling walk-forward evaluation with REAL/SHUFFLE/RANDOM controls and validation-selected operating points.
- **2026 is reserved out of sample.** The clean SSL runner physically excludes timestamps on or after `2026-01-01` from training inputs and targets.
- **Causal by contract.** Every feature/window is strictly causal (streaming == batch, per bar); the leak audit is mandatory.
- **Roll-safe bars.** Continuous futures streams are assembled by session, exclude spreads, and can be back-adjusted at contract switches.
- **Bar data only.** Tick and order-book inputs are not currently supported; aggregate source data into closed OHLCV bars first.

---

## Self-Supervised Pretraining

`futures_foundation/finetune/ssl.py` provides the task-independent SSL loop. Registered objectives live under `futures_foundation/finetune/pretext/`; each task owns its window reserve, trainer, diagnostics, and verdict additions. `scripts/mantis_ssl_clean_pipeline.py` composes the production lineage while keeping every checkpoint distinct.

### 1. Masked reconstruction

A fraction of each standardized OHLCV window is corrupted. The network reconstructs only the masked positions, forcing the encoder to use surrounding temporal context rather than memorize the visible value.

### 2. Temporal contrastive learning

Augmented views and temporally nearby windows form positives; sufficiently distant windows form negatives. This stage shapes a smooth, multi-scale market-state embedding without using future trading outcomes as labels.

### 3. Multi-horizon seq2seq forecasting

Variable-length context predicts future OHLCV moves at several horizons. Targets are expressed relative to the final context bar, making copy-the-last-value a zero forecast rather than an accidental shortcut.

### 4. Next-leg forecasting

The final pretraining stage predicts the duration of the developing leg and its counter-leg in bars while retaining the candle-forecasting objective as an anchor. Bar counts keep the target comparable across instruments and timeframes.

Shared discipline across every stage:

| Guardrail | What it does |
|---|---|
| **Warm-start chain** | every child loads the exact parent encoder; missing parents fail closed |
| **LoRA or full adaptation** | rank/alpha/dropout are configurable; merged LoRA checkpoints are ordinary encoder state dictionaries |
| **Anti-forgetting** | later stages can freeze the tokenizer and early encoder layers while refining higher layers |
| **Crash-safe save + resume** | the best checkpoint is written progressively (atomic, every val improvement) with a resume path — a disconnected GPU run never loses progress |
| **Time-split validation** | all windows and objective-specific future reserves stay inside their split; `>=2026-01-01` is excluded by the clean runner |
| **Input controls** | REAL, time-SHUFFLE, and RANDOM inputs can be trained against unchanged targets to expose temporal shortcuts |
| **Bounded memory** | training data stays resident per process; downstream embeddings and large diagnostics support disk-backed/chunked execution |
| **MPS/CUDA support** | device-specific batches, fixed sample budgets, and automatic MPS OOM fallback preserve comparable training exposure |
| **Configurable source mixture** | `bar_proportional` remains the default; opt-in `uniform_stream` chooses a ticker/timeframe stream first, then a legal window within it |

To test equal training exposure across the 9x4 corpus without changing the
chronological validation distribution, use a separate output directory:

```bash
./.venv/bin/python scripts/mantis_ssl_clean_pipeline.py \
  --sampling-mode uniform_stream \
  --out-dir temp/clean_ssl_pre2026_lora_uniform
```

The runner refuses to reuse a completed checkpoint produced under a different
sampling mode.

### LoRA

The clean runner defaults to LoRA rank 8, alpha 16. It freezes the pretrained encoder weights and learns low-rank changes in the attention projections. At save time those changes are merged into a plain encoder state dictionary, so downstream consumers do not need LoRA-specific loading code. `--lora-r 0` restores full encoder fine-tuning.

## Checkpoint Contract

The production lineage writes four independent encoder artifacts:

| Stage | Default filename |
|---|---|
| Mask | `mantis_ssl_ohlcv.pt` |
| Temporal contrastive | `mantis_ssl_regime_from_mask.pt` |
| Seq2seq | `mantis_ssl_ctr_seq2seq.pt` |
| NextLeg | `mantis_ssl_nextleg.pt` |

Each `.pt` contains the best merged **encoder** state—not its temporary training decoder or projection head. That makes every stage independently reusable as `backbone_ckpt` for downstream tasks or new pretraining branches. Each checkpoint is accompanied by:

- `<checkpoint>.report.json`: configuration, validation history, and task diagnostics;
- `<checkpoint>.data_provenance.json`: source hashes, holdout boundary, and parent-checkpoint hash;
- `pipeline_manifest.json`: final ordered lineage and SHA-256 for every stage.

Do not overwrite a parent checkpoint with its child. Promote validated artifacts from temporary run storage into a versioned checkpoint directory as one bundle.

---

## Classifier Seam

`futures_foundation.finetune.classifier` is the swap point: a `Classifier` ABC + a `get_classifier(name, **cfg)` registry. Downstream code references a classifier **by name**, so the backbone can change without changing the walk-forward harness.

```python
from futures_foundation.finetune.classifier import get_classifier

clf = get_classifier(
    "mantis",
    backbone_ckpt="checkpoints/mantis_ssl_nextleg.pt",
    ft_mode="partial",
)
```

Two ways to attach the backbone — both initialize from the SSL checkpoint via `backbone_ckpt`, both run torch in an **isolated subprocess** (the parent stays torch-free, so torch never collides with other native libraries in one process):

- **End-to-end fine-tune** — foundation model + downstream channel adapter + light head, all trained together. Maximum capacity; the backbone specializes to the task.
- **Frozen head-only** — embed each window **once** through the frozen encoder, then train a cheap **logistic or MLP head** per fold on the cached embedding (optionally concatenated with hand-crafted geometry features). This is the "embed once → head per fold" pattern: fast enough to iterate on local hardware, and a clean linear/​shallow probe of what the representation actually carries.
  - **Cross-run embedding cache** — the frozen embedding is deterministic in `(backbone_ckpt, bars, window spec)`, so it's cached to disk keyed on exactly those. The expensive embed cost is **paid once per backbone**: reruns, head swaps (logistic↔MLP), and interpretability checks reuse the cached vectors instead of re-embedding. `EMBED_CACHE=0` disables; `EMBED_CACHE_DIR` relocates it.

**Currently available:** Mantis end-to-end and frozen-embedding classifiers plus a torch-free logistic baseline. A frozen MOMENT adapter is registered as an explicit stub and raises `NotImplementedError` until its encoder integration is implemented. Heavy backbones are imported lazily so the parent orchestration process remains torch-free.

- **`logistic`** — a torch-free baseline / test vehicle for the whole pipeline.
- **Add your own backbone** by implementing `featurize()` + `fit_predict()` and registering it — the walk-forward, produce, and ONNX paths are all classifier-agnostic.

---

## Walk-forward and Production

**`futures_foundation/finetune/` is a task-pluggable harness:** streamed walk-forward evaluation with controls, production training, and ONNX export. Nothing in the harness is tied to a specific backbone or downstream task.

**What it does:** a task labeler supplies causal event candidates, multivariate context windows, targets, and an evaluator. Validation runs the **overfit-driven training loop** on rolling **train / validate / test** folds with **REAL / SHUFFLE / RANDOM** controls and a PASS/FAIL verdict. The production trainer fits one final head on the corpus before the holdout and can export its artifacts to ONNX.

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

### Downstream task contract

```python
class MyTask:
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

verdict = wf.loop_streamed(
    make_task,
    streams,
    classifier="mantis_frozen",
    clf_kwargs={"backbone_ckpt": "checkpoints/mantis_ssl_nextleg.pt"},
)
if verdict['generalizes']:
    produce.train_final_streamed(
        make_task,
        streams,
        classifier="mantis_frozen",
        clf_kwargs={"backbone_ckpt": "checkpoints/mantis_ssl_nextleg.pt"},
        export_onnx=True,
    )
```

The public `StrategyLabeler` base class centralizes causal next-bar labeling and the shared output schema. `FoldHealthMonitor` flags per-fold pathologies such as validation/test gaps, sample collapse, flat confidence, and empty-signal folds. Downstream task definitions and deployment policy remain outside the foundation-model core.

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

**Fixed-interval OHLCV bars — not tick/quote data.** Every CSV is one row per closed bar at a
chosen timeframe; there is no tick-level or order-book input path in the pipeline today
(tick and order-book support is on the roadmap, not yet implemented).
If your source data is tick-by-tick, aggregate it into bars first — `databento/build_continuous.py`
resamples raw 1-min bars to any coarser timeframe (it does not build bars from ticks); a
tick→1-min aggregation step is on you before that. `databento/append_update.py` splices new
exports into `data/` continuously. A configurable `data_dir` (e.g. a Google-Drive mount on
Colab) lets pretraining and finetuning read the same CSVs anywhere.

### Features

Raw OHLCV is the backbone's input—the foundation learns market context directly from price and volume, with no derived features fed into SSL. Public causal primitives for pivots, barriers, indicators, and sessions are available to downstream labelers and are held to the same no-look-ahead parity rule (streaming output must equal batch output at every bar).

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/                # Foundation package (torch-free to import)
│   ├── finetune/                      # ★ The model-agnostic classification pipeline
│   │   ├── ssl.py / ssl_data.py       #   SSL orchestrator + leakage-safe data assembly
│   │   ├── pretext/                   #   mask / contrastive / forecast / NextLeg tasks
│   │   │   ├── base.py                #     PretextTask interface (reserve / train / gate)
│   │   │   └── _torch/                #     per-stage GPU trainers + shared BaseTrainer (save/resume/freeze)
│   │   ├── _ssl_torch.py              #   back-compat shim → re-exports pretext/_torch (frozen embed, ONNX)
│   │   ├── ssl_probe.py               #   linear probe: regime / vol / structure (soft signal)
│   │   ├── classifier.py              #   Classifier ABC + get_classifier registry (the seam)
│   │   ├── classifiers/               #   end-to-end FT + frozen head-only (cached embeddings) + logistic
│   │   ├── wf.py                      #   streamed walk-forward honest ruler + overfit→Optuna
│   │   ├── produce.py                 #   production trainer + 2026 OOS + ONNX + contract
│   │   ├── tune.py / loop.py          #   Optuna search + overfit-driven loop
│   │   ├── _memmap.py                 #   featurize-to-disk streaming (bounded RAM)
│   │   └── base.py / health.py        #   task-labeling contract + fold health checks
│   └── primitives/                    #   certified causal trigger primitives (pivots / barriers / indicators)
├── scripts/                           # ★ local/MPS and GPU SSL runners + data audits
├── databento/                         # Continuous-contract build + incremental update
├── tests/                             # Unit tests (pre-commit gated; torch-free by contract)
└── data/                              # Raw OHLCV CSVs (gitignored)
```
---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
