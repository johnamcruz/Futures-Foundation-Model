# Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

FFM adapts **Chronos-2 Small** to causal futures OHLCV and exposes its learned
representations through a model-agnostic classifier seam. The foundation learns
from price and volume bars; private trading labels, entry rules, stops, targets,
and execution policy belong in downstream projects.

## Current status

| Capability | Status |
|---|---|
| Chronos-2 multivariate OHLCV integration | Available |
| Masked OHLCV pretraining | Validated foundation checkpoint |
| Frozen Chronos-2 embedding classifier | Available |
| Public Probe Atlas representation evaluation | Available |
| LangGraph-backed automated SSL workflow | Available |
| Volume-Structure SSL | Experimental; promotion validation pending |
| Trend/Chop contrastive SSL | Experimental; promotion validation pending |
| Fast grouped attention | Optional prototype; disabled by default |

“Available” means the implementation and its contracts exist. Experimental SSL
stages are not promoted merely because they complete training: they must beat
the validated Mask checkpoint under matched causal evaluation without losing
important retained capabilities.

## Design

> Learn reusable market context first; train decisions separately.

Four rules define the foundation:

1. **OHLCV is the model input.** SSL consumes completed open, high, low, close,
   and volume bars—not order-book data, indicators, or trading outcomes.
2. **Representations and decisions are separate.** Downstream projects attach
   lightweight task heads through the classifier interface.
3. **Time is a hard boundary.** Windows, labels, validation periods, and sealed
   holdouts are constructed causally.
4. **Artifacts are authenticated.** Data, configuration, parent checkpoint,
   controls, seed, and reports are recorded so a result can be reproduced.

## Architecture

```text
continuous-contract OHLCV (9 tickers x 4 native timeframes)
                         |
                         v
              Chronos-2 Small backbone
                         |
          +--------------+---------------+
          |                              |
          v                              v
 validated Mask SSL             candidate SSL refinements
                                 - Volume Structure
                                 - Trend / Chop contrastive
          |                              |
          +--------------+---------------+
                         |
                         v
           frozen causal Chronos embeddings
                         |
                         v
       downstream Pivot / Trend / Expansion / Entry heads
                         |
                         v
        temporal validation -> production packaging
```

The default research universe contains ES, NQ, RTY, YM, GC, SI, CL, ZB, and ZN
at 1-, 3-, 5-, and 15-minute resolution. Each five-channel OHLCV stream is an
individual multivariate series that shares the same Chronos backbone, allowing
the representation to learn recurring structures across markets and scales.

## Installation

FFM requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Large datasets, caches, checkpoints, and campaign state should live outside the
Git checkout—preferably on an SSD. Repository paths may point to those assets
without copying them back into the working tree.

## Representation learning

### Masked OHLCV pretraining

Masked pretraining is the validated base. Causal OHLCV patches are hidden and
reconstructed so Chronos must infer local price, volatility, participation, and
temporal structure from the visible context. Training uses LoRA and publishes a
native Chronos adapter checkpoint; temporary training state is not required by
downstream consumers.

The pinned repository baseline is `checkpoints/chronos2_mask_full/`; its
completion receipt is `checkpoints/chronos2_mask_full.report.json`.

```bash
./.venv/bin/python scripts/chronos/chronos2_ssl_pretrain.py \
  --parent /path/to/pinned/chronos2/adapter \
  --data-dir /path/to/ohlcv \
  --out-dir /path/to/ssd/mask-run \
  --timeframes 1min,3min,5min,15min \
  --device mps
```

### Volume-Structure SSL

The experimental Volume-Structure stage learns from OHLCV-only objectives for
masked volume/price reconstruction, participation, concentration versus
dispersion, displacement with abnormal volume, and temporal ordering. Temporary
objective heads may supervise training, but they are discarded. Promotion is
based on the resulting Chronos checkpoint itself.

```bash
./.venv/bin/python scripts/chronos/chronos2_ssl_volume_structure.py \
  --parent /path/to/mask/checkpoint \
  --base-snapshot /path/to/pinned/chronos2-small \
  --data-dir /path/to/ohlcv \
  --out-dir /path/to/ssd/volume-run \
  --preflight-only
```

Remove `--preflight-only` only after the data and lineage contracts pass.

### Trend/Chop contrastive SSL

The experimental balanced Kaufman stage uses causal efficiency-ratio teachers
to organize direction-agnostic trend and chop contexts while preserving native
Chronos geometry. Up and down trends share the same regime concept; direction
remains a downstream decision. This stage is pending matched promotion evidence.

```bash
./.venv/bin/python scripts/chronos/chronos2_ssl_contrastive.py \
  --parent /path/to/validated/parent/checkpoint \
  --parent-report /path/to/validated/parent/report.json \
  --base-snapshot /path/to/pinned/chronos2-small \
  --data-dir /path/to/ohlcv \
  --out-dir /path/to/ssd/trend-chop-run \
  --preflight-only
```

Every SSL candidate must remain usable through its encoder checkpoint alone.
Auxiliary heads, decoders, and trainer state are training tools—not serving-time
dependencies.

## Automated SSL development

FFM can execute its existing SSL commands through the shared **ML Training
Loop**, whose internal workflow is implemented with LangGraph. FFM continues to
own data preparation, objective implementations, Chronos/LoRA training, Probe
Atlas, checkpoint validation, and promotion rules. LangGraph owns durable graph
execution, routing, checkpoint/resume, interrupts, and recovery.

A versioned JSON workflow declares:

- immutable data, temporal, checkpoint, seed, and holdout contracts;
- preflight, training, Probe Atlas, control, comparison, and packaging stages;
- required artifacts and authenticated receipts;
- bounded, skill-directed reasoning when an experiment requires revision; and
- the exact settings a permitted revision may change.

Run or resume a workflow with the same run identifier:

```bash
./.venv/bin/python scripts/chronos/chronos2_ssl_training_loop.py \
  --config /path/to/ssl-workflow.json \
  --run-id chronos2-volume-v3
```

The workflow resumes completed stages from durable receipts and fails closed on
lineage drift or malformed artifacts. Reasoning may diagnose evidence and select
an allowed scientific revision; it may not silently change the data universe,
temporal split, holdout, parent checkpoint, controls, or promotion criteria.

## Probe Atlas

Probe Atlas compares frozen representations on the same causal 9-ticker x
4-timeframe corpus. It measures retained market-state information and generic
forward expansion, compression, magnitude, trend, and direction probes with
per-stream and worst-stream results.

```bash
./.venv/bin/python scripts/chronos/chronos2_probe_atlas.py \
  --checkpoint /path/to/chronos2/checkpoint \
  --control real \
  --window 256 \
  --horizons 5,10,20,50
```

Run matched `real`, `shuffle`, and `random` controls. A candidate is promoted
only from an identical corpus, split, probe configuration, solver, seed, and
checkpoint contract. Probe Atlas measures representation quality; it does not
claim trading profitability.

## Downstream classifier seam

Downstream training refers to the backbone by registered name rather than model
implementation:

```python
from futures_foundation.finetune.classifier import get_classifier

classifier = get_classifier(
    "chronos2_frozen",
    backbone_ckpt="/path/to/chronos2/checkpoint",
    device="mps",
    pool="reg",
    with_features=False,
)
```

The encoder runs in an isolated worker, embeddings can be cached by authenticated
identity, and a lightweight fold-specific head consumes the representation.
This keeps the walk-forward harness independent of the foundation backbone.

## Experimental grouped-attention switch

An optional prototype ports the split grouped-attention approach proposed for
Chronos-2:

```bash
FFM_CHRONOS2_FAST_GROUP_ATTENTION=1 <embedding-command>
```

It is disabled by default. Because its floating-point output is not bit-identical
to the legacy worker, it must start a fresh cache family and pass matched MPS
throughput and representation validation before promotion. Never enable it
halfway through an existing cache build.

## Data contract

Input files contain one completed bar per row:

```text
timestamp,open,high,low,close,volume
```

The public data layer validates timestamps, OHLC ordering, finite values,
continuous-contract construction, temporal identity, and source manifests.
Tick, quote, order-book, and unfinished higher-timeframe data are outside the
current foundation contract.

## Repository map

```text
futures_foundation/
  finetune/
    classifiers/chronos2/  # Chronos integration, SSL stages, frozen embeddings
    pretext/               # shared SSL objective infrastructure
    classifier.py          # model-agnostic classifier registry
  orchestration/           # FFM adapter for the ML Training Loop
scripts/
  chronos/                 # SSL, Probe Atlas, and benchmark entrypoints
  probe_atlas.py           # public representation evaluation
data/                      # local OHLCV files; large data remains untracked
temp/                      # local experiment state; untracked
```

## Scope

FFM is the public representation and validation foundation. Strategy-specific
labels, economic targets, trade selection, risk, execution, and live telemetry
belong in downstream repositories. A strong representation is necessary, but
profitability must still be demonstrated with causal temporal evaluation and
realistic trading economics.
