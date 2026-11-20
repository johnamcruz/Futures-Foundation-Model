# CLAUDE.md

Working notes for coding agents in this repository. The engineering contract is `docs/ML_DEVELOPMENT_PROCESS.md`; read it before
changing data, models or checkpoints.

## Foundation models: self-supervised only, no labels

FFM is the foundation layer. Every model trained here (any encoder, adapter or stage) learns from the market data itself and
nothing else.

- **No labels in training or checkpoint selection.** No trend, regime, direction, entry, outcome or other derived label may be a
  training target, a contrastive/supervised grouping, a loss weight, or the basis for picking a checkpoint.
- **Allowed objectives** are self-supervised: masked or block reconstruction, forecasting of the series' own future bars
  (the target is the data itself), and consistency/contrastive objectives whose positives come from the data alone (augmentations
  of the same window), never from a hand-made regime or outcome label.
- **New inputs** (for example order flow) enter as raw channels in a new versioned optional stream, with matched controls
  (shuffled and price-derived stand-ins) and no engineered indicators.
- **Labels live downstream.** Heads, probes that fit labels, entry rules and strategy evaluation belong in the strategy repositories
  that read FFM's frozen embeddings. The Probe Atlas is an evaluation of a frozen checkpoint; it never trains one.
- A change that adds a label-driven objective here is not done, whatever else it improves.
