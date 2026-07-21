# Related-Series Context Fusion

Status: experimental; production Mantis/NextLeg remains the incumbent.

## Design source and adaptation

Amazon Chronos-2 interleaves temporal self-attention with group self-attention. Its caller passes
one `group_id` per variate; the model constructs a mask that allows information exchange only
between variates sharing the same group. See the official Apache-2.0 implementation in
[`src/chronos/chronos2/model.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py).

FFM adopts the grouping principle, not the Chronos-2 backbone:

| Chronos-2 mechanism | Mantis adaptation |
|---|---|
| One encoder processes every variate | One compact Mantis encoder processes every OHLCV member |
| `group_ids` prevent cross-task attention | Related members are explicit in `[batch, related, ...]`; batches never mix |
| Group attention operates throughout the large encoder | One small attention block operates on final Mantis embeddings |
| Missing covariates are masked | Missing, stale, or insufficient-history contexts are masked |
| General-purpose series tasks | Primary futures stream plus exact-TF and sibling roles |

The attention result enters through `primary + tanh(gate) * context`. `gate` initializes to zero,
so an untrained fusion block is exactly the incumbent primary-only embedding. This makes the
experiment reversible and gives the baseline-parity test a precise invariant.

## Causal time contract

Repository CSV timestamps represent bar opens. A bar becomes usable at:

```text
bar_close = timestamp + timeframe_minutes
```

For a primary decision, each related stream uses the last bar satisfying:

```text
related_bar_close <= primary_bar_close
```

This prevents an unfinished 15-minute candle from entering a 1/3/5-minute example. The alignment
also requires sufficient history for the full context window and rejects related streams whose
last closed bar exceeds the configured staleness tolerance. It never substitutes the next bar.

## Roles and targets

The stable group layout is:

1. primary series;
2. one slot per configured exact timeframe of the same ticker; and
3. one same-timeframe sibling slot.

The exact-timeframe slot matching the primary is masked to avoid a duplicate primary token.
NextLeg and candle targets always come from the primary stream. Related members are context only;
they cannot change labels or objective boundaries.

## Checkpoint contract

The experiment writes a versioned `mantis-related-v1` dictionary containing:

- merged ordinary Mantis encoder weights;
- fusion weights and learned gate; and
- architecture/alignment metadata.

Plain checkpoints remain plain. Existing primary-only loaders extract the ordinary encoder state
when given a related checkpoint, while grouped consumers must call `embed_related_windows` and
provide `[N,R,C,L]` windows plus `[N,R]` masks explicitly.

## Required promotion evidence

Never judge the feature by training loss alone. A candidate must pass matched-seed comparisons:

1. primary-only incumbent;
2. same-ticker multi-timeframe context;
3. sibling-only context;
4. combined context;
5. shuffled related context while preserving primary inputs and targets;
6. missing-context and stale-context stress tests;
7. future-data mutation and unfinished-candle tests;
8. walk-forward economic, trend-capture, and temporal-stability gates.

Incremental lift must disappear or reverse under the shuffled-related control. Otherwise the
fusion is not demonstrating useful cross-series information.

The runner exposes this matched control as `--related-control shuffle`; `drop` provides exact
primary-only behavior through the same architecture and checkpoint path.
