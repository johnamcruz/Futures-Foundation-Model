# Chronos-2 Forecast-Direction SSL: Frozen Experiment Contract

Status: FROZEN 2026-09-28, before any compute. Owner and approver: John Cruz.
Code: `futures_foundation/finetune/classifiers/chronos2/forecast_ssl.py`,
`scripts/chronos/chronos2_ssl_forecast_direction.py`,
`tests/test_chronos2_forecast_ssl.py`.

## 1. Hypothesis and falsifier

Probe Atlas shows the Mask v5 encoder predicts forward range expansion
(pooled AUC 0.84–0.86) but not forward direction (0.509–0.533 at h=5–50).
None of the lineage objectives rewards the sign of the next move. Mask
reconstruction interpolates. Native pinball spends under 1% of its loss on
direction; see `chronos2_direction_research.md` §2.

**H1 (A1).** Fine-tuning Mask v5's LoRA on the native multi-horizon quantile
forecast of our own streams improves the forecast itself: lower scaled
weighted quantile loss (WQL) than the zero-shot parent.

**H2 (A2).** Adding an explicit direction term to that objective raises
forward direction AUC at h ∈ {5, 10, 20, 50} above both A1 and a causal
baseline, without degrading the forecast.

The direction term is a BCE on P(up) read from the model's own quantile CDF.

**Falsifier.** H2 is rejected if A2 fails the direction gate in §8 on the
outer fold. Stop rule: if H2 is rejected, record the null. The next step is
the magnitude-blind contrastive stage F2 (mirrored-future hard negatives).
L2/order-flow data is the last resort, not the next step.

The objective is self-supervised and strategy-agnostic. Targets are the
same stream's future closes. It uses no trade outcomes, costs, entries,
stops or sizing.

## 2. Parent and model

- Parent: Mask v5, `temp/chronos2_small_36stream/mask_full/checkpoint`, tree
  sha256 `de8dc18eedbb975c06847938365befa5fe0d527ec9d5ff8a96628c5ce3fe9547`.
  It is the encoder consumed by every `ffm-strategies` expansion artifact
  (verified 2026-09-28: 17 `encoder_sha256` references).
- Lineage: `native_1000` (`0b727917…`, native pinball, h=32) → Mask v5.
- Base: `autogluon/chronos-2-small` pinned snapshot, revision
  `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`. Loaded through
  `_load_trainable_adapter(..., base_snapshot=...)`.
- Adaptation: continue training the parent's existing LoRA (r=8, α=16 on
  attention q/k/v/o and `output_patch_embedding.output_layer`). No full
  fine-tuning, and no new trainable modules. Output: a PEFT adapter with the
  same shape and base identity as the parent, loadable by every existing
  consumer.

## 3. Data universe and observation contract

- 9 tickers (ES NQ RTY YM GC SI CL ZB ZN) × 4 timeframes (1, 3, 5, 15 min) =
  36 authenticated continuous streams. Manifests are checked by
  `seal_continuous_streams`. OHLC relationships and finiteness are checked by
  `_validate_stream`.
- Timestamps are bar opens. Decision time is the bar **close** (open + tf).
  Bars closing at or after 2026-01-01 are dropped at load time, so the
  sealed holdout never enters memory.
- One task is one stream: its five OHLCV variates, raw, forming one Chronos
  group. Chronos instance normalization scales each variate.
- Decision bar t. The context is bars t−255…t (256 bars, fully observed).
  The target is close[t+1…t+64] (4 output patches of 16).
- Direction label at h: `close[t+h] > close[t]`. This is identical to Probe
  Atlas `future_direction_h`.

## 4. Splits

The parent lineage trained on bars through 2025-07-14 (checked in
`mask_full/data_preflight_*.json`), including a native *forecasting*
fine-tune. Only the bars after that are unseen.

| Period | Decision close | Target end close | Use |
|---|---|---|---|
| train | ≥ bar 255 | < 2025-07-14 | LoRA updates, causal-baseline fit |
| select | ≥ 2025-07-15 | < 2025-10-01 | early stopping, λ choice |
| outer | ≥ 2025-10-01 | < 2026-01-01 | one-shot promotion evidence |
| sealed | 2026-01-01 → | — | untouched until the full procedure is frozen |

The 64-bar target reserve is enforced by the anchor rule: the target-end
close must fall inside the period. No target crosses a boundary. Contexts may
reach back into earlier periods, because the inputs are observable at the
decision time.

## 5. Arms (one change at a time)

| Arm | Starting weights | Objective | Compared with |
|---|---|---|---|
| A0 | Mask v5, zero-shot | none (inference) | reference for A1 |
| A1 | Mask v5 LoRA | native pinball on close[t+1…t+64] | A0 (forecast gate) |
| A2 | Mask v5 LoRA | A1 + λ · mean_h BCE(P_up_h, y_h), h ∈ {5, 10, 20, 50} | A1 (direction gate) |

A0-base (untouched Chronos-2-small, zero-shot) is scored as a diagnostic only.

Frozen hyperparameters:

- AdamW, lr 1e-5 (the `native_1000` rate), weight decay 0.01, gradient clip
  1.0.
- 32 windows per step, with streams sampled uniformly and anchors uniformly
  within the train period.
- 100 steps per epoch, at most 20 epochs, patience 3.
- Selection metric per epoch, on a fixed set of 100 select anchors per stream
  with equal stream weight: native pinball (A1), or pinball + λ·BCE (A2).
- The parent's selection metric is the starting incumbent. If no epoch beats
  it, the result is the parent and is reported as such.
- Direction weights: uniform. |z|-weighting is deferred, because it
  reintroduces magnitude.
- λ ∈ {0.3, 1, 3, 10}. λ is chosen on **select** only: the highest mean
  per-stream AUC averaged over h5, h10, h20 and h50, subject to mean scaled
  WQL no worse than A1 + 2%.
  - *Amendment, 2026-09-28, after smoke and before any full run.* The
    original grid was {0.1, 0.3, 1.0}.
  - The smoke measured the native loss at about 9 (a sum over 13 quantiles)
    against a BCE of about 0.7. The original grid therefore capped the
    direction term at about 7% of the objective.
  - The new grid reaches parity at λ ≈ 10. The change is based on loss scale
    only; the smoke AUCs came from 2 streams and 100 anchors and were not
    used.
  - The selection criterion now spans all four horizons, per the owner.
- Seeds: 0 for smoke and selection. Seeds 0, 1 and 2 for the frozen A1 and
  chosen-λ A2 before the outer read.

## 6. Metrics (per stream × horizon; pooled summaries are secondary)

- Direction:
  - AUC of P(up) = 1 − F(close[t]) from the 13-knot piecewise-linear CDF;
  - Hanley–McNeil SE using the effective n = min(n, span/h + 1);
  - sign accuracy of q50 − close[t];
  - Brier score;
  - base rate (always-long accuracy).
- Forecast:
  - scaled WQL = model pinball ÷ pinball of a flat forecast at close[t];
  - 10–90% coverage.
- Controls:
  - shuffled-label AUC (fixed permutation);
  - a **causal baseline**: per-stream logistic regression on r1, r5, r20 and
    r50 scaled by σ20, log σ20, the 20-bar range ratio, and hour sin/cos. It
    is fit on train-period anchors only;
  - the Probe Atlas REG-probe incumbent, quoted for reference only, because
    its rows differ.
- Anchors: evenly spaced, up to 2000 per stream per period.

## 7. Artifacts

Every run writes:

- a JSON report with the schema, parent and checkpoint tree hashes, base
  identity, sealed data provenance, split boundaries, config, git HEAD and
  dirty flag, and history;
- evaluation JSONs per arm and period;
- a Markdown summary.

The checkpoint is a PEFT adapter. The report records timing (seconds per
step) for budget decisions.

## 8. Promotion gates (evaluated on outer only after §5 is frozen)

- **Forecast gate (A1 vs A0).** At each horizon:
  - scaled WQL is lower on ≥ 27 of 36 streams;
  - no stream is worse by more than 5% relative.
- **Direction gate (A2 vs A1).** At **each** of h5, h10, h20 and h50 (owner
  requirement, 2026-09-28: the goal is short-term direction at all four
  horizons):
  - mean per-stream ΔAUC ≥ +0.005, with a lower 95% bound > 0 (SE across
    streams);
  - ΔAUC > 0 on ≥ 24 of 36 streams;
  - mean A2 AUC exceeds the mean causal-baseline AUC;
  - mean scaled WQL is no worse than A1 + 2%.
- **Retention gate (A1 and A2 vs Mask v5, Probe Atlas).** For every Atlas
  probe except the forward-direction targets (`pred_*direction*`):
  - pooled AUC drops by at most 0.01;
  - no single stream drops by more than 0.03.
  - Expansion, volatility, trend, squeeze, session and
    `ret_structural_direction` are all covered.
- **Probe Atlas confirmation (owner requirement).** There is one Atlas run
  each for Mask v5, final A1 and final A2, with identical settings: REG pool,
  window 256, fit before 2024, and `--eval-start 2025-07-15`.
  - The default 2025 evaluation would include 2025-01 → 07-14. A2 trained
    direction labels on those bars, so they are excluded for every
    checkpoint.
  - The same run supplies the retention gate above and the out-of-sample
    `pred_direction_h5/10/20/50` probes on the REG embedding that downstream
    strategies consume.
  - A2 is confirmed on Atlas if its mean REG direction AUC exceeds A1's and
    Mask v5's at each horizon, with ΔAUC > 0 on ≥ 24 of 36 streams.
  - The run reads outer-period labels, so it happens only after §5 selection
    is frozen.
- The select period must show the same sign as outer. A result that appears
  only on outer is not promoted.
- A2 passing is a representation result, not a trading claim. Economics
  versus cost stay with the consumer repository.

## 9. Exception (recorded before any run)

- **Rule affected.** Evaluation periods must be unseen by every ancestor,
  including their selection.
- **Deviation.** `native_1000` and Mask v5 early-stopped on 8 chronological
  validation windows per timeframe that lie in 2025-07-14 → 2025-12-31,
  which overlaps select and outer.
- **Bound.** At most 8 windows per timeframe. They selected on generic
  pinball or mask loss, never on direction.
- **Evidence risk.** Low optimistic bias on WQL in select and outer. It is
  negligible for direction.
- **Approver.** John Cruz, 2026-09-28.
- **Expiry.** When the lineage is retrained with a 2024-01-01 cutoff, or when
  the sealed 2026 confirmation is run.

## 9a. Smoke findings (2026-09-28; NQ/ES@3min, 100 select anchors; AUCs are noise)

| Checkpoint | scaled WQL h5–h50 | 80% coverage | Reading |
|---|---|---|---|
| base chronos-2-small | 0.64–0.67 | 0.76–0.81 | healthy forecaster |
| `native_1000` | 0.64–0.67 | 0.75–0.80 | healthy; native fine-tune did no harm |
| **Mask v5** | **1.23–1.48** | **0.17–0.36** | native forecast path broken: intervals collapsed, worse than a flat forecast |

- **Cause.** The Mask stage trained the encoder only through a temporary
  reconstruction decoder. The future-patch → quantile-head path drifted.
- **What it does not affect.** Consumers read REG embeddings, not the
  forecast head, so the expansion strategies are unaffected.
- **What it means for this experiment:**
  - A1 will partly be a *repair*, so the A1 > A0 forecast gate is expected to
    pass and carries little information.
  - The parent's direction BCE is 1.71 (a coin flip is 0.69): it is
    confidently wrong.
  - The informative comparison remains A2 vs A1.
- **Timing.** About 1.0 s per step on MPS with 32 windows. A full run of up
  to 2000 steps with early stopping takes under an hour, so everything runs
  locally.

## 10. Deferred (backlog, not before promotion)

- Resume/crash recovery for training.
- |z|-weighted direction loss.
- Joint 45-variate tasks.
- F2 mirrored-negative contrastive.
- Calibration of P(up).

## 11. Tests that pin the objective to direction

`tests/test_chronos2_forecast_ssl.py`, with torch tests run under
`CHRONOS_TORCH_TESTS=1`:

- The direction term is unchanged when every realized move is scaled 10×.
  It cannot profit from magnitude.
- Its gradient moves the quantile knots toward the realized side.
- On a toy forecaster, the move sign depends weakly on a feature, while the
  move size is dominated by an unrelated volatility factor. Direction-only
  training learns the sign (positive drift, AUC > 0.55).
- The direction gate covers all four horizons and fails if any one regresses.
  The retention gate fails on an expansion regression or a single-stream
  collapse, and ignores the forward-direction targets.
- Split contracts:
  - targets never cross a period boundary;
  - periods are ordered and end at the holdout;
  - causal-baseline features are unchanged when future bars change.
