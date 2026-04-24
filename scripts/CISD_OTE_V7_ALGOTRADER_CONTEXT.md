# CISD+OTE v7.0 — Algotrader Integration Context

This document is for the Claude instance in the algotrader repo.
It contains every technical fact you need to integrate `strategy_cisd_ote_v7.py`
with the live trading infrastructure. Read this before touching any strategy code.

---

## 1. What Changed from v5.1 → v7.0

| Dimension | v5.1 | v7.0 |
|---|---|---|
| Sequence length | 64 bars | **96 bars** |
| FFM features (backbone input) | 42 custom | **67 via `derive_features()`** |
| CISD strategy features | 28 | **10** |
| ONNX output[0] | `signal_probs` (post-softmax) | **`signal_logits`** (raw; apply softmax) |
| `cisd_features` input name | `cisd_features` | **`strategy_features`** |
| `candle_types` input | missing | **added as 7th ONNX input** |
| Session filter at inference | 07:00–16:00 hard gate | **removed** (confidence threshold replaces it) |
| `in_optimal_session` (09–11 ET) | hard filter | kept as **feature only** |
| Config hash | `b2951914` | **`7d2c9f8f`** |
| Checkpoint naming | `{fold}_{hash}.pt` | `{fold}_{hash}_done.pt` |
| ONNX filename | `strategy_hybrid.onnx` | **`cisd_ote_hybrid.onnx`** |
| Metadata sidecar | none | **`cisd_ote_hybrid_metadata.json`** |

**Why 28→10 CISD features?**
v5.1's CISD vector duplicated market context (HTF trend, EMA, volume, session
progress, structure state) because the backbone was weak. v7.0's 67-feature backbone
already encodes all of that from the raw sequence. The 10 remaining features cover only
zone geometry and trade mechanics that are not derivable from OHLCV alone.

---

## 2. ONNX Model — Input / Output Specification

### File
```
cisd_ote_hybrid.onnx          (F4 fold — production)
cisd_ote_hybrid_metadata.json (version, feature lists, thresholds, per-fold results)
```

### Inputs (7 total, order matters for positional ONNX calling)

| # | Name | Shape | dtype | Source |
|---|---|---|---|---|
| 0 | `features` | [B, 96, 67] | float32 | `derive_features()` → `get_model_feature_columns()` |
| 1 | `strategy_features` | [B, 10] | float32 | `_build_cisd_feature_vector()` |
| 2 | `candle_types` | [B, 96] | int64 | `ffm_df['candle_type']` |
| 3 | `time_of_day` | [B, 96] | float32 | `ffm_df['sess_time_of_day']` = (hour + min/60) / 24 |
| 4 | `day_of_week` | [B, 96] | int64 | `ffm_df['tmp_day_of_week']` = 0=Mon…4=Fri |
| 5 | `instrument_ids` | [B] | int64 | scalar per batch (see §5) |
| 6 | `session_ids` | [B, 96] | int64 | `ffm_df['sess_id']` (see §6) |

### Outputs (3 total)

| # | Name | Shape | dtype | Meaning |
|---|---|---|---|---|
| 0 | `signal_logits` | [B, 2] | float32 | Raw class logits; `softmax[:,1]` = signal probability |
| 1 | `risk_predictions` | [B, 1] | float32 | Predicted R:R (Softplus, always positive) |
| 2 | `confidence` | [B] | float32 | `max(softmax(signal_logits))` — pre-computed by model |

### Inference decision rule
```python
probs       = softmax(signal_logits[0])   # [2]
signal_prob = probs[1]
confidence  = max(probs)                  # or use output[2] directly

if signal_prob >= threshold:
    direction = +1 if cisd_vec[4] > 0 else -1   # zone_is_bullish at index 4
```

### Thresholds (walk-forward test set, 2022–2025)

| Label | Threshold | Combined precision | Profit factor |
|---|---|---|---|
| Conservative | 0.90 | 0.595 | 6.04 |
| Moderate | 0.80 | 0.514 | 4.26 |

F4 fold alone (2025 OOS data): precision @0.80 = 0.544, @0.90 = 0.603.

---

## 3. All 67 FFM Feature Columns (in order)

These are the columns returned by `get_model_feature_columns()` and must be fed
to ONNX input 0 in exactly this order.

```
Group 1 — Bar Anatomy (8)
  bar_range_atr, bar_body_atr, bar_upper_wick_atr, bar_lower_wick_atr,
  bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct, bar_direction

Group 2 — Returns & Momentum (8)
  ret_close_1, ret_close_3, ret_close_5, ret_open_close,
  ret_momentum_5, ret_momentum_10, ret_momentum_20, ret_acceleration

Group 3 — Volume Dynamics (6)
  vol_ratio_5, vol_ratio_10, vol_ratio_20,
  vol_change, vol_close_position, vol_delta_proxy

Group 4 — Volatility (6)
  vty_atr_zscore, vty_range_ratio_5, vty_range_ratio_20,
  vty_atr_of_atr, vty_realized_10, vty_realized_20

Group 5 — Session Context (5)
  sess_bars_elapsed, sess_dist_from_open, sess_dist_from_high,
  sess_dist_from_low, sess_dist_from_vwap

Group 6 — Market Structure (9)
  str_swing_high_dist, str_swing_low_dist, str_structure_state,
  str_dist_from_high_10, str_dist_from_low_10, str_range_position_10,
  str_dist_from_high_20, str_dist_from_low_20, str_range_position_20

Group 7 — CRT Sweep State (10)
  swp_1h_bull_active, swp_1h_bear_active, swp_1h_age_norm, swp_1h_magnitude,
  swp_4h_bull_active, swp_4h_bear_active, swp_4h_age_norm, swp_4h_magnitude,
  swp_tf_alignment, swp_dominant_dir

Group 8 — Candle Psychology (5) [candle_type is a separate embedding, not here]
  engulf_count, momentum_speed_ratio, wick_rejection,
  dir_consistency, bar_size_vs_session

Group 9 — HTF Price Context (6)
  htf_1h_close_pos, htf_1h_ret,
  htf_4h_close_pos, htf_4h_ret,
  htf_tf_alignment, htf_1h_structure

Group 10 — Volume Absorption & Order Flow (4)
  vol_cum_signed_5, vol_cum_signed_20, vol_absorption, vol_momentum_align
```

**`candle_type` is NOT in this list** — it feeds a dedicated embedding layer via ONNX input 2.

---

## 4. All 10 CISD Strategy Features (in order, with formulas)

These feed ONNX input 1 (`strategy_features`). Computed by `_build_cisd_feature_vector()`.

| # | Name | Formula | Notes |
|---|---|---|---|
| 0 | `zone_height_vs_atr` | `(fib_top - fib_bot) / atr` | clipped [0, 10] |
| 1 | `price_vs_zone_top` | `(close - fib_top) / zone_height` | negative = price below top |
| 2 | `price_vs_zone_bot` | `(close - fib_bot) / zone_height` | positive = price above bot |
| 3 | `zone_age_bars` | `(current_bar - created_bar) / ZONE_MAX_BARS` | clipped [0, 5] |
| 4 | `zone_is_bullish` | `+1.0` if bull zone, `-1.0` if bear | **also determines trade direction** |
| 5 | `cisd_displacement_str` | CISD ratio: `(highest_c - pot) / (top_level - pot)` | clipped [0, 5] |
| 6 | `had_liquidity_sweep` | `1.0` if wicked a swing within last 10 bars | binary |
| 7 | `entry_distance_pct` | bull: `(close - fib_top)/zh`; bear: `(fib_bot - close)/zh` | negative = inside zone; clipped [-2, 5] |
| 8 | `risk_dollars_norm` | `risk_pts × point_value / MAX_RISK_DOLLARS` | clipped [0, 5] |
| 9 | `in_optimal_session` | `1.0` if 09:00 ≤ ET hour < 11:00 | binary; feature only, not a hard gate |

**ZONE_MAX_BARS = 40, MAX_RISK_DOLLARS = 300.0**

---

## 5. Instrument ID Encoding

`instrument_ids` is a scalar int64 per batch item (shape `[B]`, not `[B, seq]`).

```python
INSTRUMENT_MAP = {
    "ES": 0, "NQ": 1, "RTY": 2, "YM": 3,
    "GC": 4, "SI": 5, "CL": 6, "NKD": 7,
}
MICRO_TO_FULL = {"MES": "ES", "MNQ": "NQ", "MRTY": "RTY", "MYM": "YM", "MGC": "GC"}
```

Map micros to their full symbol before lookup. Unknown instrument → default 0.

---

## 6. Session ID Encoding (must match training exactly)

`session_ids` is computed from ET (Eastern Time) hours directly.
`derive_features()` computes this as `ffm_df['sess_id']`.

```
0 = pre-market  (hour < 3)
1 = London      (3 ≤ hour < 8)
2 = NY AM       (8 ≤ hour < 12)
3 = NY PM       (12 ≤ hour < 16)
```

`time_of_day` = `(hour + minute/60) / 24` — float32 in [0, 1], from `ffm_df['sess_time_of_day']`.

---

## 7. Candle Type Encoding

Computed by `futures_foundation.candle_psychology`. Comes from `ffm_df['candle_type']` (int64).

```
0 = doji
1 = bull strong (large bull body, close ≥ 60% of range)
2 = bear strong (large bear body)
3 = bull pin (long lower wick, small body near top)
4 = bear pin (long upper wick, small body near bottom)
5 = neutral
```

---

## 8. CISD Detection Parameters (must match training)

```python
SWING_PERIOD        = 6       # pivot point lookback
TOLERANCE           = 0.70    # minimum CISD displacement ratio
EXPIRY_BARS         = 50      # swing point expiry
LIQUIDITY_LOOKBACK  = 10      # bars to look back for sweep confirmation
ZONE_MAX_BARS       = 40      # zones expire after this many bars
FIB_1               = 0.618   # OTE fib level 1
FIB_2               = 0.786   # OTE fib level 2
HTF_RANGE_BARS      = 96      # bars for P/D midpoint calculation
DISP_BODY_RATIO_MIN = 0.50    # displacement candle body/range minimum
DISP_CLOSE_STR_MIN  = 0.60    # displacement candle close strength minimum
```

---

## 9. Walk-Forward Training Results

Model trained on ES, NQ, RTY, YM, GC (5min bars). Folds are strictly sequential.

| Fold | Train end | Val end | Test end | Prec @0.80 | Prec @0.90 |
|---|---|---|---|---|---|
| F1 | 2022-01-01 | 2022-07-01 | 2023-01-01 | — | — |
| F2 | 2023-01-01 | 2023-07-01 | 2024-01-01 | — | — |
| F3 | 2024-01-01 | 2024-07-01 | 2025-01-01 | — | — |
| F4 | 2025-01-01 | 2025-06-01 | 2025-12-01 | **0.544** | **0.603** |
| **Combined** | | | | **0.514** | **0.595** |

Combined profit factor: **4.26 @ 0.80**, **6.04 @ 0.90**.

Warm-start compounding confirmed: each fold opens at higher initial val F1 than the
previous (F1 E1: 0.000 → F2 E1: 0.326 → F3 E1: 0.437 → F4 E1: 0.487).

---

## 10. Checkpoint and File Structure

```
OUTPUT_DIR = AI_Models/CISD_OTE_Hybrid_v7/

  cisd_ote_hybrid.onnx              ← production model (F4)
  cisd_ote_hybrid_metadata.json     ← version, feature lists, thresholds, per-fold results
  F1_7d2c9f8f_done.pt
  F2_7d2c9f8f_done.pt
  F3_7d2c9f8f_done.pt
  F4_7d2c9f8f_done.pt
```

Re-running `cisd_ote.py` in Colab skips training when `*_done.pt` files exist (matching
config hash `7d2c9f8f`). Only the labeling step and ONNX export run.

---

## 11. P/D Filter and Trend Filter (training only)

During training, two hard filters shaped the labels:

- **P/D filter** (`USE_PD_FILTER=True`): Bull zones must be in discount. Bear zones must
  be in premium, **except** if `had_sweep=1` (sweep override bypasses P/D gate for bears).
- **Trend filter** (`USE_TREND_FILTER=False`): Disabled. The backbone's `htf_1h_structure`
  feature provides this context; the model learns when to follow or fade trend.

**Do not re-apply these as hard gates at inference.** The model has internalized them.
Gate only on confidence threshold.

---

## 12. Integration Checklist

- [ ] Copy `scripts/strategy_cisd_ote_v7.py` to your strategies directory
- [ ] Install `futures_foundation` (`pip install -e path/to/Futures-Foundation-Model`)
- [ ] Point `onnx_path` to `cisd_ote_hybrid.onnx`
- [ ] Verify `instrument` string matches `INSTRUMENT_MAP` keys (map micros via `MICRO_TO_FULL`)
- [ ] Feed raw OHLCV with a `datetime` column (not index) to `on_bar()`
- [ ] Provide at least 300 bars of history before first call (`get_warmup_length()`)
- [ ] Select threshold: 0.90 = conservative (PF 6.04), 0.80 = moderate (PF 4.26)
- [ ] Do **not** re-apply session filter — `is_trading_allowed()` returns True always
- [ ] Direction is determined by `cisd_vec[4]` (zone_is_bullish), not by model logits
- [ ] `confidence` output is already `max(softmax)` in [0.5, 1.0] — no rescaling needed
- [ ] `time_of_day` ONNX input is **float32**, not int64

---

## 13. Silent Failure Hazards

| If you do this… | It will fail silently like this… |
|---|---|
| Use `opset_version=17` in ONNX export | Crash: `No module named 'onnxscript'` |
| Keep model on GPU before `export_onnx()` | Crash: "mat1 is on cpu, different from other tensors on cuda:0" |
| Use old input name `cisd_features` | ONNX session raises key error |
| Feed `signal_probs` threshold logic (v5.1) | Thresholding raw logits — completely wrong |
| Miss `candle_types` 7th input | ONNX input count mismatch |
| Feed `instrument_ids` as [B, 96] instead of [B] | Shape error or silent misalignment |
| Use SEQ_LEN=64 (v5.1) | Backbone attention mask mismatch |
| Feed 42 features (v5.1) instead of 67 | Backbone linear layer size mismatch; crash |
| Apply hard session gate at inference | Filters signals the model was trained to see |

---

## 14. Re-Generating the ONNX File (without retraining)

All 4 `_done.pt` checkpoints exist. To regenerate `cisd_ote_hybrid.onnx`:

**Option A — standalone script (fastest, no labeling):**
```
Run colabs/export_cisd_ote_v7.py in Colab
```

**Option B — full pipeline (re-labels, then exports):**
```
Run colabs/cisd_ote.py in Colab
Training is skipped (done checkpoints detected); labeling takes ~10–15 min.
```

Both require `git pull` first to get the opset 14 fix in `trainer.py`.

---

## 15. Feature Computation Notes

- **`vty_atr_raw`** is in `ffm_df` but NOT in `get_model_feature_columns()`. It is metadata
  used to normalise CISD features. Read it from `ffm_df['vty_atr_raw']`; don't feed it to the model.
- `sess_id`, `sess_time_of_day`, `candle_type`, `tmp_day_of_week` come from `ffm_df` but
  are passed as separate ONNX inputs, not part of the 67-feature tensor.
- Feed OHLCV with ET timestamps — `derive_features()` uses hour values directly, no tz conversion.

---

## 16. Key Constants Summary

```python
SEQ_LEN                = 96
NUM_FFM_FEATURES       = 67
NUM_CISD_FEATURES      = 10
THRESHOLD_CONSERVATIVE = 0.90   # PF 6.04
THRESHOLD_MODERATE     = 0.80   # PF 4.26
MODEL_VERSION          = 'cisd_ote_v7_0'
CONFIG_HASH            = '7d2c9f8f'
```
