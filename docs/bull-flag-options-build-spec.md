# Bull-Flag Breakout Options Strategy — Build Specification

> **Audience:** an AI coding agent (e.g. Claude) building this strategy end to
> end inside the Futures-Foundation-Model (FFM) repo. Every step below is
> executable. Follow the phases in order — each phase gates the next.
>
> **Status of this document:** build spec + working doc (single source of
> truth for the build). The strategy described here does not yet exist in
> this repo — you are building it.
>
> **Created:** 2026-07-10 12:36 PDT · **Plan file:**
> `~/.claude/plans/as-an-expert-research-federated-lamport.md` (approved)

> NEXT: user reviews & approves this spec (Phase A gate) → then Phase B:
> build `futures_foundation/patterns/bull_flag.py` + mechanical backtest.

---

## 0. Purpose & Scope

A **bull-flag breakout swing strategy on daily equity/ETF bars** whose
validated price-action signal is then expressed through **long calls / call
debit spreads**, executed live (paper first) via **IBKR**. It reuses FFM's
Chronos+XGBoost honest-ruler harness (`StrategyLabeler` protocol) for the ML
selection layer, exactly as `supertrend_chronos.py` does for futures.

This is a genuine extension of the repo into two new territories at once:
**equities** (repo has zero equity data today) and **options** (repo has zero
options infrastructure today). The build is therefore strictly phased: the
underlying price-action edge must pass the pre-registered honest-ruler gate
**before** any options-specific code is written, and options must survive
their own backtest gate **before** any live-execution code is written.

### Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Pattern timeframe | **Daily bars only (v1)** | Bull flag is a multi-day swing pattern (Bulkowski/O'Neil); daily maps to 3–8 week options DTE. 4h = documented fallback only if events prove too sparse |
| Universe (v1) | **12 tickers:** SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX | Deep options chains, high ADV; single names included per user decision (richer event count) |
| History depth | **10 years** daily (≈2016→present) | Event-count viability for the harness's ≥50-events-per-fold minimum |
| Earnings-gap poles | **Included + flagged** (`is_earnings_gap` feature) | User decision — let the model judge materiality, don't hard-filter |
| ML harness | **`StrategyLabeler` protocol** (Chronos honest ruler), mirroring `supertrend_chronos.py` | `evaluate()` is strategy-owned → bespoke measured-move exit fits; `pipelines/xgboost`'s hard-wired ATR-trail exit does not |
| Features | **Bespoke small set** (§4.4) — NOT the FFM-68 | ~25 of 68 FFM columns (intraday session, 1H/4H HTF) are degenerate on a daily base series |
| Data + execution vendor | **IBKR (`ib_insync`) end-to-end** | One vendor for historical bars, chains, greeks, and orders → no train/live skew |
| Bootstrap data (Phase B only) | `yfinance` daily bars | Free, adjusted, good enough to prove/kill the pattern before IBKR data spend |
| Options pricing | **In-repo Black-Scholes** (`futures_foundation/options/black_scholes.py`) | ~40 lines of numpy+scipy; repo precedent = hand-roll over heavy deps (`regime.py`) |
| Risk guard | **New local module** `risk_guards_options.py` in this repo | User decision — options bot does NOT reuse the external futures-bot risk guard (`FFM_LIVEBOT_DIR`) |
| Live-money safety | Paper-account guard (`IBKRBroker._is_paper`) mandatory in every execution path | Repo-standard, non-negotiable |

### Target module layout

```
futures_foundation/patterns/
├── __init__.py
└── bull_flag.py                 # P2: detect_pole / detect_flag / detect_breakout / scan_bull_flags
futures_foundation/options/
├── __init__.py
└── black_scholes.py             # P6: bs_price/delta/theta/vega + implied_vol_newton
bull_flag_mechanical.py          # P2: rule-based no-ML backtest (the kill-or-continue gate)
ibkr/
├── build_equity_daily.py        # P3: bulk ADJUSTED_LAST daily bars -> data/{TICKER}_1day.csv
└── append_update.py             # P3: incremental splice
audit_equity_data_quality.py     # P3: QA scanner (equity-adapted)
bull_flag_chronos.py             # P4: BullFlagBreakoutLabeler (StrategyLabeler protocol)
options_backtest.py              # P6: BS overlay replay of taken signals
risk_guards_options.py           # P7: local risk guard (loss limits, concurrency, premium stop)
analyze_options_concentration.py # P7: correlated-exposure diagnostic
ibkr_options_broker.py           # P8: IBKROptionsBroker + OptionsRatchetManager
test_ibkr_options_broker.py      # P8: offline tests, zero TWS connection
```

(P2…P8 = build phases, §9.)

---

## 1. Data Format & Ingestion

### Schema (identical to repo convention)

`data/{TICKER}_1day.csv`, columns exactly:

```
datetime, open, high, low, close, volume
```

`datetime` timezone-aware UTC, sorted ascending, one row per trading day.
This keeps `futures_foundation/chronos/data.py::load_long(timeframe='1day')`
and `walk_forward_folds` working **unmodified**.

### Phase B bootstrap (yfinance)

`yfinance` download, `auto_adjust=True` (split+dividend adjusted), 10y daily,
written to the schema above. yfinance is a bootstrap-only dev dependency —
do **not** add it to `requirements.txt` (owner installs ad hoc); the
production path replaces it in Phase C.

### Phase C production (IBKR)

`ibkr/build_equity_daily.py`: `ib_insync.reqHistoricalData` with
`whatToShow='ADJUSTED_LAST'`, `barSizeSetting='1 day'`, chunked +
pacing-safe. **Adjusted bars are mandatory** — an unadjusted 4:1 split
mid-flag masquerades as a catastrophic invalidation. Adjustment is a
backtest-integrity concern only; live strike/expiry resolution always uses
present-day unadjusted prices.

### QA — `audit_equity_data_quality.py`

Mirror `audit_data_quality.py`'s structure (`load` / per-instrument `audit` /
`main`), adapted:

- **Keep:** OHLC validity, duplicate timestamps, close-to-close spike scan,
  wick-blowout scan (equity thresholds: spike flag at >12% daily c2c for
  single names, >6% for SPY/QQQ/IWM — flags for review, not auto-repair).
- **Drop:** contract-bleed screens (no rolls in equities).
- **Add:** adjustment-consistency check (large overnight gap without a known
  split/dividend/earnings → flag) and an earnings-gap tagger (§2.5).

---

## 2. Pattern Definition [P2 · bull_flag.py]

All detection is **causal**: a pattern state is only known at the close of
the bar that confirms it. Pivots come from
`futures_foundation/primitives/detection.py::detect_pivots` with
**pivot period = 3** (a swing high/low needs 3 lower/higher bars each side →
confirmed 3 bars after the extreme; detection accounts for this lag).

ATR = Wilder ATR(14) on daily bars (reuse `_wilder_atr` pattern from
`supertrend_chronos.py`).

### 2.1 Pole

Anchored between a confirmed swing low (start) and the subsequent swing high
(end). Gates — ALL must hold:

| # | Gate | Locked value |
|---|---|---|
| P1 | Duration | 3 ≤ bars ≤ 10 |
| P2 | Height | `(high_end − low_start) ≥ 2.5 × ATR14` at pole start |
| P3 | Volume | mean pole volume ≥ 1.3 × trailing 20-day mean volume (as of pole start) |
| P4 | Efficiency ratio | `|c_end − c_start| / Σ|Δc|` over pole ≥ 0.50 (chop rejection; math per `regime.py::context_observations`) |

`pole_height = high_end − low_start`. Height in ATR units, ER, volume ratio,
and slope (linreg of closes) are all carried as features — gates use the
permissive end of the literature range; the model learns "more is better."

### 2.2 Flag

Scan forward from pole end. Running channel: `flag_high = max(high)`,
`flag_low = min(low)` since flag start. Gates:

| # | Gate | Locked value |
|---|---|---|
| F1 | Duration | 5 ≤ bars ≤ 20 (past 20 without breakout → **expired**, not invalidated) |
| F2 | Retracement | `(high_end − flag_low) / pole_height ≤ 0.618` (breach → candidate continues only while intact; close below flag_low → **invalidated**, no re-arm) |
| F3 | Volume decline | mean flag volume ≤ 0.85 × mean pole volume |
| F4 | Slope | linreg slope of flag closes within `[−0.30, +0.20] × pole_slope` (flat-to-mildly-down; steeper down = reversal, steeper up = continuation not consolidation) |

Range-contraction (last-third mean true range ÷ first-third mean true range)
is a **feature only**, not a gate — too noisy on 5–20 daily bars to hard-gate.

38.2–50% retracement ("textbook-clean" zone) is a feature, not a gate.

### 2.3 Breakout (= entry event)

First bar, while flag is intact, satisfying ALL:

| # | Gate | Locked value |
|---|---|---|
| B1 | Price | close strictly > flag_high (running max at prior bar) |
| B2 | Volume | breakout volume ≥ 1.2 × mean flag volume |
| B3 | Close strength | `(close − low)/(high − low) ≥ 0.75` (zero-range bar → 1.0) |

**Trade geometry (locked):**

```
entry  = breakout bar close
stop   = flag_low
risk   = entry − stop            (must be > 0, else skip — mirrors supertrend_chronos guard)
target = entry + pole_height     (100% measured-move, Bulkowski measure rule)
```

### 2.4 Invalidation summary

- Close < flag_low before breakout → pattern dead, no re-arm.
- Flag > 20 bars without breakout → expired (separate outcome class for
  diagnostics; neither traded nor counted as a loss).

### 2.5 Earnings-gap flag

`is_earnings_gap = True` if any overnight gap `|open/prev_close − 1| ≥ 2.5%`
occurs **within the pole window**. Phase B uses this gap proxy; Phase C
refines it by cross-referencing a real earnings calendar (IBKR fundamentals
or a static earnings-date file). Carried as a feature per user decision —
never a hard filter.

### 2.6 `scan_bull_flags` output

One row per pattern instance (breakout, invalidated, or expired), columns:
ticker, pole start/end/height/ATR-mult/ER/vol-ratio/slope, flag
start/duration/retrace/vol-decline/slope-ratio/range-contraction,
breakout bar/entry/stop/target/vol-ratio/close-strength, `is_earnings_gap`,
outcome class. Stateful bar-by-bar scan (deque technique per
`detection.py::detect_cisd_signals`) — never uses future bars.

---

## 3. Trade Resolution & Mechanical Backtest [P2 · bull_flag_mechanical.py]

Walk forward from entry bar, mirroring `supertrend_chronos.py::_resolve`
(stop-first tie rule), with one **equity-specific difference — honest gap
fills** (daily equity bars gap; the futures fill-at-exact-stop convention
would understate losses):

```
per bar j = entry+1 … entry+VERT_DAYS:
  if open_j <= stop:   exit at open_j        (gap through stop — honest fill)
  elif low_j <= stop:  exit at stop
  elif open_j >= target: exit at open_j      (favorable gap)
  elif high_j >= target: exit at target
  # same-bar stop+target touch -> stop first (conservative, repo convention)
timeout: exit at close of bar entry+VERT_DAYS
```

- `VERT_DAYS = 60` trading bars (≈3 calendar months).
- `COST_R = 0.02` per trade (spread+commission on liquid names, in R units).
- Realized R = `(exit − entry) / risk − COST_R`.

**Exit variants** (env `FFM_EXIT`, repo convention):
- `fixed` (baseline): stop/target/timeout as above.
- `ratchet`: stop = flag_low until peak favorable excursion ≥ **+1.5R**, then
  stop locks to `peak_R − 0.75R`, ratcheting up. ⚠️ UNVERIFIED: ratchet
  activation/lock values are futures-derived; tune against `fixed` in
  Phase B diagnostics before trusting.

**Report** (full stat block, repo standard): total events by outcome class,
trades, WR, meanR, sumR, PF, max consecutive losses, per-ticker table,
per-year table, earnings-gap vs clean-pole split.

**Phase B gate (pre-registered):** mechanical (take-every-breakout) backtest
must show `meanR > 0` after cost and `PF ≥ 1.15` pooled across the universe,
with no single ticker or year contributing >50% of sumR. Below that → the
pattern spec (§2) gets revised or the project stops — no ML rescue of a dead
mechanical edge.

---

## 4. ML Layer — `BullFlagBreakoutLabeler` [P4 · bull_flag_chronos.py]

`StrategyLabeler` protocol, structural mirror of `SuperTrendChronos`:

- `n_classes = 2` (take/skip).
- `calendar()` — long `[item_id, timestamp, target]`, one row per daily bar
  per ticker.
- `build(lo, hi, test_start)` — events = confirmed breakouts from
  `scan_bull_flags`; context = **128 daily log-close bars** ending at the
  breakout bar (`foundation.CTX`, unmodified — ≈6 months of pre-pattern
  context); purge any event whose `entry + VERT_DAYS` reaches ≥ `test_start`;
  label = 1 if target hit before stop within VERT_DAYS (fixed variant), else
  0. Keys = `(ticker, bar_index, …)` — `key[0]` must stay the ticker for
  `evaluate.py::_taken_tickers` to work unmodified.
- `features(keys)` — bespoke vector (float32): pole_atr_mult, pole_er,
  pole_vol_ratio, pole_slope_norm, flag_duration, flag_retrace,
  flag_vol_decline, flag_slope_ratio, flag_range_contraction,
  breakout_vol_ratio, breakout_close_strength, is_earnings_gap,
  target_R (= pole_height/risk), rv20 (20d realized vol), rv_percentile_1y.
- `evaluate(keys, preds)` — §3 resolution, R including COST_R.
- `config_dict()` — every §2/§3 constant, for bundle provenance.

**Fold windows:** `evaluate.run(labeler, …)` with `train_m=24, test_m=6`
(daily-bar events are sparse; the futures 3/1 default starves the
≥50-events-per-fold minimum). ⚠️ UNVERIFIED: event counts per fold — verify
in Phase D; widen further or extend universe if folds starve.

**Phase D gate:** the existing pre-registered REAL/SHUFFLE/RANDOM/NAIVE
honest-ruler verdict in `evaluate.py` must PASS, unmodified. FAIL → back to
§2/§4 features, never forward to options.

---

## 5. Options Structuring [P6]

### 5.1 IV proxy (backtest AND initial live — consistency over accuracy)

```
rv20   = std(log daily returns, 20d) × √252
sigma  = 1.20 × rv20                      # VRP multiplier, locked
iv_pct = percentile rank of rv20 vs its own trailing 252d distribution
r      = 0.04 flat                        # risk-free, locked
q      = per-ticker constant dividend yield (SPY .013, QQQ .006, IWM .011,
         single names 0 except AAPL/MSFT/AVGO-class ≈ .005)
```

Real IBKR chain IV is an explicitly-flagged **later upgrade** — switching it
on creates backtest/live divergence and requires its own validation pass.

### 5.2 Structure selection (locked rule)

- `iv_pct < 40` → **long call**.
- `iv_pct ≥ 40` → **call debit spread**, short leg = strike nearest `target`.

### 5.3 Strike / DTE / size

- Long leg: strike whose BS delta is nearest **0.65**.
- `DTE_target = clip(1.5 × median_calendar_days_to_target, 21, 60)` — median
  from Phase D's realized bars-to-target distribution of winners (empirical
  input, computed in Phase E, not hardcoded).
- Expiry: listed expiry nearest DTE_target.
- Size: premium at risk ≤ **1% of account equity** per trade (spread: net
  debit ≤ 1%).

### 5.4 Exits (two independent triggers, first to fire wins)

1. **Underlying trigger** — the §3 stop/target/ratchet levels on the
   underlying; when crossed, close the options position (backtest: reprice
   via BS at that bar's underlying close + decayed T; live: §7 polling).
2. **Premium hard stop** — mark ≤ **50% of premium paid** → close,
   regardless of underlying. Theta-bleed/IV-crush safety net with no
   underlying-price analogue.
3. Expiry approach: if still open at **DTE ≤ 5**, close at market (no
   assignment/pin risk into expiry week).

### 5.5 Backtest [P6 · options_backtest.py]

Replay Phase E's taken signals through BS: entry premium at breakout close
(sigma from §5.1), daily mark-to-market on underlying closes with T decaying,
exit per §5.4. Costs: options spread modeled at **1.5% of premium** per side
+ $0.65/contract commission. ⚠️ UNVERIFIED: spread-cost model is an
approximation; validate against a sample of real IBKR quotes in Phase H
paper trading.

**Phase F gate:** options-P&L expectancy remains positive and retains ≥ 60%
of the underlying-R edge's profit factor. Worse → revisit structure selection
(§5.2) / DTE / delta before touching live code.

---

## 6. Risk Guard [P7 · risk_guards_options.py]

Local module (user decision — independent of the external futures-bot guard).
Env-var configurable, all defaults locked here:

| Rule | Default |
|---|---|
| Max concurrent positions | 3 |
| Max premium at risk, aggregate | 3% of equity |
| Per-trade premium | ≤ 1% of equity (§5.3) |
| Consecutive-loss pause | 3 consecutive losing trades → no new entries 5 trading days |
| Premium hard stop | −50% (§5.4, live-enforced here) |
| Correlation cap | max 2 concurrent positions whose underlyings have 60d return correlation > 0.8 |

`analyze_options_concentration.py`: post-hoc diagnostic — concurrent-position
clusters, correlated drawdown days (do multiple positions lose on the same
macro vol event), monthly P&L concentration. (The futures
`analyze_daily_safeguards.py` session-day machinery does not apply to
multi-week swing holds — this replaces it.)

---

## 7. Live/Paper Execution [P8 · ibkr_options_broker.py]

`IBKROptionsBroker` — **sibling of**, not subclass of,
`chronos_live_bot.py::IBKRBroker` (that class is coupled to `ContFuture` /
point-value assumptions). Reuses `IBKRBroker._is_paper` directly; **hard
stop at startup on any non-paper account** until every phase gate has passed
and the user explicitly reopens.

- `resolve_option_contract(underlying, dte_target, delta_target, right)` —
  `Stock` + `reqSecDefOptParams` → expiry nearest DTE, strike by BS-delta
  inversion (§5.1 sigma) → `Option(...)` + `qualifyContracts`.
- Entry/exit orders: **marketable limit at midpoint**, re-peg once after 30s,
  then cross the spread (options spreads are wide; raw market orders leak).
- `recent_bars(n)` — daily bars for the **underlying**, exact §1 schema (the
  scanner must see live what it was backtested on).
- `OptionsRatchetManager` — mirror of `RatchetManager`'s state machine
  (`on_bar_close(high, low) -> stop_px`) on **underlying** daily closes.
- Two independent exit polls per §5.4: underlying level check (daily close)
  + option mark check via `reqMktData` on the held contract (intraday, the
  premium stop must not wait for EOD).
- Loop cadence: pattern scan + entries at daily close only; premium-stop
  poll intraday every 15 min during RTH.
- `test_ibkr_options_broker.py` first (offline, zero TWS): paper guard,
  strike/expiry selection as pure functions vs a synthetic chain, order
  construction. Then `--smoke-ibkr-options` (read-only connectivity), then
  paper loop.

---

## 8. Dependencies

Add to `requirements.txt` (Phase F): `scipy>=1.10` (currently only a
transitive dep via scikit-learn — pin it explicitly when `black_scholes.py`
lands). `ib_insync` stays import-guarded/optional (repo convention).
`yfinance` = ad-hoc dev install only, never in requirements. No `py_vollib`,
no `QuantLib`. Do not install anything without owner approval.

---

## 9. Phase Gates (each ships one validatable artifact)

| Phase | Artifact | Gate to pass |
|---|---|---|
| **A** | this spec | **user approval** ← current |
| **B** | `bull_flag.py` + `bull_flag_mechanical.py` on yfinance data | meanR > 0 after cost, PF ≥ 1.15 pooled, no >50% single-ticker/year concentration (§3) |
| **C** | `ibkr/build_equity_daily.py` + `audit_equity_data_quality.py` | Phase B result reproduces on IBKR data (PF within ±0.1) |
| **D** | `bull_flag_chronos.py` | honest-ruler PASS, unmodified `evaluate.py` (§4) |
| **E** | production bundle + threshold sweep | threshold picked from sweep table (calibrate_threshold pattern) |
| **F** | `black_scholes.py` + `options_backtest.py` | positive expectancy, ≥60% of underlying PF retained (§5.5) |
| **G** | `risk_guards_options.py` + concentration diagnostic | rules enforced in replay without destroying expectancy |
| **H** | broker + offline tests → smoke → paper loop | offline tests green → smoke green → 20+ paper trades matching backtest assumptions |

**Definition of Done, per phase:** runs and produces output · checked against
this spec · reversible (phases are additive; behavior flags default off) ·
`> NEXT` pointer in this doc updated · stat blocks never abbreviated.

## 10. Open items / ⚠️ UNVERIFIED register

1. ⚠️ Ratchet activation (+1.5R) / lock (−0.75R) values — futures-derived,
   tune in Phase B.
2. ⚠️ Fold windows 24/6 — verify ≥50 events/fold in Phase D.
3. ⚠️ Options spread-cost model (1.5%/side) — validate vs real IBKR quotes in
   Phase H paper.
4. ⚠️ VRP multiplier 1.20 and flat r=4% — sensitivity-check in Phase F
   (rerun at 1.0/1.4 and r=3%/5%; edge must not flip sign).
5. Earnings-gap proxy (2.5% overnight gap) → replace with real earnings
   calendar in Phase C.
