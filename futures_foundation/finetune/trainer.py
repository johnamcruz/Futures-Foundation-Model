"""
Labeling, reporting, and realized-R economics for strategy evaluation.

The torch walk-forward trainer was retired with the from-scratch FFM
backbone (see git tag `ffm-transformer-final` for the last full version).
Walk-forward training now lives in `futures_foundation.pipeline` (frozen Chronos-Bolt
embeddings + XGBoost selection head). What survives here are the torch-free
layers that remain useful to any strategy pipeline:

    Labeling      — run_labeling(), _labeling_cache_hash(),
                    _validate_labeler_output()
    Reporting     — print_eval_summary(), print_fold_progression(),
                    summarize_fold_precision(), _print_test_threshold_table(),
                    _print_confidence_calibration(), _print_realized_econ()
    Realized-R    — _compute_realized_r(), _realized_r_eval(), _r_stats(),
    economics       _econ_combined_objective(), _val_econ_objective()

IMPORT CONTRACT: this module must stay torch-free — parents that consume
Chronos embeddings run XGBoost, and torch+xgboost segfault in one process
on macOS (libomp collision).
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .base import StrategyLabeler

# ── Labeling ─────────────────────────────────────────────────────────────────

def _validate_labeler_output(
    strategy_feats: 'pd.DataFrame',
    labels_df: 'pd.DataFrame',
    feature_cols: list,
    expected_len: int,
    ticker: str,
) -> None:
    """Sanity-check labeler.run() output before writing to parquet cache."""
    problems = []

    if len(strategy_feats) != expected_len:
        problems.append(
            f'strategy_features has {len(strategy_feats)} rows but '
            f'ffm_df has {expected_len} rows — must be aligned to ffm_df.index')

    if len(labels_df) != expected_len:
        problems.append(
            f'labels_df has {len(labels_df)} rows but '
            f'ffm_df has {expected_len} rows — must be aligned to ffm_df.index')

    missing_feat_cols = [c for c in feature_cols if c not in strategy_feats.columns]
    if missing_feat_cols:
        problems.append(
            f'strategy_features is missing columns: {missing_feat_cols}\n'
            f'   feature_cols declares {feature_cols}')

    for col in ('signal_label', 'max_rr'):
        if col not in labels_df.columns:
            problems.append(f"labels_df is missing required column '{col}'")

    if 'signal_label' in labels_df.columns:
        n_signals = (labels_df['signal_label'] > 0).sum()
        if n_signals == 0:
            problems.append(
                f'labels_df has 0 signals — check strategy logic or data range')

    if problems:
        raise ValueError(
            f'\n\n❌ Labeler output validation failed for {ticker}:\n' +
            '\n'.join(f'  • {p}' for p in problems))


def _labeling_cache_hash(labeler: StrategyLabeler, tickers: list, timeframe: str) -> str:
    """Stable MD5 of all parameters that affect labeling output."""
    payload = {
        **labeler.config_dict(),
        'name':         labeler.name,
        'feature_cols': list(labeler.feature_cols),
        'tickers':      sorted(tickers),
        'timeframe':    timeframe,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run_labeling(
    labeler: StrategyLabeler,
    tickers: list,
    raw_dir: str,
    ffm_dir: str,
    cache_dir: str,
    micro_to_full: dict = None,
    force: bool = False,
    timeframe: str = '5min',
    use_cache: bool = False,
) -> None:
    """
    For each ticker: load raw CSV + FFM parquet, call labeler.run(),
    save strategy_features and labels to cache_dir.

    When use_cache=True the function hashes labeler.config_dict() + tickers +
    feature_cols and writes the hash to {cache_dir}/labeling_hash.txt.  A
    subsequent call with identical parameters and existing parquet files is
    skipped entirely (cache hit).  Any parameter change invalidates the cache
    and triggers a full re-label after wiping cache_dir.

    When use_cache=False (default) the function skips individual tickers whose
    parquet files already exist, matching the original per-ticker behaviour.

    Raw data is expected at {raw_dir}/{data_ticker}_{timeframe}.csv.
    FFM features at {ffm_dir}/{data_ticker}_features.parquet.
    """
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        current_hash   = _labeling_cache_hash(labeler, tickers, timeframe)
        hash_file      = os.path.join(cache_dir, 'labeling_hash.txt')
        parquet_files  = [
            os.path.join(cache_dir, f'{t}_strategy_labels.parquet') for t in tickers
        ]
        cache_valid = (
            not force
            and os.path.exists(hash_file)
            and open(hash_file).read().strip() == current_hash
            and all(os.path.exists(f) for f in parquet_files)
        )
        if cache_valid:
            print(f'⚡ Labeling cache hit (hash={current_hash}) — skipping re-label')
            return
        if os.path.exists(hash_file):
            print(f'♻️  Labeling config changed — re-labeling (hash={current_hash})')
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir, exist_ok=True)
    micro_to_full = micro_to_full or {}
    total_signals = total_bars = 0

    print(f"\n{'='*60}")
    print(f'  LABELING — {labeler.name.upper()} ({len(tickers)} tickers)')
    print(f"{'='*60}")

    for ticker in tickers:
        feat_path  = os.path.join(cache_dir, f'{ticker}_strategy_features.parquet')
        label_path = os.path.join(cache_dir, f'{ticker}_strategy_labels.parquet')

        ohlc_path = os.path.join(cache_dir, f'{ticker}_ohlc.parquet')

        if not force and os.path.exists(feat_path) and os.path.exists(label_path):
            cached = pd.read_parquet(label_path)
            sigs   = (cached['signal_label'] > 0).sum()
            note = ('' if os.path.exists(ohlc_path)
                    else '  (no _ohlc.parquet — borrow-#1 economic eval will '
                         'skip this ticker until a relabel with force=True)')
            print(f'  {ticker}: cached — {len(cached):,} bars, {sigs} signals'
                  f'{note}')
            total_signals += sigs
            total_bars    += len(cached)
            continue

        data_ticker   = micro_to_full.get(ticker, ticker)
        csv_path      = os.path.join(raw_dir, f'{data_ticker}_{timeframe}.csv')
        ffm_feat_path = os.path.join(ffm_dir,  f'{data_ticker}_features.parquet')

        if not os.path.exists(csv_path) or not os.path.exists(ffm_feat_path):
            print(f'  ⚠ Skip {ticker} — missing data'); continue

        print(f"\n{'─'*60}\n  {ticker}\n{'─'*60}")
        t0 = time.time()

        ffm_df = pd.read_parquet(ffm_feat_path)
        ffm_dt = pd.to_datetime(ffm_df['_datetime'])
        if ffm_dt.dt.tz is None:
            ffm_dt = ffm_dt.dt.tz_localize('UTC').tz_convert('America/New_York')
        ffm_df.index = ffm_dt

        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        if 'date' in df_raw.columns and 'datetime' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'date': 'datetime'})
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw.set_index('datetime', inplace=True)
        df_raw.sort_index(inplace=True)
        try:
            df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('America/New_York')
        except TypeError:
            if df_raw.index.tz is not None:
                df_raw.index = df_raw.index.tz_convert('America/New_York')
        print(f'  Loaded {len(df_raw):,} {timeframe} bars')

        strategy_feats, labels_df = labeler.run(df_raw, ffm_df, ticker)

        # Validate labeler output before saving to avoid corrupted cache files
        _validate_labeler_output(strategy_feats, labels_df, labeler.feature_cols,
                                 len(ffm_df), ticker)

        strategy_feats.to_parquet(feat_path,  index=False)

        # Borrow #1: cache the OHLC+ATR price path aligned 1:1 to ffm_df rows
        # (== labels_df rows) so the eval-stage realized-R backtest can slice
        # it identically to features/labels. atr = vty_atr_raw (kept metadata
        # in the prepared parquet; NOT a model feature, so absent from the
        # eval feature matrix — must be carried here for the trail).
        ohlc = df_raw[['open', 'high', 'low', 'close']].reindex(ffm_df.index)
        ohlc.insert(0, 'datetime', ffm_df['_datetime'].values)
        ohlc['atr'] = (ffm_df['vty_atr_raw'].values
                       if 'vty_atr_raw' in ffm_df.columns else np.nan)
        ohlc.reset_index(drop=True).to_parquet(ohlc_path, index=False)

        # Borrow #1 (b2): when the labeler emits an OPTIONAL per-row
        # `direction` column (>0 long / <0 short on signal rows), compute
        # realized-R under the uniform framework exit policy (entry =
        # next-bar open; exit = generic realized_r_trailing) and persist a
        # `realized_r` column alongside the labels. Back-compat: absent
        # `direction` → no `realized_r` column (eval skips w/ notice),
        # exactly mirroring the _ohlc.parquet pattern. This is eval/report
        # metadata only — the signal head stays pure binary and the risk
        # head still trains on max_rr (never on realized_r).
        if 'direction' in labels_df.columns:
            sig_pos = np.flatnonzero(labels_df['signal_label'].values > 0)
            is_long = labels_df['direction'].values[sig_pos] > 0
            sl_dist = (labels_df['sl_distance'].values[sig_pos]
                       if 'sl_distance' in labels_df.columns else None)
            rr = _compute_realized_r(
                ohlc['open'].values, ohlc['high'].values,
                ohlc['low'].values, ohlc['close'].values,
                ohlc['atr'].values, sig_pos, is_long, sl_dist)
            realized = np.full(len(labels_df), np.nan, dtype=np.float32)
            realized[sig_pos] = rr
            labels_df = labels_df.copy()
            labels_df['realized_r'] = realized

        labels_df.to_parquet(label_path, index=False)

        sigs = (labels_df['signal_label'] > 0).sum()
        total_signals += sigs
        total_bars    += len(labels_df)
        print(f'  ✓ {ticker}: {sigs} signals | ({time.time() - t0:.1f}s)')

    print(f"\n{'='*60}")
    print(f'  ✅ LABELING COMPLETE — {total_bars:,} bars | {total_signals} signals')
    print(f'  {"✅ density OK" if total_signals >= 500 else "⚠️  density LOW (<500)"}')
    print(f"{'='*60}")

    if use_cache:
        open(hash_file, 'w').write(current_hash)


# ── Per-fold reporting ───────────────────────────────────────────────────────

def _print_test_threshold_table(test_metrics: dict, fold_name: str, rr_target: float = 2.0) -> None:
    """Print per-threshold precision/recall/EV/AvgMaxRR table for a fold's test results."""
    if test_metrics is None:
        return
    n_test  = len(test_metrics['all_labels'])
    n_sig   = test_metrics['tp'] + test_metrics['fn']
    breakeven = 1.0 / (rr_target + 1.0)

    conf_arr = np.array(test_metrics['all_conf'])
    lab_arr  = np.array(test_metrics['all_labels'])
    pred_arr = np.array(test_metrics['all_preds'])
    rr_arr   = np.array(test_metrics['all_max_rr']) if test_metrics.get('all_max_rr') else None

    print(f'\n  {fold_name} test: {n_test:,} bars | {n_sig} actual signals')
    print(f'  {"Thresh":>6}  {"N":>6}  {"Correct":>7}  {"Prec":>6}  '
          f'{"EV@{:.0f}R".format(rr_target):>7}  {"Recall":>6}  {"Rate":>5}  {"AvgRR":>7}  Status')

    rows = []
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        m    = conf_arr >= thresh
        htp  = int(((pred_arr[m] > 0) & (lab_arr[m] > 0)).sum())
        hfp  = int(((pred_arr[m] > 0) & (lab_arr[m] == 0)).sum())
        n    = htp + hfp
        if n == 0:
            continue
        prec = htp / n
        ev   = prec * (rr_target + 1.0) - 1.0
        rec  = htp / max(n_sig, 1)
        rate = n / max(len(lab_arr), 1) * 100

        if rr_arr is not None:
            win_mask = m & (pred_arr > 0) & (lab_arr > 0)
            avg_rr   = float(rr_arr[win_mask].mean()) if win_mask.sum() > 0 else float('nan')
            rr_str   = f'{avg_rr:.2f}R' if not np.isnan(avg_rr) else '   —   '
        else:
            rr_str = '   —   '

        if ev > 0 and n >= 10:
            status = '✅ VIABLE'
        elif ev > 0 or (prec >= breakeven * 0.75 and n >= 5):
            status = '⚠️  MARGINAL'
        else:
            status = '❌'

        ev_str = f'{ev:+.2f}R'
        print(f'  {thresh:>6.2f}  {n:>6}  {htp:>7}  {prec:>5.1%}  '
              f'{ev_str:>7}  {rec:>5.1%}  {rate:>4.1f}%  {rr_str:>7}  {status}')
        rows.append((thresh, n, prec, ev))

    print(f'\n  EV@{rr_target:.0f}R = P×{rr_target+1:.0f} − 1  |  '
          f'Breakeven: P≥{breakeven:.1%}  |  '
          f'✅ EV>0 & N≥10  ⚠️  EV>0 or approaching  ❌ not viable')

    _print_realized_econ(conf_arr, pred_arr,
                         test_metrics.get('all_realized_r'), fold_name)


def _print_realized_econ(conf_arr, pred_arr, realized_arr, scope: str) -> None:
    """Borrow #1: realized-R economics, printed ALONGSIDE the MFE (max_rr)
    AvgRR so the two R's are never conflated. Filtered to predicted-positive
    rows with a resolved realized R (a real entry under the uniform framework
    exit policy). Back-compat: skipped with an explicit notice when no
    realized_r is available (labeler emitted no `direction` column, or the
    cache predates borrow #1 — relabel with force=True to enable)."""
    if realized_arr is None or len(realized_arr) == 0:
        return
    ra = np.asarray(realized_arr, float)
    pa = np.asarray(pred_arr)
    ca = np.asarray(conf_arr, float)
    if not np.isfinite(ra).any():
        print(f'\n  💰 Realized-R econ ({scope}): — skipped (no `realized_r`; '
              f'labeler emitted no `direction` column or cache predates '
              f'borrow #1 — relabel with force=True to enable)')
        return
    print(f'\n  💰 REALIZED-R ECONOMICS ({scope}) — realized exit, NOT MFE/max_rr')
    print(f'  {"Thresh":>6}  {"N":>5}  {"MeanR":>7}  {"WR":>6}  '
          f'{"PF":>6}  {"MaxDD":>8}  {"no-top1%":>9}')
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        m = (ca >= thresh) & (pa > 0)
        if m.sum() == 0:
            continue
        st = _r_stats(ra[m])
        if st['n'] == 0:
            continue
        pf    = st['profit_factor']
        pf_s  = f'{pf:>6.2f}' if pf < 9999 else '   ∞  '
        print(f'  {thresh:>6.2f}  {st["n"]:>5}  {st["mean_r"]:>+7.2f}  '
              f'{st["win_rate"]:>5.1%}  {pf_s}  {st["max_dd"]:>+8.2f}  '
              f'{st["no_top1"]:>+9.2f}')


def _print_confidence_calibration(test_metrics: dict) -> None:
    """Win-rate histogram by confidence band, filtered to predicted positives only.

    Monotonically rising win-rate = well-calibrated confidence.
    Non-monotonic = model is guessing at high confidence — investigate before deploying.
    """
    if test_metrics is None:
        return

    conf_arr = np.array(test_metrics['all_conf'])
    lab_arr  = np.array(test_metrics['all_labels'])
    pred_arr = np.array(test_metrics['all_preds'])
    pos_mask = pred_arr > 0

    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    rows  = []
    for lo, hi in bands:
        mask = pos_mask & (conf_arr >= lo) & (conf_arr < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        wins   = int((lab_arr[mask] > 0).sum())
        losses = n - wins
        wr     = wins / n
        rows.append((lo, hi, wins, losses, wr))

    if not rows:
        return

    win_rates = [r[4] for r in rows]
    monotonic = all(
        win_rates[i] <= win_rates[i + 1] + 0.02
        for i in range(len(win_rates) - 1)
    )

    print(f'\n  Confidence calibration (predicted positives):')
    print(f'  {"Band":>9}  {"Wins":>5}  {"Loss":>5}  {"WinRate":>8}')
    for lo, hi, wins, losses, wr in rows:
        bar    = '█' * int(wr * 16)
        deploy = ' ◄' if lo >= 0.80 else ''
        print(f'  {lo:.1f}–{hi:.1f}   {wins:>5}  {losses:>5}  {wr:>7.1%}  {bar}{deploy}')
    cal_str = '✅ monotonic' if monotonic else '⚠️  non-monotonic — check confidence calibration'
    print(f'  Calibration: {cal_str}')


# ── Evaluation summary ────────────────────────────────────────────────────────

def print_eval_summary(
    fold_results: dict,
    baseline_wr: dict = None,
    output_dir: str = None,
) -> None:
    """
    Print the full walk-forward evaluation table (confidence thresholds,
    per-fold breakdown, and learning verification vs mechanical baseline).
    Equivalent to Cell 5 in the strategy notebook.
    """
    all_conf_c = []; all_labels_c = []; all_preds_c = []; all_rr_c = []
    all_realized_c = []
    for fname, metrics in fold_results.items():
        if fname == '_model' or metrics is None:
            continue
        all_conf_c.extend(metrics['all_conf'])
        all_labels_c.extend(metrics['all_labels'])
        all_preds_c.extend(metrics['all_preds'])
        all_rr_c.extend(metrics['all_max_rr'])
        all_realized_c.extend(metrics.get('all_realized_r') or [])

    if not all_labels_c:
        print('No fold results available.')
        return

    all_conf   = np.array(all_conf_c)
    all_labels = np.array(all_labels_c)
    all_preds  = np.array(all_preds_c)
    all_rr     = np.array(all_rr_c)

    n_sig = (all_labels > 0).sum()
    print(f'\n📊 Combined ({len(all_labels):,} bars): {n_sig} signals | '
          f'{len(all_labels)-n_sig} noise')
    print(f'   Signal rate: {n_sig/len(all_labels)*100:.2f}%')

    print(f'\n🎯 CONFIDENCE THRESHOLDS')
    print('='*72)
    print(f'   {"Thresh":>6}  {"Trades":>6}  {"Correct":>7}  {"Prec":>6}  '
          f'{"Recall":>6}  {"AvgRR":>6}  {"PF":>7}  Verdict')
    print(f'  {"-"*66}')

    for thresh in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = all_conf >= thresh
        if mask.sum() == 0:
            print(f'  {thresh:.2f}{"":>12}  — no trades'); continue
        t_labels = all_labels[mask]; t_preds = all_preds[mask]; t_rr = all_rr[mask]
        called   = t_preds > 0
        if called.sum() == 0: continue
        tp = (called & (t_labels > 0)).sum()
        fp = (called & (t_labels == 0)).sum()
        fn = (~called & (t_labels > 0)).sum()
        trades  = int(tp + fp)
        prec    = tp / max(trades, 1)
        rec     = tp / max(tp + fn, 1)
        wins_rr = t_rr[called & (t_labels > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:>7.2f}' if pf < 9999 else '    ∞  '
        verdict = '✅ EDGE' if prec >= 0.40 and trades >= 5 else ('⚠️ LOW' if trades < 5 else '❌')
        print(f'  {thresh:.2f}  {trades:>6}  {int(tp):>7}  {prec:>6.3f}  {rec:>6.3f}  '
              f'{avg_rr:>6.2f}  {pf_str}  {verdict}')

    print(f'\n{"="*72}')
    print(f'  📊 PER-FOLD BREAKDOWN (conf ≥ 0.90)')
    print(f'{"="*72}')
    for fname, metrics in fold_results.items():
        if fname == '_model': continue
        if metrics is None: print(f'  {fname}: no data'); continue
        ca = np.array(metrics['all_conf']); la = np.array(metrics['all_labels'])
        pa = np.array(metrics['all_preds']); ra = np.array(metrics['all_max_rr'])
        m  = ca >= 0.90; called = pa[m] > 0
        if called.sum() == 0: print(f'  {fname}: 0 trades at 0.90'); continue
        tp = (called & (la[m] > 0)).sum(); fp = (called & (la[m] == 0)).sum()
        wins_rr = ra[m][called & (la[m] > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:.2f}' if pf < 9999 else '∞'
        pl_r    = wins_rr.sum() - fp
        print(f'  {fname}: {int(tp+fp)} trades | Prec:{tp/max(tp+fp,1):.3f} | '
              f'AvgRR:{avg_rr:.2f} | PF:{pf_str} | {pl_r:+.1f}R')

    _print_realized_econ(all_conf, all_preds, all_realized_c, 'COMBINED')

    if baseline_wr:
        baseline_avg = np.mean(list(baseline_wr.values()))
        print(f'\n{"="*72}')
        print(f'  🧠 LEARNING VERIFICATION (vs mechanical baseline)')
        print(f'{"="*72}')
        print(f'  Mechanical baseline avg: {baseline_avg*100:.1f}%')
        print(f'  {"Thresh":>6}  {"Trades":>6}  {"Prec":>6}  {"vs Base":>8}  Verdict')
        print(f'  {"-"*48}')
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            mask   = all_conf >= thresh
            called = all_preds[mask] > 0
            if called.sum() < 3: continue
            tp_n = (called & (all_labels[mask] > 0)).sum()
            fp_n = (called & (all_labels[mask] == 0)).sum()
            prec  = tp_n / max(tp_n + fp_n, 1)
            delta = prec - baseline_avg
            ok    = '✅ LEARNING' if delta > 0 else '❌ BELOW'
            print(f'  {thresh:.2f}  {int(tp_n+fp_n):>6}  {prec:>6.3f}  {delta:>+8.1%}  {ok}')

    if output_dir:
        print(f'\n  Output dir: {output_dir}')


def print_fold_progression(
    fold_results: dict,
    ref: dict = None,
    ref_label: str = 'ref',
    gate2_desc: str = 'backbone compounding',
) -> None:
    """
    Print fold-to-fold P@80 learning progression table and Gate 2 check.

    Parameters
    ----------
    fold_results : dict
        Output of run_walk_forward.
    ref : dict, optional
        Prior-run P@80 per fold for comparison (e.g. {'F1': 0.527, 'F2': 0.577, ...}).
        Rows without a matching key show no comparison column.
    ref_label : str
        Label for the reference values shown in each row (e.g. 'v17', 'v18-5m').
    gate2_desc : str
        Short phrase completing "must improve F1→F3 for <gate2_desc> to be working".
    """
    fold_order = ['F1', 'F2', 'F3', 'F4', 'F5']
    prev_p80   = None
    f1_p80     = None
    f3_p80     = None

    print(f'\n{"="*60}')
    print('  FOLD-TO-FOLD LEARNING PROGRESSION')
    print(f'{"="*60}')
    print(f'  {"Fold":<6}  {"P@80":>6}  {"N@80":>5}  {"Delta":>7}  Status')
    print(f'  {"-"*50}')

    for fn in fold_order:
        m = fold_results.get(fn)
        if m is None:
            print(f'  {fn:<6}  {"—":>6}')
            continue
        conf  = np.array(m['all_conf'])
        lab   = np.array(m['all_labels'])
        preds = np.array(m.get('all_preds', []))
        if len(preds) == len(conf):
            mask = (conf >= 0.80) & (preds > 0)
        else:
            mask = conf >= 0.80
        n    = int(mask.sum())
        p80  = float((lab[mask] > 0).mean()) if n > 0 else 0.0

        delta_str = '—'
        status    = ''
        if prev_p80 is not None:
            delta     = p80 - prev_p80
            delta_str = f'{delta:+.1%}'
            status    = '✅' if delta > 0 else ('➡' if abs(delta) < 0.02 else '❌')

        ref_str = ''
        if ref is not None:
            ref_val = ref.get(fn)
            ref_str = f'  vs {ref_label}:{ref_val:.0%}' if ref_val is not None else ''

        print(f'  {fn:<6}  {p80:>6.1%}  {n:>5}  {delta_str:>7}  {status}{ref_str}')

        if fn == 'F1': f1_p80 = p80
        if fn == 'F3': f3_p80 = p80
        prev_p80 = p80

    print(f'{"="*60}')
    print(f'  Gate 2: P@80 must improve F1→F3 for {gate2_desc} to be working')
    if f1_p80 is not None and f3_p80 is not None:
        gate2 = '✅ PASS' if f3_p80 > f1_p80 else '❌ FAIL'
        print(f'  Gate 2: F1={f1_p80:.1%} → F3={f3_p80:.1%}  {gate2}')
    print(f'{"="*60}')


def summarize_fold_precision(fold_results: dict) -> dict:
    """
    Return per-fold precision at standard confidence thresholds.

    Parameters
    ----------
    fold_results : dict
        Output of run_walk_forward — keys are fold names, values contain
        'all_conf' and 'all_labels' arrays.

    Returns
    -------
    dict
        {fold_name: {'signals': int, 'prec_at_70': float|None,
                     'prec_at_80': float|None, 'prec_at_90': float|None}}
    """
    summary = {}
    for fname, metrics in fold_results.items():
        if fname == '_model' or metrics is None:
            continue
        confs  = np.array(metrics['all_conf'])
        labels = np.array(metrics['all_labels'])
        preds  = np.array(metrics.get('all_preds', []))
        entry: dict = {'signals': int((labels > 0).sum())}
        for key, thr in [('prec_at_70', 0.70), ('prec_at_80', 0.80), ('prec_at_90', 0.90)]:
            if len(preds) == len(confs):
                mask = (confs >= thr) & (preds > 0)
            else:
                mask = confs >= thr
            entry[key] = round(float((labels[mask] > 0).mean()), 3) if mask.sum() > 0 else None
        summary[fname] = entry
    return summary


# =============================================================================
# Borrow #1 — realized-R economic eval (reuses generic realized_r_trailing)
# =============================================================================

def _compute_realized_r(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                        trail_atr_k: float = 2.0, activate_r: float = 1.0,
                        max_hold: int = 130) -> np.ndarray:
    """Per-signal realized R under the uniform framework exit policy
    (entry = NEXT bar open after the signal; risk = labeler sl_distance,
    atr fallback; exit = generic unit-tested `realized_r_trailing`).

    Returns an array of len(sig_idx); entries that cannot resolve
    (i+1 out of range / non-finite entry / unusable risk) are np.nan so
    the caller can scatter values back to row positions or drop them."""
    from futures_foundation.primitives import realized_r_trailing
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    atr = np.asarray(atr, float)
    n = len(c)
    out = np.full(len(sig_idx), np.nan, dtype=float)
    for k, idx in enumerate(sig_idx):
        i = int(idx)
        if i + 1 >= n:
            continue
        a = atr[i + 1]
        risk = float(sl_dist[k]) if sl_dist is not None else float('nan')
        if not np.isfinite(risk) or risk <= 0:          # atr fallback
            if not np.isfinite(a) or a <= 0:
                continue
            risk = a
        entry = o[i + 1]
        if not np.isfinite(entry):
            continue
        long_ = bool(is_long[k])
        sl_price = entry - risk if long_ else entry + risk
        res = realized_r_trailing(
            h, l, c, entry_idx=i + 1, is_long=long_, entry_price=entry,
            sl_price=sl_price, atr=(a if np.isfinite(a) and a > 0 else risk),
            trail_atr_k=trail_atr_k, activate_r=activate_r, max_hold=max_hold)
        out[k] = res['realized_r']
    return out


def _r_stats(rs) -> dict:
    """Aggregate realized-R economic stats. {n, mean_r, win_rate,
    profit_factor (ΣR+/|ΣR-|), max_dd (cumulative-R equity), no_top1
    (mean with top 1% removed = tail-fragility)}. NaNs are dropped."""
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return {'n': 0, 'mean_r': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'max_dd': 0.0, 'no_top1': 0.0}
    gp = float(rs[rs > 0].sum()); gl = float(-rs[rs < 0].sum())
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    notop = np.sort(rs)[:max(1, int(len(rs) * 0.99))]
    return {'n': int(len(rs)), 'mean_r': float(rs.mean()),
            'win_rate': float((rs > 0).mean()),
            'profit_factor': (gp / gl) if gl > 0 else float('inf'),
            'max_dd': float((eq - peak).min()),
            'no_top1': float(notop.mean())}


def _realized_r_eval(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                     trail_atr_k: float = 2.0, activate_r: float = 1.0,
                     max_hold: int = 130) -> dict:
    """Back-compat wrapper: per-signal realized R then aggregate stats.
    Equivalent to the original behavior (skipped signals dropped)."""
    rs = _compute_realized_r(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                             trail_atr_k, activate_r, max_hold)
    return _r_stats(rs)


def _econ_combined_objective(realized_r, min_trades: int = 5,
                             n_cap: int = 100, sortino_cap: float = 4.0) -> float:
    """Borrow #3: a CAGR·√Sortino-style PRODUCT score over a realized-R
    series (chronological, predicted-positive trades only — NaNs dropped).

    Why a product (not a sum / not precision): it cannot be gamed by the
    two degenerate collapses a precision/F1 objective rewards —
      • never-trade (recall→0): n=0 ⇒ score 0.
      • trade-everything (recall→1): the unselective R mix has poor downside
        ⇒ Sortino→~0 ⇒ score collapses toward 0.
    Only a *selective, positive-expectancy, low-downside* policy scores high.

    score = mean_r · √min(sortino, sortino_cap) · √(min(n,n_cap)/n_cap)
      sortino     = mean_r / (downside_dev + 1e-9)
      downside_dev= √mean(min(r,0)²)              (0-target Sortino)
    Sortino is CAPPED (default 4.0): a near-zero-downside val slice is
    almost always a small-N lucky artifact, and an uncapped ratio explodes
    on the ε term — the cap makes the score bounded and robust. Sparse n is
    √-down-weighted so a 2-trade fluke can't win. Returns 0.0 for empty /
    <min_trades / non-positive edge (clean, monotone, bounded — safe as a
    'higher is better' checkpoint selector)."""
    rs = np.asarray(realized_r, float)
    rs = rs[np.isfinite(rs)]
    if len(rs) < min_trades:
        return 0.0
    mean_r = float(rs.mean())
    if mean_r <= 0.0:
        return 0.0
    downside = float(np.sqrt(np.mean(np.minimum(rs, 0.0) ** 2)))
    sortino  = min(mean_r / (downside + 1e-9), sortino_cap)
    if sortino <= 0.0:
        return 0.0
    n_factor = min(len(rs), n_cap) / float(n_cap)
    return float(mean_r * np.sqrt(sortino) * np.sqrt(n_factor))


def _val_econ_objective(va: dict, thresh: float = 0.80) -> float:
    """Borrow #3: economic objective from a validation _evaluate() dict —
    realized R at predicted-positive rows with confidence ≥ thresh, scored
    by _econ_combined_objective. Returns NaN when borrow #1's realized_r is
    unavailable (labeler emitted no `direction`) so the caller can skip the
    econ tier and fall back to the proven _p80s priority (back-compat)."""
    rr = va.get('all_realized_r')
    if not rr:
        return float('nan')
    rr  = np.asarray(rr, float)
    if not np.isfinite(rr).any():
        return float('nan')
    conf = np.asarray(va['all_conf'], float)
    pred = np.asarray(va['all_preds'])
    m = (conf >= thresh) & (pred > 0)
    if m.sum() == 0:
        return 0.0
    return _econ_combined_objective(rr[m])
