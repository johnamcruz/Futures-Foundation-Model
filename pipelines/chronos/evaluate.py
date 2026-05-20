"""Honest-ruler evaluation — strategy- and head-agnostic.

Frozen Chronos embedding (+ optional strategy features) -> a pluggable
head, scored leak-free over walk-forward x {REAL, SHUFFLE, RANDOM} x seeds.
A number is believed ONLY if REAL clearly beats SHUFFLE and RANDOM on
realized R (cost lives inside the strategy's evaluate()). No strategy or
head specifics here — only the StrategyLabeler protocol and a head with
fit(X,y,seed)/predict(X).
"""
from collections import defaultdict

import numpy as np

from .data import walk_forward_folds
from . import backbone
from .head_xgb import XGBHead, XGBRiskHead


def _stats(R):
    R = np.asarray(R, float)
    if not len(R):
        return "trades=0"
    return (f"trades={len(R)} win={np.mean(R > 0):.1%} "
            f"sumR={R.sum():+.1f} meanR={R.mean():+.3f}")


def _featurize(labeler, contexts, keys):
    """Frozen Chronos embedding, fused with the strategy's own features if
    the labeler exposes an optional features(keys) hook."""
    X = backbone.embed(contexts)
    feats = getattr(labeler, 'features', None)
    if feats is not None:
        extra = np.asarray(feats(keys), np.float32)
        if extra.size:
            X = np.hstack([X, extra.reshape(len(X), -1)])
    return X


def _taken_tickers(keys, preds):
    """Per-trade ticker list aligned with the labeler.evaluate R array
    (convention: preds==0 = skip; else = a trade; key[0] = ticker by
    convention in concrete labelers). Returns [] if keys aren't tuples
    (e.g., synthetic test labelers using int keys) — gracefully skips
    the per-ticker breakdown for those."""
    taken = [keys[i] for i in np.flatnonzero(np.asarray(preds) != 0)]
    return [k[0] for k in taken if isinstance(k, tuple) and len(k)]


def _agg_stats(name, R, tks=None):
    """Aggregate stats line; optional per-ticker breakdown."""
    R = np.asarray(R, float)
    if not len(R):
        print(f"  [{name:<7}] trades=0")
        return
    print(f"  [{name:<7}] {_stats(R)}")
    if tks is None:
        return
    by_tk = defaultdict(list)
    for r, t in zip(R, tks):
        by_tk[t].append(r)
    for tk in sorted(by_tk):
        Rt = np.asarray(by_tk[tk])
        print(f"            {tk:<5} {_stats(Rt)}")


def run(labeler, head_factory=None, seeds=(0, 1, 2), train_m=3, test_m=1,
        max_folds=None):
    """labeler: a StrategyLabeler. head_factory: nc -> head (default
    XGBHead). max_folds=None -> sweep every available OOS month-pair
    (XGBoost-pipeline convention). Prints REAL/SHUFFLE/RANDOM per
    fold-seed, an aggregate stats block + per-ticker breakdown for
    REAL, AND (binary labelers only) a WR-by-confidence-threshold sweep
    — the entry-signal calibration test for downstream RL/account
    management. Returns the per-(fold,seed) records."""
    head_factory = head_factory or (lambda nc: XGBHead(nc))
    binary = labeler.n_classes == 2
    out = []
    pool = {'REAL': ([], []), 'SHUFFLE': ([], []), 'RANDOM': ([], [])}
    proba_records = []
    # Phase 1 — collect all (Ctr, Ytr, Ktr, Cte, Yte, Kte) across folds
    # WITHOUT calling backbone.embed. Pure pandas/numpy; fast.
    fold_data = []
    for fold, tr, te in walk_forward_folds(labeler.calendar(), train_m,
                                           test_m):
        if max_folds is not None and len(fold_data) >= max_folds:
            break
        ts0 = te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(),
                                       tr['timestamp'].max(), ts0)
        Cte, Yte, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50:
            continue
        fold_data.append(dict(fold=fold, Ctr=Ctr, Ytr=np.asarray(Ytr),
                              Ktr=Ktr, Cte=Cte, Yte=np.asarray(Yte),
                              Kte=Kte))
    if not fold_data:
        print("No productive folds — nothing to evaluate.")
        return out
    # Phase 2 — ONE batched Chronos embed across all productive folds.
    # Replaces ~2*N_folds subprocess loads with 1. Major speedup on long
    # walk-forwards; numpy/torch parallelize the single big inference call.
    flat_contexts = []
    for d in fold_data:
        flat_contexts.extend(d['Ctr']); flat_contexts.extend(d['Cte'])
    print(f"\n[batch-embed] {len(flat_contexts):,} contexts across "
          f"{len(fold_data)} folds in ONE Chronos call...")
    flat_embed = backbone.embed(flat_contexts)
    # slice back per fold + fuse with labeler features
    o = 0
    for d in fold_data:
        ntr, nte = len(d['Ctr']), len(d['Cte'])
        Etr = flat_embed[o:o + ntr]; o += ntr
        Ete = flat_embed[o:o + nte]; o += nte
        feats_fn = getattr(labeler, 'features', None)
        if feats_fn is not None:
            xtr_extra = np.asarray(feats_fn(d['Ktr']), np.float32)
            xte_extra = np.asarray(feats_fn(d['Kte']), np.float32)
            d['Xtr'] = (np.hstack([Etr, xtr_extra.reshape(len(Etr), -1)])
                        if xtr_extra.size else Etr)
            d['Xte'] = (np.hstack([Ete, xte_extra.reshape(len(Ete), -1)])
                        if xte_extra.size else Ete)
        else:
            d['Xtr'], d['Xte'] = Etr, Ete
    print(f"[batch-embed] done. feat_dim={fold_data[0]['Xtr'].shape[1]}\n")

    # Phase 3 — per-(fold, seed) XGBoost work, parallelized via threads.
    # XGBoost releases the GIL in its C tree-builder + numpy ops, so a
    # ThreadPoolExecutor extracts multi-core speedup without pickling the
    # labeler's large bar arrays (which a ProcessPool would force).
    # Risk-head training target: max_rr_realized embedded in keys (4th
    # element). Falls back to 0 if labeler keys are 3-tuples (legacy).
    def _max_rr_targets(keys):
        return np.asarray(
            [k[3] if len(k) > 3 else 0.0 for k in keys], np.float32)

    def _work(d, seed):
        Ytr, Xtr, Xte, Kte = d['Ytr'], d['Xtr'], d['Xte'], d['Kte']
        Yte_np = d['Yte']
        nc = labeler.n_classes
        # signal head — binary take/skip
        head = head_factory(nc).fit(Xtr, Ytr, seed)
        p_real = head.predict(Xte)
        R = labeler.evaluate(Kte, p_real)               # fixed-TP baseline
        proba = (head.predict_proba(Xte)[:, 1].astype(np.float32)
                 if binary else None)
        # risk head — predicts max_rr_realized per signal (binary only;
        # 3-class direction strategies don't have a single risk regression
        # target). Trained alongside on the same Xtr.
        risk_pred = None
        if binary:
            mrr_tr = _max_rr_targets(d['Ktr'])
            risk_head = XGBRiskHead().fit(Xtr, mrr_tr, seed)
            risk_pred = risk_head.predict(Xte)
        # shuffle + random controls (unchanged)
        ysh = Ytr.copy()
        np.random.default_rng(seed + 1).shuffle(ysh)
        p_sh = head_factory(nc).fit(Xtr, ysh, seed).predict(Xte)
        Rs = labeler.evaluate(Kte, p_sh)
        p_rd = np.random.default_rng(seed + 2).integers(0, nc, len(Kte))
        Rr = labeler.evaluate(Kte, p_rd)
        return dict(fold=d['fold'], seed=seed, R=R, Rs=Rs, Rr=Rr,
                    p_real=p_real, p_sh=p_sh, p_rd=p_rd,
                    proba=proba, risk_pred=risk_pred,
                    Yte=Yte_np, Kte=Kte)

    import concurrent.futures as cf
    import os
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"[parallel] running {len(fold_data) * len(seeds)} (fold, seed) "
          f"work units across {n_workers} threads...")
    work_units = [(d, s) for d in fold_data for s in seeds]
    results = [None] * len(work_units)
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_work, d, s): i
                   for i, (d, s) in enumerate(work_units)}
        for fut in cf.as_completed(futures):
            results[futures[fut]] = fut.result()

    # Print in fold/seed order + aggregate.
    by_fold = defaultdict(list)
    for r in results:
        by_fold[r['fold']].append(r)
    for fold in sorted(by_fold):
        d = next(x for x in fold_data if x['fold'] == fold)
        print(f"== fold {fold} | ntr={len(d['Ytr'])} nte={len(d['Kte'])} "
              f"| feat_dim={d['Xtr'].shape[1]} | classes={labeler.n_classes} ==")
        for r in sorted(by_fold[fold], key=lambda x: x['seed']):
            print(f"  seed {r['seed']}  [REAL   ] {_stats(r['R'])}")
            print(f"          [SHUFFLE] {_stats(r['Rs'])}")
            print(f"          [RANDOM ] {_stats(r['Rr'])}")
            out.append({'fold': r['fold'], 'seed': r['seed'],
                        'REAL': r['R'], 'SHUFFLE': r['Rs'],
                        'RANDOM': r['Rr']})
            pool['REAL'][0].append(r['R'])
            pool['REAL'][1].extend(_taken_tickers(r['Kte'], r['p_real']))
            pool['SHUFFLE'][0].append(r['Rs'])
            pool['SHUFFLE'][1].extend(_taken_tickers(r['Kte'], r['p_sh']))
            pool['RANDOM'][0].append(r['Rr'])
            pool['RANDOM'][1].extend(_taken_tickers(r['Kte'], r['p_rd']))
            if binary and r['proba'] is not None:
                proba_records.append(
                    (r['Kte'], r['proba'], r['Yte'].astype(np.int8),
                     r['risk_pred']))
    print(f"\n== AGGREGATE | folds={len({r['fold'] for r in out})} "
          f"seeds={len(seeds)} | pooled across all (fold,seed) ==")
    for name in ('REAL', 'SHUFFLE', 'RANDOM'):
        R = (np.concatenate(pool[name][0]) if pool[name][0]
             else np.array([], float))
        _agg_stats(name, R, tks=pool[name][1] if name == 'REAL' else None)
    print("\n-> Believe a result only if REAL clearly beats SHUFFLE AND "
          "RANDOM on sumR/meanR (cost in evaluate()), AND the per-ticker "
          "lift over the labeler's NAIVE baseline is consistent.")

    # ---- Confidence-threshold dashboard (binary labelers only) -----------
    # FFM/XGBoost-pipeline-style: per threshold, report trades / wins / WR
    # (= Prec) / Recall / PF / vs-NAIVE / verdict. The entry-signal
    # calibration test for downstream RL/account managers — pick the
    # threshold where WR is high enough AND trade frequency is meaningful.
    if binary and proba_records:
        all_y = np.concatenate([y for _, _, y, _ in proba_records])
        naive_wr = float(np.mean(all_y == 1)) if len(all_y) else 0.0
        total_pos = int((all_y == 1).sum())
        # risk-head MAE on confirmed signals (only ones we'd take) — the
        # quality of the dynamic-TP prediction.
        rp_have_risk = any(r is not None for _, _, _, r in proba_records)
        if rp_have_risk:
            r_pred_all = np.concatenate(
                [r for _, _, _, r in proba_records if r is not None])
            r_true_all = np.concatenate(
                [np.asarray([k[3] if len(k) > 3 else 0.0 for k in keys],
                            np.float32)
                 for keys, _, _, _ in proba_records])
            mae = float(np.mean(np.abs(r_pred_all - r_true_all)))
            corr = float(np.corrcoef(r_pred_all, r_true_all)[0, 1]) \
                if r_pred_all.std() > 0 and r_true_all.std() > 0 else 0.0
            print(f"\n📐 RISK-HEAD (predicts max_rr_realized per signal)")
            print(f"   MAE: {mae:.2f}R  |  Pearson r(pred, actual): "
                  f"{corr:+.3f}  |  pred mean: {r_pred_all.mean():.2f}R  |  "
                  f"actual mean: {r_true_all.mean():.2f}R")

        # ---- Fixed-TP confidence dashboard (signal-head only) ------------
        print(f"\n🎯 CONFIDENCE THRESHOLDS — REAL @ fixed-TP (pooled)")
        print(f"   {'Thresh':>6}  {'Trades':>6}  {'Wins':>5}  {'WR':>6}  "
              f"{'Recall':>6}  {'PF':>6}  {'vs NAIVE':>8}  Verdict")
        print(f"   {'-' * 66}")
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            R_all, tp_total, fp_total = [], 0, 0
            for keys, proba, yte, _r in proba_records:
                preds_t = (proba >= thr).astype(int)
                R_t = labeler.evaluate(keys, preds_t)
                if len(R_t):
                    R_all.append(R_t)
                tp_total += int(((preds_t == 1) & (yte == 1)).sum())
                fp_total += int(((preds_t == 1) & (yte == 0)).sum())
            trades = tp_total + fp_total
            if trades == 0:
                print(f"   {thr:>5.2f}  {0:>6}     —     —       —"
                      f"      —      —      ⚠️ no trades")
                continue
            Rc = np.concatenate(R_all)
            wr = tp_total / trades
            recall = (tp_total / total_pos) if total_pos else 0.0
            wins_R = Rc[Rc > 0]; losses_R = Rc[Rc < 0]
            pf = (wins_R.sum() / abs(losses_R.sum())
                  if len(losses_R) and losses_R.sum() < 0 else float('inf'))
            pf_s = f"{pf:>6.2f}" if pf < 999 else "   inf"
            delta = wr - naive_wr
            verdict = ('✅ EDGE' if wr >= 0.60 and trades >= 20
                       else ('⚠️ THIN' if trades < 20
                             else ('🟡 LIFT' if delta > 0.05 else '❌')))
            print(f"   {thr:>5.2f}  {trades:>6}  {tp_total:>5}  "
                  f"{100 * wr:>5.1f}%  {100 * recall:>5.1f}%  {pf_s}  "
                  f"{100 * delta:>+7.1f}%  {verdict}")

        # ---- Dynamic-TP confidence dashboard (signal + risk head) --------
        # The deployment-equivalent metric: per signal, TP is set from
        # the risk-head prediction. This is what the bot will earn.
        if rp_have_risk:
            print(f"\n💎 CONFIDENCE THRESHOLDS — REAL @ dynamic-TP "
                  f"(TP = clip(0.8 × R̂, 1.5, 8.0))")
            print(f"   {'Thresh':>6}  {'Trades':>6}  {'WR':>6}  "
                  f"{'AvgR':>6}  {'sumR':>7}  {'PF':>6}  Verdict")
            print(f"   {'-' * 56}")
            for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
                R_all = []
                for keys, proba, _yte, risk in proba_records:
                    if risk is None:
                        continue
                    preds_t = (proba >= thr).astype(int)
                    R_t = labeler.evaluate(keys, preds_t, risk_preds=risk)
                    if len(R_t):
                        R_all.append(R_t)
                if not R_all:
                    print(f"   {thr:>5.2f}  {0:>6}     —      —       —"
                          f"      —     ⚠️ no trades")
                    continue
                Rc = np.concatenate(R_all)
                trades = len(Rc)
                wr = float(np.mean(Rc > 0))
                meanR = float(Rc.mean())
                wins_R = Rc[Rc > 0]; losses_R = Rc[Rc < 0]
                pf = (wins_R.sum() / abs(losses_R.sum())
                      if len(losses_R) and losses_R.sum() < 0
                      else float('inf'))
                pf_s = f"{pf:>6.2f}" if pf < 999 else "   inf"
                # The combine-pass metric — both axes captured.
                verdict = ('✅ EDGE' if meanR >= 1.5 and trades >= 20
                           else ('⚠️ THIN' if trades < 20
                                 else ('🟡 LIFT' if meanR > 0.5 else '❌')))
                print(f"   {thr:>5.2f}  {trades:>6}  {100*wr:>5.1f}%  "
                      f"{meanR:+5.2f}R  {Rc.sum():+7.1f}  {pf_s}  {verdict}")

        print(f"\n   NAIVE-WR baseline (take every signal): "
              f"{100 * naive_wr:.1f}%")
        print(f"   Target: WR ≥ 60% AND meanR ≥ 1.5R at meaningful trade "
              f"frequency (combine-pass-ready entry signal).")
    return out
