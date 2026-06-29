"""Final-model training + held-out OOS — the 'produce' half (mirrors pipeline.produce).

Two-step, same as the Chronos process:
  1. wf.run(holdout_start='2026-01-01')  -> walk-forward honest ruler over PRE-2026
     folds (does the edge generalize?).  loop.train_loop wraps this with the
     overfit→Optuna→rerun guard.
  2. produce.train_final(holdout_start='2026-01-01') -> train the FINAL model on ALL
     data < 2026 (inner val + early-stop) and score the held-out 2026 OOS ONCE, with
     a SHUFFLE control.  2026 is touched exactly once, at the end.

Classifier-agnostic (Mantis/logistic/future backbones via the Classifier seam).
"""
import numpy as np
import pandas as pd

from .classifier import get_classifier
from .wf import (_pct_threshold, _arm_R, _meanR, _standardize_on_train,
                 OP_PERCENTILE, PASS_LIFT_MARGIN_R)


def train_final(labeler, classifier='mantis', clf_kwargs=None, holdout_start='2026-01-01',
                val_frac=0.15, seed=0, max_train=None, verbose=True):
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    cal = labeler.calendar()
    lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
    Ctr, Ytr, Ktr = labeler.build(lo, hs, hs)
    Cte, Yte, Kte = labeler.build(hs, hi + pd.Timedelta('1ns'), None)
    Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
    if len(Ytr) < 50 or len(Kte) < 20:
        raise ValueError(f"insufficient data: train={len(Ytr)} oos={len(Kte)}")
    # cap train BEFORE featurizing (lean parent)
    if max_train and len(Ktr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Ktr), max_train, replace=False)
        Ktr = [Ktr[j] for j in sub]; Ytr = Ytr[sub]
    # inner train/val carve for early-stop (val is NEVER the OOS)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Ktr)); rng.shuffle(idx)
    nv = max(10, int(len(idx) * val_frac))
    va_i, tr_i = idx[:nv], idx[nv:]
    Ktr_tr = [Ktr[j] for j in tr_i]; Ytr_tr = Ytr[tr_i]
    Ktr_va = [Ktr[j] for j in va_i]; Ytr_va = Ytr[va_i]

    Xtr = clf.featurize(labeler, Ktr_tr)
    Xval = clf.featurize(labeler, Ktr_va)
    Xte = clf.featurize(labeler, Kte)
    if clf.needs_standardize:
        Xtr, Xval, Xte = _standardize_on_train(Xtr, Xval, Xte)
    if verbose:
        print(f"=== PRODUCE ({classifier}: train < {holdout_start}, 2026 OOS) ===")
        print(f"  train={len(tr_i)} val={len(va_i)} oos={len(Kte)} X={tuple(Xtr.shape[1:])} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)

    p_val, p_te, ba = clf.fit_predict(Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    ysh = Ytr_tr.copy(); rng.shuffle(ysh)
    psv, ps, _ = clf.fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed)
    Rs = _arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))

    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = _meanR(R) - _meanR(Rs)
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R), shuffle_meanR=_meanR(Rs),
               edge_shuffle=edge, n_train=len(tr_i), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=bool(edge >= PASS_LIFT_MARGIN_R))
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
              f"edge {edge:+.3f} (trades={out['oos_trades']})")
        print(f"  -> {'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")
    return out
