"""XGBoost selection-head hyperparameter tuner (Optuna, anti-overfit bounds).

Strategy- and backbone-agnostic. Tunes ONLY the XGBHead params on top of the
frozen Chronos embedding + the labeler's features — it does NOT touch the
architecture, the signal generator, or any labeler. Use it to fill the
"XGB head tuning" lever: the pipeline otherwise ships fixed params.

Method (overfit-aware):
  1. Build the walk-forward folds for a labeler, embed ONCE (batched), fuse the
     labeler features — same construction the evaluator uses.
  2. Split folds in time: the first (1-holdout_frac) are the TUNE set, the last
     holdout_frac are an untouched GUARD set.
  3. Optuna (TPE + MedianPruner) searches the tight anti-overfit bounds, scoring
     each trial by POOLED realized meanR (labeler.evaluate) across the TUNE
     folds' OOS test slices — never on training rows.
  4. The winning params are then scored on the GUARD folds. If GUARD meanR
     collapses vs TUNE meanR, the tuning overfit — reported, not hidden.
  5. A default-params baseline is scored on the same folds, so the output answers
     one question directly: does tuning LIFT over the shipped defaults?

The winning params are returned (and printed as a head_factory snippet). NOTHING
is auto-applied — certify with evaluate.run(head_factory=...) on the full
walk-forward, and only then pass them to produce.train.

CLI:
    python -m futures_foundation.chronos.tune_head \
        --labeler colabs.kalman_nw_chronos2:KalmanNWChronos2 \
        --trials 80 --max-folds 14
"""
import argparse
import importlib

import numpy as np
import pandas as pd

from .data import walk_forward_folds
from futures_foundation import foundation as backbone
from . import context_fusion
from .head_xgb import XGBHead

# Tight anti-overfit search space (ported from the original XGBoost pipeline's
# spec-section-6 bounds — deliberately narrow; do NOT widen without re-checking
# the overfit guard).
BOUNDS = dict(
    max_depth=(3, 6), learning_rate=(0.01, 0.1),
    subsample=(0.6, 0.85), colsample_bytree=(0.7, 1.0),
    reg_lambda=(1.0, 10.0), min_child_weight=(5, 50),
    n_estimators=(200, 800),
)


def _suggest(trial):
    return dict(
        max_depth=trial.suggest_int('max_depth', *BOUNDS['max_depth']),
        learning_rate=trial.suggest_float('learning_rate',
                                          *BOUNDS['learning_rate'], log=True),
        subsample=trial.suggest_float('subsample', *BOUNDS['subsample']),
        colsample_bytree=trial.suggest_float('colsample_bytree',
                                             *BOUNDS['colsample_bytree']),
        reg_lambda=trial.suggest_float('reg_lambda', *BOUNDS['reg_lambda'],
                                       log=True),
        min_child_weight=trial.suggest_int('min_child_weight',
                                           *BOUNDS['min_child_weight']),
        n_estimators=trial.suggest_int('n_estimators', *BOUNDS['n_estimators'],
                                       step=50),
    )


def _build_folds(labeler, train_m, test_m, max_folds):
    """Build + batch-embed + fuse folds (mirrors evaluate.run phases 1-2)."""
    fold_data = []
    for fold, tr, te in walk_forward_folds(labeler.calendar(), train_m, test_m):
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
                              Ktr=Ktr, Cte=Cte, Kte=Kte))
    if not fold_data:
        raise SystemExit("No productive folds to tune on.")
    flat = []
    for d in fold_data:
        flat.extend(d['Ctr']); flat.extend(d['Cte'])
    print(f"[tune] batch-embed {len(flat):,} contexts across "
          f"{len(fold_data)} folds...")
    emb = backbone.embed(flat)
    feats_fn = getattr(labeler, 'features', None)
    o = 0
    for d in fold_data:
        ntr, nte = len(d['Ctr']), len(d['Cte'])
        Etr = emb[o:o + ntr]; o += ntr
        Ete = emb[o:o + nte]; o += nte
        xtr = (np.asarray(feats_fn(d['Ktr']), np.float32)
               if feats_fn is not None else None)
        xte = (np.asarray(feats_fn(d['Kte']), np.float32)
               if feats_fn is not None else None)
        # fuse (heads=None) — embed + labeler features, same as evaluate
        d['Xtr'] = context_fusion.fuse(Etr, xtr, None, 'both')
        d['Xte'] = context_fusion.fuse(Ete, xte, None, 'both')
        del d['Ctr'], d['Cte']                       # free raw contexts
    print(f"[tune] feat_dim={fold_data[0]['Xtr'].shape[1]}")
    return fold_data


def _pooled_meanR(labeler, folds, params, seed):
    """Fit XGBHead(**params) per fold, predict its OOS slice, pool realized R."""
    nc = labeler.n_classes
    Rs = []
    for d in folds:
        head = XGBHead(nc, **params).fit(d['Xtr'], d['Ytr'], seed)
        preds = head.predict(d['Xte'])
        R = labeler.evaluate(d['Kte'], preds)
        if len(R):
            Rs.append(R)
    if not Rs:
        return 0.0, 0
    R = np.concatenate(Rs)
    return float(R.mean()), int(len(R))


def tune_head(labeler, *, n_trials=80, seed=42, max_folds=14,
              holdout_frac=0.3, train_m=3, test_m=1):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    folds = _build_folds(labeler, train_m, test_m, max_folds)
    n_guard = max(1, int(round(len(folds) * holdout_frac)))
    tune_folds, guard_folds = folds[:-n_guard], folds[-n_guard:]
    print(f"[tune] {len(tune_folds)} tune folds, {len(guard_folds)} guard "
          f"(held-out) folds\n")

    def objective(trial):
        p = _suggest(trial)
        Rs = []
        for k, d in enumerate(tune_folds):
            head = XGBHead(labeler.n_classes, **p).fit(d['Xtr'], d['Ytr'], seed)
            R = labeler.evaluate(d['Kte'], head.predict(d['Xte']))
            if len(R):
                Rs.append(R)
            running = float(np.concatenate(Rs).mean()) if Rs else 0.0
            trial.report(running, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.concatenate(Rs).mean()) if Rs else -1.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    print(f"[tune] Optuna {n_trials} trials over anti-overfit bounds...\n")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    # default-params baseline vs tuned, on BOTH tune and guard folds
    base = {}
    b_tune, n_bt = _pooled_meanR(labeler, tune_folds, base, seed)
    b_guard, n_bg = _pooled_meanR(labeler, guard_folds, base, seed)
    t_tune, n_tt = _pooled_meanR(labeler, tune_folds, best, seed)
    t_guard, n_tg = _pooled_meanR(labeler, guard_folds, best, seed)

    bar = "=" * 64
    print(f"\n{bar}\n🔧 HEAD-TUNER RESULT\n{bar}")
    print(f"  {'arm':<16}{'TUNE meanR':>12}{'GUARD meanR':>13}")
    print(f"  {'default':<16}{b_tune:>+12.3f}{b_guard:>+13.3f}")
    print(f"  {'tuned':<16}{t_tune:>+12.3f}{t_guard:>+13.3f}")
    lift_guard = t_guard - b_guard
    print(f"\n  GUARD lift (tuned − default): {lift_guard:+.3f}R  "
          f"(this is the honest, held-out lift)")
    overfit = t_tune - t_guard
    if overfit > 0.15 and t_guard < b_guard:
        print(f"  ⚠ OVERFIT: tuned wins on TUNE (+{t_tune-b_tune:+.3f}) but "
              f"loses on GUARD — params overfit the tune folds. Keep defaults.")
    print(f"\n  best params: {best}")
    print(f"\n  → certify before use:")
    print(f"    ev.run(lab, head_factory=lambda nc: XGBHead(nc, "
          f"**{best}))")
    return dict(params=best, tune_meanR=t_tune, guard_meanR=t_guard,
                default_tune_meanR=b_tune, default_guard_meanR=b_guard,
                guard_lift=lift_guard)


def _load_labeler(spec):
    mod_name, cls_name = spec.split(':')
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labeler', required=True,
                    help='module:Class, e.g. colabs.kalman_nw_chronos2:'
                         'KalmanNWChronos2')
    ap.add_argument('--trials', type=int, default=80)
    ap.add_argument('--max-folds', type=int, default=14)
    ap.add_argument('--holdout-frac', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    backbone.stamp_active_source(context='head-tuner')
    lab = _load_labeler(args.labeler)
    tune_head(lab, n_trials=args.trials, seed=args.seed,
              max_folds=args.max_folds, holdout_frac=args.holdout_frac)


if __name__ == '__main__':
    main()
