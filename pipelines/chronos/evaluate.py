"""Honest-ruler evaluation — strategy- and head-agnostic.

Frozen Chronos embedding (+ optional strategy features) -> a pluggable
head, scored leak-free over walk-forward x {REAL, SHUFFLE, RANDOM} x seeds.
A number is believed ONLY if REAL clearly beats SHUFFLE and RANDOM on
realized R (cost lives inside the strategy's evaluate()). No strategy or
head specifics here — only the StrategyLabeler protocol and a head with
fit(X,y,seed)/predict(X).
"""
import numpy as np

from .data import walk_forward_folds
from . import backbone
from .head_xgb import XGBHead


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


def run(labeler, head_factory=None, seeds=(0, 1, 2), train_m=3, test_m=1,
        max_folds=1):
    """labeler: a StrategyLabeler. head_factory: nc -> head (default
    XGBHead). Prints REAL/SHUFFLE/RANDOM per fold-seed; returns the
    per-(fold,seed) R arrays."""
    head_factory = head_factory or (lambda nc: XGBHead(nc))
    out, done = [], 0
    for fold, tr, te in walk_forward_folds(labeler.calendar(), train_m,
                                           test_m):
        if done >= max_folds:
            break
        ts0 = te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(),
                                       tr['timestamp'].max(), ts0)
        Cte, _, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50:
            continue                       # unproductive fold: don't count
        done += 1
        Ytr = np.asarray(Ytr)
        Xtr = _featurize(labeler, Ctr, Ktr)
        Xte = _featurize(labeler, Cte, Kte)
        print(f"\n== fold {fold} | ntr={len(Ytr)} nte={len(Cte)} | "
              f"feat_dim={Xtr.shape[1]} | classes={labeler.n_classes} ==")
        for seed in seeds:
            R = labeler.evaluate(
                Kte, head_factory(labeler.n_classes)
                .fit(Xtr, Ytr, seed).predict(Xte))
            ysh = Ytr.copy()
            np.random.default_rng(seed + 1).shuffle(ysh)
            Rs = labeler.evaluate(
                Kte, head_factory(labeler.n_classes)
                .fit(Xtr, ysh, seed).predict(Xte))
            rnd = np.random.default_rng(seed + 2).integers(
                0, labeler.n_classes, len(Kte))
            Rr = labeler.evaluate(Kte, rnd)
            print(f"  seed {seed}  [REAL   ] {_stats(R)}")
            print(f"          [SHUFFLE] {_stats(Rs)}")
            print(f"          [RANDOM ] {_stats(Rr)}")
            out.append({'fold': fold, 'seed': seed,
                        'REAL': R, 'SHUFFLE': Rs, 'RANDOM': Rr})
    print("\n-> Believe a result only if REAL clearly beats SHUFFLE AND "
          "RANDOM on sumR/meanR across seeds (cost already in evaluate()).")
    return out
