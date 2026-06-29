"""Optuna tuner (tune.py) + overfit loop (loop.py) — torch-free via 'logistic'.

Exercises the generic overfit/Optuna machinery end-to-end (Optuna search on a val
guard; the loop's default->check->tune->rerun->final flow) without torch.
"""
import numpy as np
import pandas as pd

from futures_foundation.finetune import tune, loop


class SyntheticLabeler:
    n_classes = 2

    def __init__(self, n_bars=1500, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'SYN', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        idx = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        return [None] * len(idx), self.y[idx], [(int(i),) for i in idx]

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[0]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


def test_tune_returns_valid_search_result():
    lab = SyntheticLabeler(seed=0)
    rng = np.random.default_rng(0)
    N, C, T = 400, 4, 32
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    X[y == 1, 0, -6:] += 2.5
    res = tune.tune(lab, 'logistic', X[:300], y[:300], X[300:], y[300:],
                    n_trials=4, seed=0, verbose=False)
    for k in ('params', 'val_auc', 'guard_lift', 'generalizes'):
        assert k in res
    assert 0.0 <= res['val_auc'] <= 1.0
    assert isinstance(res['params'], dict)


def test_train_loop_runs_and_returns_final():
    lab = SyntheticLabeler(n_bars=1500, seed=0)
    out = loop.train_loop(lab, classifier='logistic', max_iters=1, loop_max_folds=1,
                          final_max_folds=1, seeds=(0,), train_m=12, val_m=3, test_m=3,
                          holdout_start=None, n_trials=3, verbose=False)
    assert 'final' in out and 'history' in out and 'params' in out
    assert out['final']['n_folds'] >= 1
    # learnable signal -> the final walk-forward discriminates
    assert out['final']['auc'] is not None and out['final']['auc'] > 0.8
