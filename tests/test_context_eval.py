"""Tests for the walk-forward context-head evaluation (context_eval).

head_verdict (one fold) + aggregate_verdict (across folds) are the pure decision
cores; run_context_eval is exercised on small synthetic inputs (real tiny
XGBoost, no foundation embedding) over real walk-forward folds.
"""
import numpy as np
import pandas as pd

from futures_foundation import context_eval as CE


# ---- head_verdict (one fold, pure) -----------------------------------------
def test_reg_head_accurate():
    v = CE.head_verdict('reg', 0.60, 0.55, 0.52, trivial_m=0.20, shuffle_m=0.00)
    assert v['generalizes'] and v['has_skill']
    assert v['beats_trivial'] and v['beats_shuffle'] and v['accurate']


def test_reg_head_fails_when_not_generalizing():
    v = CE.head_verdict('reg', 0.60, 0.55, 0.40, trivial_m=0.20, shuffle_m=0.0)
    assert v['generalizes'] is False and v['accurate'] is False


def test_reg_head_fails_below_skill_floor():
    v = CE.head_verdict('reg', 0.06, 0.05, 0.03, trivial_m=0.0, shuffle_m=0.0)
    assert v['has_skill'] is False and v['accurate'] is False


def test_clf_head_accurate_and_overfit_flag():
    v = CE.head_verdict('clf', 0.95, 0.78, 0.76, trivial_m=0.60, shuffle_m=0.50)
    assert v['generalizes'] and v['has_skill'] and v['accurate']
    assert v['overfit'] is True


def test_clf_head_not_generalizing():
    v = CE.head_verdict('clf', 0.80, 0.78, 0.70, trivial_m=0.6, shuffle_m=0.5)
    assert v['generalizes'] is False and v['accurate'] is False


# ---- aggregate_verdict (across folds, pure) --------------------------------
def _fv(val, test, gen, triv=0.2, shuf=0.0):
    return dict(val=val, test=test, generalizes=gen, trivial=triv, shuffle=shuf)


def test_aggregate_accurate_when_generalizes_most_folds():
    folds = [_fv(0.5, 0.5, True) for _ in range(5)]
    v = CE.aggregate_verdict('reg', folds, min_generalize_frac=0.6)
    assert v['gen_frac'] == 1.0 and v['accurate'] is True
    assert v['n_folds'] == 5 and v['mean_test'] == 0.5


def test_aggregate_fails_when_too_few_folds_generalize():
    folds = [_fv(0.5, 0.5, i < 2) for i in range(5)]      # 2/5 generalize
    v = CE.aggregate_verdict('reg', folds, min_generalize_frac=0.6)
    assert v['gen_frac'] == 0.4 and v['accurate'] is False


def test_aggregate_fails_below_floor():
    folds = [_fv(0.04, 0.03, True, triv=None, shuf=None) for _ in range(3)]
    v = CE.aggregate_verdict('reg', folds, 0.6)
    assert v['has_skill'] is False and v['accurate'] is False


def test_aggregate_fails_when_not_beating_trivial():
    folds = [_fv(0.5, 0.30, True, triv=0.40) for _ in range(4)]  # test < trivial
    v = CE.aggregate_verdict('reg', folds, 0.6)
    assert v['beats_trivial'] is False and v['accurate'] is False


def test_aggregate_empty():
    v = CE.aggregate_verdict('reg', [])
    assert v['n_folds'] == 0 and v['accurate'] is False


# ---- run_context_eval (synthetic walk-forward, end-to-end) -----------------
def _synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    ts = pd.Series(pd.date_range('2019-01-01', '2024-06-01', periods=n, tz='UTC'))
    y_reg = (3.0 * X[:, 0] + 0.2 * rng.normal(size=n)).astype(np.float32)
    y_clf = (X[:, 1] + 0.2 * rng.normal(size=n) > 0).astype(np.float32)
    labels = pd.DataFrame({'volatility': y_reg, 'vol_expansion': y_clf})
    T = X[:, 5:7].copy()                          # trivial baseline = unrelated cols
    items = np.array(['SYN'] * n)
    return X, labels, ts, T, items


def test_run_context_eval_walkforward_learnable_heads():
    X, labels, ts, T, items = _synth()
    res = CE.run_context_eval(
        X, labels, ts, ts_end=ts, item_ids=items,
        train_m=6, val_m=2, test_m=2, T=T,
        specs=[('volatility', 'reg'), ('vol_expansion', 'clf')], verbose=False)
    assert set(res) == {'volatility', 'vol_expansion'}
    for name, v in res.items():
        assert v['n_folds'] >= 2
        assert v['has_skill'] is True            # learnable → clears the floor
        assert v['gen_frac'] >= 0.5              # holds across most folds
        assert v['accurate'] is True


def test_run_context_eval_skips_unlabeled_head():
    X, labels, ts, T, items = _synth()
    labels['empty_head'] = np.nan
    res = CE.run_context_eval(
        X, labels, ts, ts_end=ts, item_ids=items,
        train_m=6, val_m=2, test_m=2, T=T,
        specs=[('empty_head', 'reg'), ('volatility', 'reg')], verbose=False)
    assert res['empty_head']['n_folds'] == 0 and res['empty_head']['accurate'] is False
    assert res['volatility']['accurate'] is True
