"""Tests for the shared, metric-agnostic overfit/generalization library."""
import pytest

from futures_foundation import overfit as O


# ---- overfit_trigger -------------------------------------------------------
def test_overfit_trigger_fires_and_quiets():
    assert O.overfit_trigger(1.0, 0.0, 0.30) is True       # gap 1.0 > 0.30
    assert O.overfit_trigger(0.50, 0.40, 0.30) is False    # gap 0.10 < 0.30
    assert O.overfit_trigger(0.40, 0.50, 0.30) is False    # val > train
    # metric-agnostic: works for AUC tolerances too
    assert O.overfit_trigger(0.90, 0.80, 0.05) is True     # 0.10 AUC gap > 0.05
    assert O.overfit_trigger(0.82, 0.80, 0.05) is False    # 0.02 AUC gap < 0.05


# ---- generalizes -----------------------------------------------------------
def test_generalizes_gate():
    assert O.generalizes(0.50, 0.55, 0.30) is True         # test > val
    assert O.generalizes(0.50, 0.45, 0.30) is True         # small decay
    assert O.generalizes(1.00, 0.50, 0.30) is False        # 0.50 decay > 0.30
    assert O.generalizes(0.62, 0.55, 0.05) is False        # 0.07 AUC decay > 0.05
    assert O.generalizes(0.62, 0.59, 0.05) is True         # 0.03 AUC decay <= 0.05


def test_generalizes_missing_is_false():
    assert O.generalizes(None, 0.5, 0.3) is False
    assert O.generalizes(0.5, None, 0.3) is False


# ---- gen_score -------------------------------------------------------------
def test_gen_score_prefers_stable_over_peaky():
    stable = O.gen_score([1.0, 1.0, 1.0])
    peaky = O.gen_score([2.0, 0.0, 1.0])                   # same mean, unstable
    assert stable > peaky
    assert stable == pytest.approx(1.0)


def test_gen_score_matches_formula_and_penalty():
    import numpy as np
    pf = [1.5, 0.5]
    assert O.gen_score(pf, 0.5) == pytest.approx(
        float(np.mean(pf)) - 0.5 * float(np.std(pf)))
    # bigger penalty punishes instability harder
    assert O.gen_score(pf, 1.0) < O.gen_score(pf, 0.5)


def test_gen_score_empty_is_floor():
    assert O.gen_score([]) == float('-inf')
    assert O.gen_score(None) == float('-inf')


# ---- best_config (selection + auto-fallback) -------------------------------
def test_best_config_picks_highest_above_default():
    a, b, c = {'i': 1}, {'i': 2}, {'i': 3}
    assert O.best_config(0.10, [(a, 0.20), (b, 0.50), (c, 0.30)]) is b


def test_best_config_falls_back_when_none_beat_default():
    a, b = {'i': 1}, {'i': 2}
    assert O.best_config(0.60, [(a, 0.20), (b, 0.55)]) is None


def test_best_config_accept_margin():
    a = {'i': 1}
    # tuned 0.33 vs default 0.32: clears margin 0 but not margin 0.05
    assert O.best_config(0.32, [(a, 0.33)], accept_margin=0.0) is a
    assert O.best_config(0.32, [(a, 0.33)], accept_margin=0.05) is None


def test_best_config_empty_and_none_metrics():
    assert O.best_config(0.30, []) is None
    assert O.best_config(0.30, [({'i': 1}, None)]) is None


# ---- regularized_fit (the extracted auto-regularize wheel) ------------------
def test_regularized_fit_keeps_default_when_not_overfit():
    fits = []
    def fit(cfg):
        fits.append(cfg); return cfg
    m, rem, va = O.regularized_fit(fit, lambda m: 0.50, lambda m: 0.50,
                                   reg_candidates=[{'id': 'A'}], overfit_gap=0.30)
    assert rem is None and m == {} and va == 0.50
    assert fits == [{}]                     # candidates NOT fit when no overfit


def test_regularized_fit_swaps_to_best_val_candidate():
    A, B = {'id': 'A'}, {'id': 'B'}
    def fit(cfg):
        return cfg
    def score_train(m):
        return 0.90 if m == {} else 0.60
    def score_val(m):
        return {'': 0.40, 'A': 0.50, 'B': 0.58}[m.get('id', '') if m else '']
    m, rem, va = O.regularized_fit(fit, score_train, score_val,
                                   reg_candidates=[A, B], overfit_gap=0.30)
    assert rem is B and m is B and va == 0.58


def test_regularized_fit_falls_back_when_no_candidate_beats_default():
    A = {'id': 'A'}
    def fit(cfg):
        return cfg
    m, rem, va = O.regularized_fit(
        fit, lambda m: 0.90, lambda m: 0.50 if m == {} else 0.45,
        reg_candidates=[A], overfit_gap=0.30)
    assert rem is None and m == {} and va == 0.50


def test_regularized_fit_no_candidates_keeps_default_even_if_overfit():
    fits = []
    def fit(cfg):
        fits.append(cfg); return cfg
    m, rem, va = O.regularized_fit(fit, lambda m: 0.90, lambda m: 0.40,
                                   reg_candidates=(), overfit_gap=0.30)
    assert rem is None and m == {} and fits == [{}]
