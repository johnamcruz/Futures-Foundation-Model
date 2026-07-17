"""HEAD-FIT REPORT + honest OOS labels — the training-side logging added 2026-07-16.

head_fit_report captures the fit's own curve: an ENCODER diagnostic (fast convergence on a
frozen embedding = the task is near-linear in that space) plus the under-training caveat
(a max_iter-capped fit taints every downstream number). DIAGNOSTIC ONLY — never a gate.
"""
import numpy as np
import pytest

from futures_foundation.finetune.classifiers.mantis.frozen import head_fit_report


def _xy(n=300, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)   # learnable signal
    return X, y


def test_report_flags_a_capped_fit_as_not_converged():
    """max_iter=1 CANNOT converge -> converged=False. This is the caveat flag: every metric
    below an under-trained head is suspect, and the log must say so."""
    from sklearn.linear_model import LogisticRegression
    X, y = _xy()
    with pytest.warns(Warning):                       # sklearn warns on non-convergence
        clf = LogisticRegression(max_iter=1).fit(X, y)
    rep = head_fit_report(clf, fit_seconds=1.5)
    assert rep['converged'] is False
    assert rep['n_iter'] >= 1 and rep['fit_seconds'] == 1.5


def test_report_flags_a_healthy_fit_as_converged():
    from sklearn.linear_model import LogisticRegression
    X, y = _xy()
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    rep = head_fit_report(clf)
    assert rep['converged'] is True
    assert rep['n_iter'] < 1000
    assert 'fit_seconds' not in rep                   # omitted when not supplied


def test_report_captures_mlp_early_stopping_curve():
    """The MLP curve is the encoder signal: epochs_to_best on a frozen embedding says how
    linearly separable the task is in that space. Curve is truncated to the last 20 points."""
    from sklearn.neural_network import MLPClassifier
    X, y = _xy(n=400)
    clf = MLPClassifier((16,), max_iter=60, early_stopping=True, n_iter_no_change=50,
                        random_state=0).fit(X, y)
    rep = head_fit_report(clf, fit_seconds=12.0)
    assert 'val_curve' in rep and 0 < len(rep['val_curve']) <= 20
    assert rep['epochs_to_best'] >= 1
    assert -1.0 <= rep['best_val_score'] <= 1.0
    assert rep['best_val_score'] == max(rep['val_curve']) or rep['epochs_to_best'] <= len(clf.validation_scores_)
    assert 'final_train_loss' in rep


def test_report_is_empty_for_a_head_without_curves():
    """No n_iter_/curves (e.g. a stub or non-iterative head) -> {} , never a crash."""
    class _Stub:
        pass
    assert head_fit_report(_Stub()) == {}
    assert head_fit_report(_Stub(), fit_seconds=3.0) == {'fit_seconds': 3.0}


def test_produce_logs_the_actual_oos_window_not_a_hardcoded_2026():
    """REGRESSION: the operating-point/alignment headers were hardcoded '2026 OOS', so every
    anchored fold (2022/2023/2024) printed '2026 OOS' — a lie in the run logs. Titles are
    parameterized now and default to a neutral 'OOS'."""
    import inspect
    from futures_foundation.finetune import produce
    src = inspect.getsource(produce)
    assert '2026 OOS' not in src
    assert inspect.signature(produce._print_operating_points).parameters['title'].default == 'OOS'
    assert inspect.signature(produce._print_alignment).parameters['title'].default == 'OOS'
    assert inspect.signature(produce._fit_score).parameters['title'].default == 'OOS'
