"""Classifier seam + registry + torch-free LogisticClassifier (featurize/fit_predict).

Torch-free — runs in the default suite. Mantis is tested in test_mantis_classifier.py
(torch-gated).
"""
import numpy as np
import pytest

from futures_foundation.finetune.classifier import (
    Classifier, get_classifier, register_classifier, _REGISTRY)


class _StubLabeler:
    """Minimal labeler exposing mv_contexts (what featurize delegates to)."""
    def __init__(self, X):
        self._X = X
    def mv_contexts(self, keys):
        return np.stack([self._X[k] for k in keys])


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Classifier()                       # abstract featurize/fit_predict


def test_registry_and_get_classifier():
    @register_classifier('_unit_dummy')
    class _Dummy(Classifier):
        def __init__(self, n_channels=1, **kw):
            self.n_channels = n_channels
        def featurize(self, labeler, keys):
            return np.zeros((len(keys), 1, 1))
        def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
            return np.zeros(len(Xval)), np.zeros(len(Xeval)), 0.5
    assert '_unit_dummy' in _REGISTRY
    clf = get_classifier('_unit_dummy', n_channels=3)
    assert isinstance(clf, Classifier) and clf.n_channels == 3


def test_get_classifier_unknown_raises():
    with pytest.raises(KeyError):
        get_classifier('does_not_exist')


def test_get_classifier_lazy_loads_logistic():
    assert isinstance(get_classifier('logistic'), Classifier)


def test_logistic_featurize_delegates_to_mv_contexts():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4, 8)).astype(np.float32)
    lab = _StubLabeler(X)
    clf = get_classifier('logistic')
    out = clf.featurize(lab, [0, 2, 5])
    assert out.shape == (3, 4, 8)


def test_logistic_fit_predict_shapes_and_learns():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    N, C, T = 400, 4, 16
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    X[y == 1, 0, -4:] += 2.0                 # separable signal
    clf = get_classifier('logistic')
    p_val, p_eval, ba = clf.fit_predict(X[:280], y[:280], X[280:340], y[280:340],
                                        X[340:], seed=0)
    assert p_val.shape == (60,) and p_eval.shape == (60,)
    assert np.all((p_eval >= 0) & (p_eval <= 1))
    assert ba > 0.85                          # val guard AUC
    assert roc_auc_score(y[340:], p_eval) > 0.85
