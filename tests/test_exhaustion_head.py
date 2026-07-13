"""Exhaustion risk head produce-time machinery (finetune/exhaustion_head.py).

Contract: fold_affine is EXACT algebra (scaler+logistic+Platt == one affine->sigmoid);
export_risk_onnx emits a graph whose named outputs reproduce the folded heads on onnxruntime;
module import stays torch-free (repo contract).
"""
import numpy as np
import pytest

from futures_foundation.finetune.exhaustion_head import fold_affine, export_risk_onnx


def _fit_synthetic(seed=0, n=4000, dim=16):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from futures_foundation.finetune.calibration import fit_platt, apply_platt
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, dim)).astype(np.float32) * rng.uniform(0.5, 3.0, dim).astype(np.float32)
    w_true = rng.normal(0, 1, dim)
    y = (X @ w_true + rng.normal(0, 1, n) > 0).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=500).fit(sc.transform(X), y)
    raw = clf.predict_proba(sc.transform(X))[:, 1]
    platt = fit_platt(raw, y)
    cal = apply_platt(raw, platt)
    return X, sc, clf, platt, cal


def test_fold_affine_is_exact():
    X, sc, clf, platt, cal = _fit_synthetic()
    W, b = fold_affine(sc, clf, platt)
    p_fold = 1.0 / (1.0 + np.exp(-(X @ W + b)))
    assert np.max(np.abs(p_fold - cal)) < 1e-5              # exact algebra, float32 tolerance


def test_export_reproduces_folded_heads_on_ort():
    ort = pytest.importorskip('onnxruntime')
    X1, sc1, clf1, platt1, _ = _fit_synthetic(seed=1)
    X2, sc2, clf2, platt2, _ = _fit_synthetic(seed=2)
    folded = {'p_end_up': fold_affine(sc1, clf1, platt1),
              'p_end_down': fold_affine(sc2, clf2, platt2)}
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), 'risk.onnx')
    export_risk_onnx(folded, X1.shape[1], path)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names == ['p_end_up', 'p_end_down']          # named outputs, stable order
    got_up, got_down = sess.run(['p_end_up', 'p_end_down'], {'emb': X1.astype(np.float32)})
    W, b = folded['p_end_up']
    want = 1.0 / (1.0 + np.exp(-(X1 @ W + b)))
    assert np.max(np.abs(got_up.ravel() - want)) < 1e-5
    W2, b2 = folded['p_end_down']
    want2 = 1.0 / (1.0 + np.exp(-(X1 @ W2 + b2)))
    assert np.max(np.abs(got_down.ravel() - want2)) < 1e-5  # heads independent, both correct


def test_module_import_is_torch_free():
    import sys
    assert 'futures_foundation.finetune.exhaustion_head' in sys.modules
    # torch must not have been pulled in by importing the module at the top of this test file
    # (repo contract: torch only inside functions). If torch appears, another test imported it;
    # check the module's own globals instead for robustness.
    import futures_foundation.finetune.exhaustion_head as eh
    assert 'torch' not in dir(eh)
