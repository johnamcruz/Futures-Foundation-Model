"""Tests — futures_foundation.context (ContextHeads + promoted labels).

xgboost runs in-process here (the default suite is torch-free since the
Bolt-foundation refactor, so no libomp collision). No torch imports.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation import context as ctx
from futures_foundation.context import ContextHeads, compute_context_labels

RNG = np.random.default_rng(11)


def test_probe_script_uses_library_generators():
    """The probe script must alias the library functions — no drift."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'probe_context_heads',
        Path(__file__).resolve().parents[1] / 'scripts'
        / 'probe_context_heads.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.compute_labels is compute_context_labels
    assert probe.HEADS is ctx.HEAD_SPECS


def _synthetic_dataset(n=3000, d=24, informative=True):
    """Embeddings where the labels ARE (noisily) recoverable from the
    first dims — so every head must clear its gate; or pure noise so none
    may. Returns (E_tr, lab_tr, E_va, lab_va)."""
    E = RNG.normal(0, 1, (n, d)).astype(np.float32)
    lab = pd.DataFrame()
    if informative:
        lab['fwd_return'] = np.clip(E[:, 0] + RNG.normal(0, .3, n), -4, 4)
        lab['vol_expansion'] = (E[:, 1] + RNG.normal(0, .3, n) > 0).astype(float)
        lab['volatility'] = 1 / (1 + np.exp(-(E[:, 2] + RNG.normal(0, .3, n))))
        s = E[:, 3] + RNG.normal(0, .3, n)
        lab['structure'] = np.where(s > .5, 1.0, np.where(s < -.5, 0.0, np.nan))
        lab['range_pos'] = 1 / (1 + np.exp(-(E[:, 4] + RNG.normal(0, .3, n))))
    else:
        lab['fwd_return'] = RNG.normal(0, 1, n)
        lab['vol_expansion'] = (RNG.random(n) > .5).astype(float)
        lab['volatility'] = RNG.random(n)
        lab['structure'] = (RNG.random(n) > .5).astype(float)
        lab['range_pos'] = RNG.random(n)
    cut = int(n * .8)
    return E[:cut], lab.iloc[:cut], E[cut:], lab.iloc[cut:]


@pytest.fixture(scope='module')
def fitted_heads():
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(informative=True)
    return (ContextHeads(seed=0, n_estimators=60)
            .fit(E_tr, lab_tr, E_va, lab_va, verbose=False)), E_va


def test_fit_gates_pass_on_recoverable_labels(fitted_heads):
    heads, _ = fitted_heads
    for name, kind in ctx.HEAD_SPECS:
        m = heads.metrics[name]
        assert m['passed'], f"{name} should clear its gate: {m}"
        assert m['metric'] == ('pearson_r' if kind == 'reg' else 'auc')
    assert heads.active_names == [f'ctx_{n}' for n, _ in ctx.HEAD_SPECS]


def test_transform_shape_order_dtype(fitted_heads):
    heads, E_va = fitted_heads
    X = heads.transform(E_va)
    assert X.shape == (len(E_va), len(heads.active_names))
    assert X.dtype == np.float32
    # clf columns are probabilities
    i_ve = heads.active_names.index('ctx_vol_expansion')
    assert 0.0 <= X[:, i_ve].min() and X[:, i_ve].max() <= 1.0


def test_transform_include_override_for_ablation(fitted_heads):
    heads, E_va = fitted_heads
    X = heads.transform(E_va, include=['volatility', 'vol_expansion'])
    assert X.shape == (len(E_va), 2)


def test_noise_labels_fail_gates_and_transform_is_empty():
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(informative=False)
    heads = ContextHeads(seed=0, n_estimators=40).fit(
        E_tr, lab_tr, E_va, lab_va, verbose=False)
    assert heads.active_names == []          # nothing may pass on noise
    assert heads.transform(E_va).shape == (len(E_va), 0)


def test_save_load_roundtrip(fitted_heads, tmp_path):
    heads, E_va = fitted_heads
    heads.meta = {'cutoff': str(ctx.HEADS_CUTOFF), 'note': 'unit'}
    p = heads.save(tmp_path / 'heads.joblib')
    loaded = ContextHeads.load(p)
    assert loaded.active_names == heads.active_names
    assert loaded.meta['note'] == 'unit'
    np.testing.assert_allclose(loaded.transform(E_va), heads.transform(E_va),
                               rtol=1e-6)


def test_structure_nan_rows_dropped_not_filled(fitted_heads):
    heads, _ = fitted_heads
    m = heads.metrics['structure']
    # synthetic structure has NaN (mixed) rows — they must be excluded
    assert m['n_train'] < 2400 and m['n_val'] < 600


def test_too_few_rows_marks_head_skipped():
    E_tr, lab_tr, E_va, lab_va = _synthetic_dataset(n=400)
    heads = ContextHeads(n_estimators=10).fit(
        E_tr, lab_tr, E_va, lab_va, verbose=False)   # 320 train < 500 min
    assert all(not m['passed'] for m in heads.metrics.values())
    assert all('skipped' in m for m in heads.metrics.values())


def test_heads_cutoff_constant_is_utc_2023():
    assert ctx.HEADS_CUTOFF == pd.Timestamp('2023-01-01', tz='UTC')
    assert ctx.MAX_LABEL_HORIZON == 20
