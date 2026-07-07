"""Walk-forward honest ruler (finetune/wf.py) — full flow, torch-free.

Uses a synthetic StrategyLabeler + the torch-free 'logistic' classifier so the
ENTIRE pipeline is exercised — walk_forward_folds, mv_contexts, the isolated
subprocess worker, realized-R eval, REAL/SHUFFLE/RANDOM, and the verdict — with
no torch. (Mantis-specific behavior is covered in test_mantis_classifier.py.)
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import wf


class SyntheticLabeler:
    """Daily bars over ~5y; class-1 windows carry a separable signal (channel 0
    elevated in the last bars). evaluate: a taken good pivot = +2R, bad = -1R."""
    n_classes = 2

    def __init__(self, n_bars=1600, seq=32, C=4, seed=0):
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
        K = [(int(i),) for i in idx]
        Y = self.y[idx]
        C = [self.W[i, 0] for i in idx]      # 1-D context (protocol filler)
        return C, Y, K

    def mv_contexts(self, keys):
        return np.stack([self.W[k[0]] for k in keys]) if keys else np.zeros((0, self.C, self.seq))

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


# ---- pure helper unit tests ------------------------------------------------
def test_pct_threshold():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert wf._pct_threshold(p, 0.5) == pytest.approx(0.25, abs=1e-9)
    assert wf._pct_threshold(np.array([]), 0.5) == 1.0


def test_meanR():
    assert wf._meanR([1.0, -1.0, 2.0]) == pytest.approx(2 / 3)
    assert wf._meanR([]) == 0.0


def test_arm_R_selects_top_and_evaluates():
    lab = SyntheticLabeler(n_bars=50)
    keys = [(i,) for i in range(20)]
    proba = np.linspace(0, 1, 20)
    R = wf._arm_R(lab, keys, proba, thr=0.5)       # take top ~half
    assert R.ndim == 1 and len(R) > 0
    # all-skip -> empty
    assert wf._arm_R(lab, keys, np.zeros(20), thr=0.5).size == 0


# ---- full flow (subprocess worker + logistic, no torch) --------------------
def test_wf_full_flow_real_beats_controls():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    v = wf.run(lab, classifier='logistic', clf_kwargs=dict(),
               seeds=(0,), train_m=12, val_m=3, test_m=3, max_folds=1,
               holdout_start=None, verbose=False)
    # verdict shape
    for k in ('all_pass', 'generalizes', 'auc', 'real_meanR', 'shuffle_meanR',
              'random_meanR', 'gap', 'n_folds', 'real_trades'):
        assert k in v
    assert v['n_folds'] >= 1
    assert v['real_trades'] > 0
    # the signal is learnable -> REAL must beat both controls and discriminate
    assert v['auc'] is not None and v['auc'] > 0.8
    assert v['real_meanR'] - v['shuffle_meanR'] >= wf.PASS_LIFT_MARGIN_R
    assert v['real_meanR'] - v['random_meanR'] >= wf.PASS_LIFT_MARGIN_R


def test_wf_no_productive_folds_returns_zero():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    # holdout in the far future -> no test data -> zero folds, graceful
    v = wf.run(lab, classifier='logistic', seeds=(0,), train_m=12, val_m=3,
               test_m=3, max_folds=1, holdout_start='2099-01-01', verbose=False)
    assert v['n_folds'] == 0 and v['real_trades'] == 0


# ---- disconnect-resume (shared toolkit: finetune/resume.py) ----------------
def test_resume_keyed_checkpoint_roundtrip(tmp_path):
    """KeyedCheckpoint: state survives a 'disconnect' (reload), a changed config key starts
    fresh, and a missing/corrupt file is treated as absent — never an error."""
    from futures_foundation.finetune.resume import KeyedCheckpoint, config_signature
    key = config_signature('logistic', {'C': 1.0}, 0, [[10, 2, 3]])
    path = tmp_path / 'ck.npz'
    ck = KeyedCheckpoint(path, key)
    default = dict(real=[], valm=[])
    res = ck.load(default)
    assert res == default and ck.done(res, 'real') == 0
    res['real'].append(np.array([1.0, -1.0])); res['valm'].append(0.5)
    ck.save(res)
    res2 = KeyedCheckpoint(path, key).load(default)                 # reload = resume
    assert ck.done(res2, 'real') == 1 and float(res2['valm'][0]) == 0.5
    assert np.allclose(res2['real'][0], [1.0, -1.0])
    other = config_signature('logistic', {'C': 2.0}, 0, [[10, 2, 3]])   # config changed
    assert KeyedCheckpoint(path, other).load(default) == default        # -> fresh start
    path.write_bytes(b'garbage')                                        # corrupt file
    assert KeyedCheckpoint(path, key).load(default) == default          # -> fresh, no raise
    KeyedCheckpoint(None, key).save(res)                                # no path = no-op


def test_resume_sqlite_snapshot_restore(tmp_path):
    """sqlite_snapshot -> consistent durable copy; sqlite_restore only fills an ABSENT local DB
    (an existing local DB is newer by construction)."""
    import sqlite3
    from futures_foundation.finetune.resume import sqlite_snapshot, sqlite_restore
    db, bak = str(tmp_path / 'study.db'), str(tmp_path / 'drive' / 'study.db')
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE t (x INTEGER)'); con.execute('INSERT INTO t VALUES (42)')
    con.commit(); con.close()
    assert sqlite_snapshot(db, bak) is True
    import os
    os.remove(db)                                        # the 'disconnect' (local disk lost)
    assert sqlite_restore(bak, db) is True
    con = sqlite3.connect(db)
    assert con.execute('SELECT x FROM t').fetchone()[0] == 42
    con.close()
    assert sqlite_restore(bak, db) is False              # local exists -> no clobber
    assert sqlite_snapshot(str(tmp_path / 'nope.db'), bak) is False   # missing DB -> no-op


def test_run_folds_disconnect_resume(tmp_path):
    """_run_folds with fold_ckpt: a rerun after a 'disconnect' refits NOTHING (all folds
    resume from the checkpoint) and returns the identical verdict; a changed config refits."""
    from futures_foundation.finetune.classifier import register_classifier, Classifier

    calls = {'fits': 0}

    @register_classifier('_countfit')
    class _CountFit(Classifier):
        def __init__(self, **kw):
            self.kw = kw

        def featurize(self, labeler, keys):              # not used by _run_folds
            raise NotImplementedError

        def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0, keys_tr=None, keys_val=None):
            calls['fits'] += 1
            n_v = len(np.load(Xval, mmap_mode='r')) if isinstance(Xval, str) else len(Xval)
            n_e = len(np.load(Xeval, mmap_mode='r')) if isinstance(Xeval, str) else len(Xeval)
            return np.linspace(0, 1, n_v), np.linspace(0, 1, n_e), 0.5

    class _EvalLab:
        def evaluate(self, keys, preds):
            return np.array([2.0 if int(k[1]) % 2 else -1.0
                             for k, p in zip(keys, preds) if p == 1])

    N = 60
    Xall = str(tmp_path / 'X.npy')
    np.save(Xall, np.random.default_rng(0).standard_normal((N, 2, 3)).astype(np.float32))
    Y = (np.arange(N) % 2).astype(int)
    keys = [('S@1', int(i), 1) for i in range(N)]
    folds = [(np.arange(0, 20), np.arange(20, 30), np.arange(30, 40)),
             (np.arange(10, 30), np.arange(30, 40), np.arange(40, 50))]
    ckpt = str(tmp_path / 'folds.npz')

    v1 = wf._run_folds('_countfit', {}, Xall, Y, keys, _EvalLab(), tmp_path, folds,
                       seed=0, verbose=False, monitor=None, fold_ckpt=ckpt)
    assert calls['fits'] == 2 * 2                        # 2 folds x (REAL + SHUFFLE)
    v2 = wf._run_folds('_countfit', {}, Xall, Y, keys, _EvalLab(), tmp_path, folds,
                       seed=0, verbose=False, monitor=None, fold_ckpt=ckpt)
    assert calls['fits'] == 2 * 2                        # resumed: ZERO new fits
    for k in ('real_meanR', 'shuffle_meanR', 'random_meanR', 'gap', 'real_trades'):
        assert v1[k] == v2[k], k                         # byte-identical verdict
    wf._run_folds('_countfit', {'C': 9.0}, Xall, Y, keys, _EvalLab(), tmp_path, folds,
                  seed=0, verbose=False, monitor=None, fold_ckpt=ckpt)
    assert calls['fits'] == 4 * 2                        # config changed -> key changed -> refit


def test_resume_atomic_replace_cross_device(tmp_path, monkeypatch):
    """EXDEV regression (Colab: local tmp -> Drive FUSE): os.replace cannot cross filesystems;
    atomic_replace must fall back to a destination-side staged copy + replace."""
    import errno
    import os as _os
    from futures_foundation.finetune import resume as R
    src = tmp_path / 'a.txt'; src.write_text('payload')
    dst = tmp_path / 'drive' / 'b.txt'
    real_replace = _os.replace
    calls = {'n': 0}

    def fake_replace(a, b):
        calls['n'] += 1
        if calls['n'] == 1:                                # first attempt = cross-device
            raise OSError(errno.EXDEV, 'Invalid cross-device link')
        return real_replace(a, b)

    monkeypatch.setattr(R.os, 'replace', fake_replace)
    R.atomic_replace(str(src), str(dst))
    assert dst.read_text() == 'payload' and not src.exists()
    assert not (tmp_path / 'drive' / 'b.txt.staged').exists()
