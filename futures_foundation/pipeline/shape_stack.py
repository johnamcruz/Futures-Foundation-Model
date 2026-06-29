"""OOF-stacked shape-adapter feature for the selection pipeline.

Turns the ShapeAwareAdapter (which learns the DEVELOPMENT of per-bar features over
time) into a leak-safe, overfit-guarded FEATURE the pipeline can fuse with the
Chronos embed + handcraft. Torch-free ORCHESTRATOR — it spawns the adapter worker
in a subprocess, so it's safe to import in the xgboost parent (libomp isolation).

Discipline (matches the rest of the pipeline):
  - LEAK-SAFE: adapter trained OUT-OF-FOLD on train (K-1 folds -> CLS for the held
    fold); train-on-all-train -> clean CLS for the test rows. XGBoost never sees a
    CLS the adapter fit on that row's label.
  - OVERFIT-GUARDED: every adapter fit uses an inner train/val split with EARLY
    STOPPING + LR SCHEDULING (in fit_and_infer) and returns its best VAL AUC.
  - MONITORED: mean val AUC is returned so the caller can gate the VAL->TEST gap
    (val >> test => overfit; reuse finetune.health thresholds).

Usage:
    cls, info = oof_adapter_feature(seq, y, train_mask, device='mps')
    X = np.hstack([chronos_embed, handcraft, cls])      # fuse, then XGBoost
    gap = val_test_gap(info['mean_val_auc'], test_auc)  # monitor
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
VAL_TEST_GAP_THRESHOLD = 0.10           # matches finetune.health default


def _run_adapter_worker(seq, y, train_mask, *, proj_dim, epochs, device, proto, lr,
                        patience):
    with tempfile.TemporaryDirectory() as d:
        tok = os.path.join(d, 't.npy'); np.save(tok, seq.astype(np.float32))
        yp = os.path.join(d, 'y.npy'); np.save(yp, np.asarray(y).astype(np.int64))
        trp = os.path.join(d, 'tr.npy'); np.save(trp, np.asarray(train_mask, bool))
        pref = os.path.join(d, 'o')
        env = dict(os.environ, PYTHONPATH=str(_ROOT), ADAPTER_PROJ=str(proj_dim),
                   ADAPTER_PROTO='1' if proto else '0', ADAPTER_LR=str(lr),
                   ADAPTER_PATIENCE=str(patience))
        r = subprocess.run(
            [sys.executable, '-m', 'futures_foundation.extractors.chronos.shape_adapter',
             tok, yp, trp, pref, device, str(epochs)],
            cwd=str(_ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("shape-adapter worker failed:\n" + r.stderr[-3000:])
        cls = np.load(pref + '_cls.npy')
        probs = np.load(pref + '_probs.npy')
        val_auc = float(np.load(pref + '_valauc.npy')[0])
        return cls, probs, val_auc


def oof_adapter_feature(seq, y, train_mask, *, n_folds=3, proj_dim=64, epochs=80,
                        device='cpu', proto=False, lr=2e-3, patience=10,
                        standardize=True, seed=0, verbose=True):
    """Leak-safe OOF shape-adapter feature.

    seq        : [N, T, F] per-bar feature sequences (e.g. all handcraft over time)
    y          : [N] binary labels
    train_mask : [N] bool (True = train; False = test/OOS)
    returns    : (cls [N, proj_dim], info) where info has per-fold + mean val AUC.

    Standardizes seq per-feature on TRAIN stats only (no leak). Each fold's adapter
    is early-stopped on its own inner val split."""
    seq = np.asarray(seq, np.float32)
    y = np.asarray(y)
    tr = np.asarray(train_mask, bool)
    N, _, F = seq.shape
    if standardize:                                    # train-only stats -> no leak
        flat = seq[tr].reshape(-1, F)
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        seq = ((seq - mu) / sd).astype(np.float32)
    cls = np.zeros((N, proj_dim), np.float32)
    tr_idx = np.flatnonzero(tr)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(tr_idx), n_folds)
    probs = np.zeros(N, np.float32)                    # adapter P(good), for the gap monitor
    val_aucs = []
    for k in range(n_folds):
        held = folds[k]
        mask = np.zeros(N, bool)
        mask[np.setdiff1d(tr_idx, held)] = True
        c, p, va = _run_adapter_worker(seq, y, mask, proj_dim=proj_dim, epochs=epochs,
                                       device=device, proto=proto, lr=lr, patience=patience)
        cls[held] = c[held]
        probs[held] = p[held]
        val_aucs.append(va)
        if verbose:
            print(f"[shape_stack] OOF fold {k+1}/{n_folds} val_auc={va:.4f}", flush=True)
    c, p, va = _run_adapter_worker(seq, y, tr, proj_dim=proj_dim, epochs=epochs,
                                   device=device, proto=proto, lr=lr, patience=patience)
    cls[~tr] = c[~tr]
    probs[~tr] = p[~tr]
    val_aucs.append(va)
    info = {'val_aucs': val_aucs, 'mean_val_auc': float(np.mean(val_aucs)),
            'adapter_probs': probs, 'proj_dim': proj_dim, 'n_folds': n_folds}
    if verbose:
        print(f"[shape_stack] mean val_auc={info['mean_val_auc']:.4f}", flush=True)
    return cls, info


def val_test_gap(mean_val_auc, test_auc, threshold=VAL_TEST_GAP_THRESHOLD):
    """Overfit monitor: val AUC minus test AUC. gap > threshold => the adapter
    over-fit its val window and test generalization is poor (reject / regularize).
    Returns (gap, overfit_flag)."""
    gap = float(mean_val_auc) - float(test_auc)
    return gap, gap > threshold
