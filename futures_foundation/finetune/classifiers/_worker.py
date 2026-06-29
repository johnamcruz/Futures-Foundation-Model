"""Subprocess worker — runs the Mantis torch fine-tune in ISOLATION.

Run as: python -m futures_foundation.finetune.classifiers._worker <workdir>
Reads  : Xtr/ytr/Xval/yval/Xeval.npy + cfg.json
Writes : p_val.npy, p_eval.npy, meta.npy = [best_val_auc, best_epoch]

Loads ONLY torch + the Mantis backbone (never xgboost). Fresh process per call →
MPS/RAM released on exit.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(wd):
    wd = Path(wd)
    cfg = json.loads((wd / 'cfg.json').read_text())
    Xtr, ytr = np.load(wd / 'Xtr.npy'), np.load(wd / 'ytr.npy')
    Xval, yval = np.load(wd / 'Xval.npy'), np.load(wd / 'yval.npy')
    Xeval = np.load(wd / 'Xeval.npy')
    from futures_foundation.finetune.classifiers._mantis_torch import fit_predict_torch
    p_val, p_eval, ba, be = fit_predict_torch(Xtr, ytr, Xval, yval, Xeval, **cfg)
    np.save(wd / 'p_val.npy', p_val)
    np.save(wd / 'p_eval.npy', p_eval)
    np.save(wd / 'meta.npy', np.array([ba, be], float))


if __name__ == '__main__':
    main(sys.argv[1])
