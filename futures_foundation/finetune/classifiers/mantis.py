"""MantisClassifier — torch-free parent-side adapter.

featurize() builds the strategy's multivariate windows (numpy, via the labeler's
mv_contexts). fit_predict() spawns the isolated torch worker (_worker -> _mantis_torch)
so torch never shares a process with xgboost (libomp segfault). Each fit_predict is a
fresh subprocess → MPS/RAM freed on exit.

Registered as 'mantis'. Config kwargs (new_channels, ft_mode, unfreeze_blocks, epochs,
batch, lr, weight_decay, patience, threads, device, max_train, verbose) are forwarded
to the torch trainer.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..classifier import Classifier, register_classifier


@register_classifier('mantis')
class MantisClassifier(Classifier):
    needs_standardize = True            # harness standardizes [N,C,seq] on train stats

    def __init__(self, **cfg):
        self.cfg = cfg

    def featurize(self, labeler, keys):
        return np.asarray(labeler.mv_contexts(keys), np.float32)

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, arr in [('Xtr', Xtr), ('ytr', ytr), ('Xval', Xval),
                              ('yval', yval), ('Xeval', Xeval)]:
                np.save(d / f'{name}.npy', np.asarray(arr))
            (d / 'cfg.json').write_text(json.dumps(dict(self.cfg, seed=int(seed))))
            r = subprocess.run(
                [sys.executable, '-m', 'futures_foundation.finetune.classifiers._worker', str(d)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"mantis worker failed:\n{r.stderr[-3000:]}")
            meta = np.load(d / 'meta.npy')
            return np.load(d / 'p_val.npy'), np.load(d / 'p_eval.npy'), float(meta[0])
