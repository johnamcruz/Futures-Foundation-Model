"""Chronos-Bolt backbone seam — load, pristine reset, causal pooling.

The ONLY module that touches the Chronos library. `embed()` runs the
torch/Chronos work in an ISOLATED SUBPROCESS: on macOS torch's libomp and
xgboost's libomp segfault in one process, so the parent (which also runs
XGBoost) must stay torch-free. The pretrained-load / pool / fresh_model
helpers below are torch and used ONLY inside the subprocess worker or the
legacy in-process NN fine-tune path — never by the XGBoost eval parent.
"""
from pathlib import Path

import numpy as np

MODEL = 'amazon/chronos-bolt-tiny'
D_MODEL = 256                          # bolt-tiny (torch-free constant)
_PIPE = None
_PRISTINE = None                       # cloned pretrained state_dict


def pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained(
            MODEL, device_map='cpu', dtype=torch.float32)
    return _PIPE


def _model():
    p = pipeline()
    return getattr(p, 'inner_model', None) or p.model


def d_model():
    return _model().config.d_model


def fresh_model():
    """The backbone reset to pretrained weights (independent fine-tunes)."""
    global _PRISTINE
    m = _model()
    if _PRISTINE is None:
        _PRISTINE = {k: v.detach().clone()
                     for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(_PRISTINE)
    return m


def pool(m, ctx):
    """Masked-mean pool of the encoder hidden states. ctx: [B,L] tensor of
    causal log-price context (bars <= decision t — the caller's contract)."""
    h, _ls, _emb, mask = m.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


def embed(contexts, batch=64):
    """FROZEN batched embeddings, computed in an isolated subprocess so the
    torch-free parent can run XGBoost safely (macOS OpenMP segfault if torch
    + xgboost share a process). Deterministic; pretrained weights, no grad.
    contexts: iterable of equal-length 1-D causal windows. -> float32
    [N, D_MODEL]. This function imports NO torch in the parent."""
    import os
    import sys
    import subprocess
    import tempfile

    X = np.asarray(contexts, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, D_MODEL), np.float32)
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as d:
        ip, op = os.path.join(d, 'in.npy'), os.path.join(d, 'out.npy')
        np.save(ip, X)
        env = dict(os.environ,
                   PYTHONPATH=str(root) + os.pathsep
                   + os.environ.get('PYTHONPATH', ''))
        r = subprocess.run(
            [sys.executable, '-m', 'pipelines.chronos._embed_worker',
             ip, op, str(batch)],
            cwd=str(root), env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError(
                "chronos embed worker failed:\n" + r.stderr[-2000:])
        return np.load(op)
