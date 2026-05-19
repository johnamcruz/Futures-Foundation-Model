"""Subprocess worker: frozen Chronos embeddings.

Isolates torch's OpenMP from xgboost's — they segfault in one process on
macOS. NOT imported by the parent; invoked as
`python -m pipelines.chronos._embed_worker IN.npy OUT.npy BATCH`.
Reuses backbone's pretrained-load + masked-mean pool (torch loads HERE).
"""
import sys

import numpy as np


def main(inp, outp, batch):
    import torch

    from . import backbone
    m = backbone._model()
    m.eval()
    X = np.load(inp).astype(np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(backbone.pool(
                m, torch.tensor(X[s:s + batch])).cpu().numpy())
    np.save(outp, np.concatenate(out).astype(np.float32))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
