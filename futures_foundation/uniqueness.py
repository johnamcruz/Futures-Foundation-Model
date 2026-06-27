"""Average-uniqueness sample weights (López de Prado, Advances in Financial ML).

Trading labels built from overlapping outcome windows are NOT IID: a signal at
bar i and the next at bar i+1 share almost their whole VERT-bar forward window,
so they're near-duplicate labels. Training as if they're independent inflates
overfit and distorts feature importance.

The fix: weight each signal by its AVERAGE UNIQUENESS — 1 / (number of concurrent
labels) averaged over its outcome window. Signals whose windows pile up get
down-weighted; isolated signals keep full weight. Feeds straight into XGBoost as
`sample_weight`. Improves generalization for ANY overlapping-label strategy, so
it lives in the foundation; the strategy only supplies the horizon (VERT).

Leak-safety: this is a TRAINING WEIGHT (not a feature). It uses the label spans,
which are forward by definition — but it never enters the model as input and is
applied only to the training set, exactly as López de Prado prescribes.
"""
from __future__ import annotations

import numpy as np


def uniqueness_weights(keys, horizon, normalize=True):
    """Per-signal average-uniqueness weights -> [N] float32. keys[i] =
    (item_id, bar_index, ...); each label spans bars [bar+1, bar+1+horizon].
    Concurrency is counted PER STREAM (item_id). With normalize=True the weights
    are scaled to mean 1 (a reweighting, not a magnitude change)."""
    H = int(horizon)
    N = len(keys)
    w = np.ones(N, np.float32)
    if N == 0:
        return w
    bars = np.fromiter((int(k[1]) for k in keys), np.int64, N)
    streams = {}
    for n, k in enumerate(keys):
        streams.setdefault(k[0], []).append(n)

    for rows in streams.values():
        rows = np.asarray(rows)
        b = bars[rows]
        start = b + 1                       # outcome window [start, end] inclusive
        end = b + 1 + H
        lo = int(start.min())
        L = int(end.max()) - lo + 1         # bar positions 0..L-1
        # concurrency per bar via difference array (counts EACH label, incl. both
        # directions of a bar — they're separate overlapping labels)
        diff = np.zeros(L + 1)
        np.add.at(diff, start - lo, 1.0)
        np.add.at(diff, end - lo + 1, -1.0)
        conc = np.maximum(np.cumsum(diff)[:L], 1e-9)
        inv = 1.0 / conc                    # 1/concurrency per bar
        cuminv = np.concatenate(([0.0], np.cumsum(inv)))
        s = start - lo
        e = end - lo
        w[rows] = ((cuminv[e + 1] - cuminv[s]) / (e - s + 1)).astype(np.float32)

    if normalize:
        m = float(w.mean())
        if m > 0:
            w = (w / m).astype(np.float32)
    return w
