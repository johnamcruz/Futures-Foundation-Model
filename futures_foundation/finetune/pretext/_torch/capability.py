"""CAPABILITY PROBES — the representation's report card, emitted DURING SSL training.

The problem this fixes: SSL trainers report PRETEXT metrics (skill, legR, emb_std) — they say the
objective is converging, NOT whether the REPRESENTATION improved. Capability only surfaced later,
via the separate three-gate ritual (scorecard -> dry-run -> WR), after a full run was already
spent. lc512 / turn-electra burned complete runs before their drift-damage showed.

This runs a tiny frozen-embedding linear probe battery on a held-out slice every N epochs, so
capability formation is VISIBLE while training:
    trend_start   — is this pivot a REAL start (vs whipsaw)?      [the money dimension]
    vol_regime    — is the window in a high-vol regime?           [retention: was it KEPT?]
    ...caller-supplied; any (name -> y) map of causal/known truths.

★ DIAGNOSTIC ONLY — NEVER a training signal. No early-stopping on probes, no objective
weighting, no gradient. The moment probes steer training, the encoder optimizes for the probes
and the drift-damage law comes for it (banked: turn-electra). Report; don't chase. Gates stay
scorecard -> tiers -> ruler.
"""
import numpy as np


def probe_battery(embed_fn, windows, truths, max_n=4000, seed=0):
    """Fit/score a 2-fold linear probe per truth on frozen embeddings of `windows`.

    embed_fn: [N,C,seq] -> [N,D] (the trainer's own frozen-encoder embed, torch-free to caller)
    truths:   {name: y[N] bool/int} — causal or known-at-eval ground truths
    returns:  {name: auc} — chance = 0.5; a LOW retention AUC = the encoder DISCARDS that info.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = min(len(windows), max_n)
    idx = rng.choice(len(windows), n, replace=False) if len(windows) > n else np.arange(n)
    X = np.asarray(embed_fn(np.asarray(windows)[idx]), np.float32)
    half = len(idx) // 2
    out = {}
    for name, y in truths.items():
        y = np.asarray(y)[idx].astype(int)
        if y.std() == 0 or half < 50:
            continue
        aucs = []
        for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
            if y[a].std() == 0 or y[b].std() == 0:
                continue
            clf = LogisticRegression(max_iter=500, C=0.1).fit(X[a], y[a])
            aucs.append(roc_auc_score(y[b], clf.predict_proba(X[b])[:, 1]))
        if aucs:
            out[name] = float(np.mean(aucs))
    return out


def format_probes(probes):
    """One-line log fragment: 'probe start=0.771 vol=0.884' (empty string when no probes)."""
    return ('  probe ' + ' '.join(f'{k}={v:.3f}' for k, v in sorted(probes.items()))
            if probes else '')
