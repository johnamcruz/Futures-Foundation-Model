"""Linear-probe validation — does the SSL backbone learn regime / volatility /
structure useful for downstream classification? (the BERT 'probing' check).

Freeze the encoder, take its embedding of clean held-out windows, and fit a LINEAR
probe to predict OHLCV-derived properties of each window. If the SSL-adapted encoder
beats the VANILLA Mantis encoder on these, the contrastive pretraining demonstrably
encoded more downstream-useful structure — BEFORE any strategy head is trained.

Probe targets (computed from the raw window, no strategy labels, no future leak):
  * vol          — realized volatility (std of log-returns)            [regression R2]
  * trend_eff    — directional efficiency |net|/sum|step| (trend<->chop)[regression R2]
  * range_expand — log(2nd-half range / 1st-half range) (compression-> [regression R2]
                   expansion = momentum compression/expansion)
  * direction    — sign of net return                                  [binary AUC]

Target computation is torch-free (numpy) + testable; the embedding extraction lives in
_ssl_torch.embed_encoder (lazy torch).
"""
import numpy as np

_TARGET_KIND = {'vol': 'reg', 'trend_eff': 'reg', 'range_expand': 'reg',
                'fwd_absmove': 'reg', 'direction': 'bin', 'fwd_dir': 'bin'}
# CORE targets define a "useful for buy/sell classification" representation: regime/vol/
# structure (descriptive) + fwd_absmove (is there a tradeable move next — buy/sell-relevant
# and learnable). direction (in-window) and fwd_dir (forward direction) are reported but
# NOT in the pass criterion — directional prediction is genuinely hard / noisy as a gate.
_CORE_TARGETS = ('vol', 'trend_eff', 'range_expand', 'fwd_absmove')


def targets_from_windows(big, starts, seq, fwd_k=16):
    """Compute the probe targets for each window [start, start+seq). big: [T, 5]
    (O,H,L,C,V). Descriptive targets come from the window; FORWARD targets (buy/sell-
    relevant) come from the next `fwd_k` bars AFTER the window — strictly future relative
    to the embedded window (no leak). Caller keeps fwd_k <= max_jitter so the forward bars
    stay in-stream. Returns dict name -> float array [M]."""
    big = np.asarray(big, np.float64)
    s = np.asarray(starts, np.int64)
    rows = s[:, None] + np.arange(seq)[None, :]              # [M, seq]
    win = big[rows]                                          # [M, seq, 5]
    high, low, close = win[:, :, 1], win[:, :, 2], win[:, :, 3]
    logret = np.diff(np.log(np.clip(close, 1e-9, None)), axis=1)   # [M, seq-1]
    net = logret.sum(1)
    vol = logret.std(1)
    trend_eff = np.abs(net) / (np.abs(logret).sum(1) + 1e-9)
    h = seq // 2
    r1 = high[:, :h].max(1) - low[:, :h].min(1)
    r2 = high[:, h:].max(1) - low[:, h:].min(1)
    range_expand = np.log((r2 + 1e-9) / (r1 + 1e-9))
    # FORWARD (buy/sell): log-return over the next fwd_k bars after the window end
    end = np.clip(s + seq - 1, 0, len(big) - 1)
    fwd = np.clip(s + seq - 1 + fwd_k, 0, len(big) - 1)
    fwd_ret = (np.log(np.clip(big[fwd, 3], 1e-9, None))
               - np.log(np.clip(big[end, 3], 1e-9, None)))
    return {'vol': vol.astype(np.float32), 'trend_eff': trend_eff.astype(np.float32),
            'range_expand': range_expand.astype(np.float32),
            'fwd_absmove': np.abs(fwd_ret).astype(np.float32),
            'direction': (net > 0).astype(np.int32),
            'fwd_dir': (fwd_ret > 0).astype(np.int32)}


def _score_predictions(y, pred, kind):
    """Score already-computed predictions, including safe one-class handling."""
    from sklearn.metrics import r2_score, roc_auc_score
    if kind == 'bin':
        return 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, pred))
    return float(r2_score(y, pred))


def _fit_scores(emb, y, kind, tr, te, test_groups=None):
    """Fit once, then return the aggregate score and optional per-stream scores."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(emb[tr])
    Xtr, Xte = sc.transform(emb[tr]), sc.transform(emb[te])
    if kind == 'bin':
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            pred = np.full(len(te), 0.5, dtype=np.float64)
        else:
            # 1280-dim correlated embeddings -> lbfgs hits its iter cap; bump it, and suppress
            # the diagnostic warning. The score remains valid without exact coefficient
            # convergence, and the identical fitted model is scored on every stream below.
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
            pred = m.predict_proba(Xte)[:, 1]
    else:
        pred = Ridge(alpha=1.0).fit(Xtr, y[tr]).predict(Xte)
    overall = _score_predictions(y[te], pred, kind)
    per_group = {}
    if test_groups is not None:
        test_groups = np.asarray(test_groups)
        for group in np.unique(test_groups):
            mask = test_groups == group
            per_group[int(group)] = _score_predictions(y[te][mask], pred[mask], kind)
    return overall, per_group


def _fit_score(emb, y, kind, tr, te):
    """Fit a linear probe on emb[tr] -> score on emb[te]. R2 (reg) or AUC (bin)."""
    return _fit_scores(emb, y, kind, tr, te)[0]


def _stream_groups(starts):
    """Infer contiguous source-stream runs from globally offset validation starts.

    ``ssl.assemble`` appends every stream's legal starts in stream order. Starts are consecutive
    within a stream; the next stream begins after a large global-offset jump. Keeping this logic
    here avoids changing the trainer's long-standing ndarray validation contract.
    """
    starts = np.asarray(starts, np.int64)
    if not len(starts):
        return np.array([], np.int32)
    return np.cumsum(np.r_[False, np.diff(starts) != 1]).astype(np.int32)


def _balanced_probe_sample(starts, max_windows, seed=0):
    """Sample each source stream equally, returning sorted starts and aligned group ids."""
    starts = np.asarray(starts, np.int64)
    groups = _stream_groups(starts)
    if len(starts) <= max_windows:
        return starts, groups
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    quota, remainder = divmod(int(max_windows), len(unique))
    chosen = []
    for pos, group in enumerate(unique):
        idx = np.flatnonzero(groups == group)
        take = min(len(idx), quota + (pos < remainder))
        chosen.append(np.sort(rng.choice(idx, take, replace=False)))
    selected = np.concatenate(chosen)
    return starts[selected], groups[selected]


def _grouped_temporal_split(starts, groups, test_frac=0.3, purge=0):
    """Temporal head/tail split *inside every stream*, purged around each boundary.

    A single split over concatenated streams silently trains on early tickers and tests on late
    tickers. A random split leaks near-duplicate overlapping windows. This split does neither.
    ``purge`` is measured in raw bars and removes train windows whose target horizon reaches the
    first test window.
    """
    starts, groups = np.asarray(starts, np.int64), np.asarray(groups, np.int32)
    train, test = [], []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        cut = max(1, min(len(idx) - 1, int(len(idx) * (1 - test_frac))))
        te = idx[cut:]
        tr = idx[:cut]
        if len(te) and purge:
            tr = tr[starts[tr] + int(purge) < starts[te[0]]]
        if len(tr) and len(te):
            train.append(tr)
            test.append(te)
    if not train:
        raise ValueError('not enough windows for a grouped temporal probe split')
    return np.concatenate(train), np.concatenate(test)


def probe_embedding(emb, y, kind, seed=0, test_frac=0.3, folds=1):
    """Fit a linear probe and score it. folds<=1 -> a single random train/test split (fast).
    folds>1 -> k-fold CV, returning the MEAN score over folds -> a robust, low-variance estimate
    (use this to reliably RANK candidates, e.g. channel-weight configs, where a single split is
    too noisy to trust the ordering)."""
    emb = np.asarray(emb, np.float32)
    n = len(emb)
    # ── SPLIT (2026-07-16): rows are OVERLAPPING windows in stream order, so a SHUFFLED split
    # puts near-duplicate windows on both sides — every probe score (the SSL gates) reads
    # optimistic. Default = CONTIGUOUS blocks (KFold shuffle=False / temporal tail), which keeps
    # overlap only at block edges. PROBE_SPLIT=random reproduces legacy (banked bar) numbers —
    # never compare a contiguous score against a banked shuffled one. ──
    import os
    legacy = os.environ.get('PROBE_SPLIT', 'contiguous') == 'random'
    if folds and folds > 1:
        from sklearn.model_selection import KFold
        kf = (KFold(n_splits=int(folds), shuffle=True, random_state=seed) if legacy
              else KFold(n_splits=int(folds), shuffle=False))
        return float(np.mean([_fit_score(emb, y, kind, tr, te) for tr, te in kf.split(emb)]))
    if legacy:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        nt = int(n * (1 - test_frac))
        return _fit_score(emb, y, kind, idx[:nt], idx[nt:])
    nt = int(n * (1 - test_frac))                       # temporal: test = the stream tail
    idx = np.arange(n)
    return _fit_score(emb, y, kind, idx[:nt], idx[nt:])


def compare(emb_ssl, emb_vanilla, targets, seed=0, folds=1, split=None,
            groups=None, group_names=None):
    """Probe both embeddings on every target; return per-target {ssl, vanilla, delta} plus
    aggregate deltas that the gate uses. We separate DESCRIPTIVE content (vol/trend_eff/
    range_expand — easy in-window stats) from FORWARD predictive content (fwd_absmove size,
    fwd_dir direction), because a shortcut embedding can lift the easy descriptive stats while
    the forward (genuinely predictive) targets barely move. The gate keys on the FORWARD ones.
    folds>1 -> k-fold CV per probe (robust deltas for ranking candidates)."""
    res, core_deltas, desc_deltas = {}, [], []
    group_deltas = {}
    for name, y in targets.items():
        kind = _TARGET_KIND[name]
        if split is None:
            a = probe_embedding(emb_ssl, y, kind, seed, folds=folds)
            b = probe_embedding(emb_vanilla, y, kind, seed, folds=folds)
            a_groups = b_groups = {}
        else:
            tr, te = split
            te_groups = None if groups is None else np.asarray(groups)[te]
            a, a_groups = _fit_scores(emb_ssl, y, kind, tr, te, te_groups)
            b, b_groups = _fit_scores(emb_vanilla, y, kind, tr, te, te_groups)
        res[name] = {'ssl': a, 'vanilla': b, 'delta': a - b, 'kind': kind}
        for group in a_groups:
            label = (str(group_names[group]) if group_names is not None and group < len(group_names)
                     else str(group))
            group_deltas.setdefault(label, {})[name] = float(a_groups[group] - b_groups[group])
        if name in _CORE_TARGETS:
            core_deltas.append(a - b)
        if name in ('vol', 'trend_eff', 'range_expand'):
            desc_deltas.append(a - b)
    mean_core = float(np.mean(core_deltas)) if core_deltas else 0.0
    fwd_absmove_delta = float(res['fwd_absmove']['delta']) if 'fwd_absmove' in res else 0.0
    fwd_dir_delta = float(res['fwd_dir']['delta']) if 'fwd_dir' in res else 0.0
    per_group = {}
    for label, deltas in group_deltas.items():
        core = [deltas[name] for name in _CORE_TARGETS if name in deltas]
        per_group[label] = {'per_target_delta': deltas,
                            'mean_core_delta': float(np.mean(core)) if core else 0.0,
                            # Chronos-style task wins are bounded and remain interpretable when
                            # a tiny fast-screen slice has near-zero target variance (raw R2 can
                            # then explode to large negative values).
                            'core_target_win_rate': (float(np.mean(np.asarray(core) > 0.0))
                                                     if core else 0.0)}
    stream_values = [row['mean_core_delta'] for row in per_group.values()]
    stream_target_wins = [row['core_target_win_rate'] for row in per_group.values()]
    return {'per_target': res, 'mean_core_delta': mean_core,
            'descriptive_delta': float(np.mean(desc_deltas)) if desc_deltas else 0.0,
            'fwd_absmove_delta': fwd_absmove_delta,   # forward MOVE SIZE (R2) vs vanilla
            'fwd_dir_delta': fwd_dir_delta,           # forward DIRECTION (AUC) vs vanilla
            'forward_score': fwd_absmove_delta + fwd_dir_delta,   # combined forward relevance
            'per_stream': per_group,
            'stream_win_rate': (float(np.mean(np.asarray(stream_target_wins) >= 0.5))
                                if stream_target_wins else None),
            'average_target_win_rate': (float(np.mean(stream_target_wins))
                                        if stream_target_wins else None),
            'worst_stream_win_rate': (float(np.min(stream_target_wins))
                                      if stream_target_wins else None),
            # Retained as a diagnostic only; do not gate on this unbounded R2 delta.
            'worst_stream_delta': (float(np.min(stream_values)) if stream_values else None),
            'learns_regime_vol_structure': bool(mean_core > 0.0)}


def run_probe(big, starts, seq, ssl_ckpt, *, model_id='paris-noah/Mantis-8M',
              device=None, max_windows=20000, batch=512, seed=0, fwd_k=16, folds=1,
              group_names=None, verbose=True):
    """Extract SSL-adapted vs vanilla encoder embeddings for held-out windows and
    compare on the probe targets (regime/vol/structure + forward buy/sell move). Returns
    the compare() dict. folds>1 -> k-fold CV per probe (robust deltas for ranking candidates)."""
    from . import _ssl_torch
    used, groups = _balanced_probe_sample(starts, max_windows, seed)
    emb_ssl, used = _ssl_torch.embed_encoder(big, used, seq, ckpt=ssl_ckpt,
                                             model_id=model_id, device=device,
                                             batch=batch, max_windows=len(used), seed=seed)
    emb_van, _ = _ssl_torch.embed_encoder(big, used, seq, ckpt=None, model_id=model_id,
                                          device=device, batch=batch,
                                          max_windows=len(used), seed=seed)
    tgt = targets_from_windows(big, used, seq, fwd_k=fwd_k)
    split = _grouped_temporal_split(used, groups, purge=seq + fwd_k)
    out = compare(emb_ssl, emb_van, tgt, seed=seed, folds=folds, split=split,
                  groups=groups, group_names=group_names)
    out['split_schema'] = 'balanced_stream_purged_temporal_v1'
    out['probe_streams'] = int(len(np.unique(groups)))
    out['probe_windows'] = int(len(used))
    # Stable representation-scale measurement used when finalizing an already
    # completed REAL checkpoint without reconstructing its task-specific decoder.
    out['embedding_std'] = float(np.std(emb_ssl, axis=0).mean())
    if verbose:
        for name, d in out['per_target'].items():
            print(f"  [probe] {name:>12} ({d['kind']}) ssl={d['ssl']:.4f} "
                  f"vanilla={d['vanilla']:.4f} delta={d['delta']:+.4f}", flush=True)
        print(f"  [probe] mean_core_delta={out['mean_core_delta']:+.4f} "
              f"learns_regime_vol_structure={out['learns_regime_vol_structure']}", flush=True)
        print(f"  [probe] stream_win_rate={out['stream_win_rate']:.1%} "
              f"avg_target_win_rate={out['average_target_win_rate']:.1%} "
              f"worst_stream_win_rate={out['worst_stream_win_rate']:.1%} "
              f"streams={len(out['per_stream'])}", flush=True)
    return out
