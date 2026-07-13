"""Exhaustion risk head — produce-time generation (part of the ONE-RUN bundle).

THE COHERENT-BUNDLE RULE (from the 2026-07-12 checkpoint-mismatch incident): every head shipped
in a bundle must be fit on embeddings from the SAME checkpoint the bundle's encoder was exported
from. The incident: a risk head fit out-of-session on a stale checkpoint copy paired with a
deployed encoder from different weights — both outputs biased (down +0.14), an RL exploited it,
a combine replay blew. This module makes coherence STRUCTURAL: the strategy's produce calls
`produce_exhaustion_head(...)` with the same `backbone_ckpt` it just exported the encoder from,
in the same run — the risk head is never a separate step again.

WHAT THE HEAD IS (validated 2026-07-10/11, colabs exhaustion_rl_scan + per-bar validation):
per-direction P(the ridden trend structurally ENDS here) — Dow HH/HL-vs-LH/LL lifecycle ground
truth — fit as logistic-on-frozen-embedding per direction on 'continue' pivots (train <2025),
Platt-calibrated on 2025, folded (scaler+logistic+Platt -> ONE affine->Sigmoid per direction,
exact algebra) and exported as a tiny ONNX beside the entry head:
    model_risk_head.onnx : emb [None, E] float32 -> p_end_up [None,1], p_end_down [None,1]
    model_risk.json      : usage contract + Platt + per-side eval stats + BACKBONE FINGERPRINT

The ground-truth labels npz is built by the strategy side (Dow-structure lifecycle labeler —
strategy IP, not in this repo); this module only requires its portable schema:
    ticker, tf, confirm, ts, trend_dir, is_start, ended, kind   (one row per confirmed pivot)

Torch is imported inside functions only (repo contract: torch-free module import).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

SEQ = 64
ACTIVATION = 0.70
TRAIN_END = '2025-01-01'
EVAL_END = '2026-01-01'
CAP = 400_000


def fold_affine(sc, clf, platt):
    """StandardScaler -> LogisticRegression -> Platt(on logit) folded to ONE (W, b):
    p = sigmoid(W.x + b). Exact algebra, no approximation."""
    w = clf.coef_.ravel() / sc.scale_
    c = float(clf.intercept_[0] - np.dot(clf.coef_.ravel(), sc.mean_ / sc.scale_))
    A, B = platt
    return (A * w).astype(np.float32), np.float32(A * c + B)


def export_risk_onnx(folded, n_features, path):
    """Folded per-direction heads -> one ONNX graph: emb [None,n] -> one Sigmoid output per head."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes, inits = [], []
    for name, (W, b) in folded.items():
        inits.append(numpy_helper.from_array(W.reshape(-1, 1), f'{name}_W'))
        inits.append(numpy_helper.from_array(np.array([[b]], np.float32), f'{name}_b'))
        nodes += [helper.make_node('MatMul', ['emb', f'{name}_W'], [f'{name}_mm']),
                  helper.make_node('Add', [f'{name}_mm', f'{name}_b'], [f'{name}_z']),
                  helper.make_node('Sigmoid', [f'{name}_z'], [name])]
    graph = helper.make_graph(
        nodes, 'exhaustion_risk_head',
        [helper.make_tensor_value_info('emb', TensorProto.FLOAT, [None, int(n_features)])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [None, 1]) for n in folded], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def _embed_labeled_pivots(d, ckpt, out_path, data_dir, verbose=True):
    """Per-stream fp16 disk-memmap embed of every labeled pivot's [5,SEQ] window for the GIVEN
    checkpoint (RAM = one stream; cuda/mps/cpu). Returns keep mask (row-aligned with the labels)."""
    from futures_foundation.finetune.pretext._torch.common import embed_windows
    import torch
    dev = ('cuda' if torch.cuda.is_available()
           else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tickers = d['ticker']; tfs = d['tf']; confirm = d['confirm']
    n = len(tickers)
    keep = np.zeros(n, bool)
    emb_out = None
    for tk in np.unique(tickers):
        for tf in np.unique(tfs):
            idx = np.where((tickers == tk) & (tfs == tf))[0]
            if not len(idx):
                continue
            csv = os.path.join(data_dir, f'{tk}_{tf}.csv')
            df = pd.read_csv(csv, usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df = df[df['datetime'] < pd.Timestamp(EVAL_END, tz='UTC')].reset_index(drop=True)
            arr = np.stack([df[k].to_numpy(np.float32) for k in
                            ('open', 'high', 'low', 'close', 'volume')])
            valid = idx[confirm[idx] + 1 >= SEQ]
            if not len(valid):
                continue
            win = np.empty((len(valid), 5, SEQ), np.float32)
            for j, i in enumerate(valid):
                cf = confirm[i]
                win[j] = arr[:, cf - SEQ + 1:cf + 1]
            del arr
            stream_emb = embed_windows(win, ckpt=ckpt, device=dev, batch=512)
            del win
            if emb_out is None:
                emb_out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16,
                                                    shape=(n, stream_emb.shape[1]))
            emb_out[valid] = stream_emb.astype(np.float16)
            keep[valid] = True
            if verbose:
                print(f'  [risk-embed] {tk}@{tf}: {len(valid)}/{len(idx)}', flush=True)
    if emb_out is None:
        raise RuntimeError('no labeled pivots embeddable — check labels npz vs data_dir')
    emb_out.flush()
    return keep


def produce_exhaustion_head(*, backbone_ckpt, labels_npz, out_dir, data_dir,
                            cache_dir=None, activation=ACTIVATION, verbose=True):
    """Fit + calibrate + export the exhaustion risk head for the GIVEN checkpoint, into out_dir.

    Called by a strategy's produce right after its encoder/entry-head export, with the SAME
    backbone_ckpt — the coherent-bundle rule enforced by construction. Reuses (or builds) a
    per-checkpoint embedding cache under cache_dir. Returns the contract dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from futures_foundation.finetune.calibration import fit_platt, apply_platt

    cache_dir = cache_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    ck_sha = hashlib.sha256(open(backbone_ckpt, 'rb').read()).hexdigest()
    emb_cache = os.path.join(cache_dir, f'trend_probe_emb_{os.path.basename(backbone_ckpt)}.npy')
    keep_cache = os.path.join(cache_dir, f'trend_probe_keep_{os.path.basename(backbone_ckpt)}.npy')
    if verbose:
        print(f'[risk-head] backbone={os.path.basename(backbone_ckpt)} sha={ck_sha[:16]}... '
              f'labels={labels_npz}', flush=True)

    d = np.load(labels_npz, allow_pickle=True)
    if not (os.path.exists(emb_cache) and os.path.exists(keep_cache)):
        if verbose:
            print(f'[risk-head] embedding labeled pivots for THIS checkpoint (cache miss)...',
                  flush=True)
        keep = _embed_labeled_pivots(d, backbone_ckpt, emb_cache, data_dir, verbose=verbose)
        np.save(keep_cache, keep)
    emb = np.load(emb_cache, mmap_mode='r')
    keep = np.load(keep_cache)

    ts = pd.to_datetime(d['ts'], utc=True)
    kind = d['kind']; is_start = d['is_start'].astype(bool)
    ended = d['ended'].astype(bool); tdir = d['trend_dir'].astype(int)
    is_cont = (~is_start) & (kind != 0)
    tr = (ts < pd.Timestamp(TRAIN_END, tz='UTC')) & keep & is_cont
    ev = ((ts >= pd.Timestamp(TRAIN_END, tz='UTC')) & (ts < pd.Timestamp(EVAL_END, tz='UTC'))
          & keep & is_cont)

    rng = np.random.RandomState(0)
    folded, report = {}, {}
    for dd, name in ((1, 'p_end_up'), (-1, 'p_end_down')):
        itr = np.where(tr & (tdir == dd))[0]
        if len(itr) > CAP:
            itr = np.sort(rng.choice(itr, CAP, replace=False))
        Xtr = emb[itr].astype(np.float32)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ended[itr].astype(int))
        iev = np.where(ev & (tdir == dd))[0]
        Xev = emb[iev].astype(np.float32)
        raw = clf.predict_proba(sc.transform(Xev))[:, 1]
        platt = fit_platt(raw, ended[iev].astype(int))
        cal = apply_platt(raw, platt)
        auc = roc_auc_score(ended[iev], cal)
        W, b = fold_affine(sc, clf, platt)
        p_fold = 1.0 / (1.0 + np.exp(-(Xev @ W + b)))
        parity = float(np.max(np.abs(p_fold - cal)))
        assert parity < 1e-5, f'{name}: fold parity failed ({parity:.2e})'
        folded[name] = (W, b)
        report[name] = dict(train_n=int(len(itr)), eval_n=int(len(iev)),
                            eval_auc=round(float(auc), 4),
                            platt=[float(platt[0]), float(platt[1])], fold_parity=parity,
                            eval_fire_rate_at_activation=round(float((cal >= activation).mean()), 4))
        if verbose:
            print(f'  [{name}] train n={len(itr):,}  eval AUC={auc:.4f}  '
                  f'fold-parity={parity:.2e}  fires>={activation}: '
                  f'{(cal >= activation).mean()*100:.1f}% of continues', flush=True)

    onnx_path = export_risk_onnx(folded, emb.shape[1], os.path.join(out_dir, 'model_risk_head.onnx'))
    try:                                                   # prove the bot's actual runtime path
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        iev = np.where(ev & (tdir == 1))[0][:2048]
        Xev = emb[iev].astype(np.float32)
        got = sess.run(['p_end_up'], {'emb': Xev})[0].ravel()
        W, b = folded['p_end_up']
        want = 1.0 / (1.0 + np.exp(-(Xev @ W + b)))
        ort_parity = float(np.max(np.abs(got - want)))
        assert ort_parity < 1e-5, f'ORT parity failed ({ort_parity:.2e})'
        if verbose:
            print(f'  [onnx] ORT parity max|diff|={ort_parity:.2e}', flush=True)
    except ImportError:
        ort_parity = None

    contract = {
        'name': 'exhaustion_risk_head',
        'pairs_with': 'the entry bundle produced in the SAME run (same backbone checkpoint)',
        'backbone_ckpt': os.path.basename(backbone_ckpt),
        'backbone_sha256': ck_sha,                          # THE coherence fingerprint — the bot
        'backbone_bytes': os.path.getsize(backbone_ckpt),   # verifies its encoder came from this
        'input': {'emb': f'[None, {emb.shape[1]}] float32 — the model_encoder.onnx output'},
        'outputs': {'p_end_up': 'P(the UP-trend structurally ends at this pivot/bar)',
                    'p_end_down': 'P(the DOWN-trend structurally ends at this pivot/bar)'},
        'usage': {
            'query_at': 'any CLOSED bar while a trade is open (per-bar validated 2026-07-11)',
            'pick_output_by': 'the trend direction being ridden (long in an uptrend -> p_end_up)',
            'activation_rule': f'proba >= {activation} -> exit or tighten the trail (bot-side '
                               'A/B on 2026 decides the policy AND re-validates the threshold '
                               'for THIS fit)',
            'hard_signal': 'a confirmed OPPOSITE-structure pivot = the break HAPPENED — exit '
                           'unconditionally, no proba needed',
        },
        'training': {'label': 'structural trend-end among continue-pivots (Dow lifecycle)',
                     'window': f'train <{TRAIN_END}, Platt-calibrated on '
                               f'[{TRAIN_END}, {EVAL_END})'},
        'heads': report,
        'ort_parity': ort_parity,
        'content_sha256': hashlib.sha256(open(onnx_path, 'rb').read()).hexdigest(),
    }
    with open(os.path.join(out_dir, 'model_risk.json'), 'w') as f:
        json.dump(contract, f, indent=2)
    if verbose:
        print(f'[risk-head] artifacts -> {out_dir}/model_risk_head.onnx + model_risk.json  '
              f'(content sha {contract["content_sha256"][:16]}...)', flush=True)
    return contract
