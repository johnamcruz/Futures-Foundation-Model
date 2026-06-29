"""Mantis torch fine-tune — runs ONLY inside the subprocess worker.

PER-BATCH + memmap-friendly so we can train on FULL data without holding the feature
array in RAM/GPU: Xtr/Xval/Xeval may be in-RAM arrays OR disk memmaps; each batch is
indexed (paged from disk if memmap), standardized with the train mu/sd, and moved to
the device — so memory = one batch + the model, regardless of N. Standardize stats are
passed in (computed once on the train memmap) rather than materializing a standardized
copy. Partial FT (last K blocks + adapter + head), val early-stop, MPS cache clearing,
thread cap. torch loads here only (parent adapter is torch-free → no xgboost collision).
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn


def build_model(C, *, new_channels=10, ft_mode='partial', unfreeze_blocks=2,
                device='cpu', model_id='paris-noah/Mantis-8M'):
    """Mantis backbone + channel adapter + head with the freeze policy applied:
    'full' = all trainable; 'partial' = last `unfreeze_blocks` blocks + adapter + head;
    'head' = adapter + head only. Returns (model, new_c)."""
    from mantis.architecture import Mantis8M
    from mantis.adapters import LinearChannelCombiner
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    new_c = min(new_channels, C)
    net = Mantis8M.from_pretrained(model_id)
    adapter = LinearChannelCombiner(num_channels=C, new_num_channels=new_c)
    head = nn.Sequential(nn.LayerNorm(net.hidden_dim * new_c),
                         nn.Linear(net.hidden_dim * new_c, 2))
    model = FineTuningNetwork(net, head, adapter).to(device)
    if ft_mode in ('partial', 'head'):
        for p in net.parameters():
            p.requires_grad = False
    if ft_mode == 'partial':
        for blk in net.vit_unit.transformer.layers[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    return model, new_c


def export_model_onnx(model, C, seq, path, device='cpu'):
    """Export the fitted model to ONNX. Input = standardized window [batch, C, seq];
    output logits [batch, 2]. Serve path: build window via mv_contexts, standardize
    with the contract mu/sd, softmax(logits)[:,1]."""
    model.eval()
    dummy = torch.zeros(1, C, seq, device=device)
    torch.onnx.export(model, dummy, path, input_names=['window'], output_names=['logits'],
                      dynamic_axes={'window': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17)
    return path


def fit_predict_torch(Xtr, ytr, Xval, yval, Xeval, *, new_channels=10, ft_mode='partial',
                      unfreeze_blocks=2, epochs=40, batch=64, lr=3e-4, weight_decay=0.05,
                      patience=10, threads=2, device=None, model_id='paris-noah/Mantis-8M',
                      max_train=None, standardize_mu=None, standardize_sd=None,
                      export_onnx_path=None, seed=0, verbose=True):
    """Returns (p_val, p_eval, best_val_auc, best_epoch). Xtr/Xval/Xeval: arrays or
    memmaps. standardize_mu/sd: per-channel arrays applied per-batch (None = input
    already standardized)."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    torch.set_num_threads(int(threads))
    dev = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ytr = np.asarray(ytr).astype(np.int64)
    yval = np.asarray(yval).astype(int)
    n = len(Xtr)
    tr_rows = np.arange(n)
    if max_train and n > max_train:
        tr_rows = np.sort(rng.choice(n, max_train, replace=False))
    C = int(Xtr.shape[1]); seq = int(Xtr.shape[2])
    mu = None if standardize_mu is None else np.asarray(standardize_mu, np.float32)
    sd = None if standardize_sd is None else np.asarray(standardize_sd, np.float32)

    def std(xb):
        if mu is None:
            return xb
        return (xb - mu[None, :, None]) / sd[None, :, None]

    def to_dev(X, rows):
        xb = np.asarray(X[rows], np.float32)
        return torch.tensor(std(xb), device=dev)

    model, new_c = build_model(C, new_channels=new_channels, ft_mode=ft_mode,
                               unfreeze_blocks=unfreeze_blocks, device=dev, model_id=model_id)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def pred(M, X):
        M.eval(); out = []
        for s in range(0, len(X), batch):
            out.append(torch.softmax(M(to_dev(X, np.arange(s, min(s + batch, len(X))))), 1)
                       [:, 1].float().cpu().numpy())
        return np.concatenate(out) if len(X) else np.array([])

    best, best_state, bad, best_epoch = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = tr_rows[rng.permutation(len(tr_rows))]
        for s in range(0, len(perm), batch):
            bid = np.sort(perm[s:s + batch])               # sorted: faster memmap reads
            loss = crit(model(to_dev(Xtr, bid)),
                        torch.tensor(ytr[bid], dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if dev == 'mps':
            torch.mps.empty_cache()
        va = roc_auc_score(yval, pred(model, Xval)) if len(np.unique(yval)) == 2 else 0.5
        if va > best + 1e-4:
            best, bad, best_epoch = va, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"    ep{ep:>2} val_auc={va:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    if export_onnx_path:
        export_model_onnx(model, C, seq, export_onnx_path, device=dev)
    return pred(model, Xval), pred(model, Xeval), float(best), int(best_epoch)
