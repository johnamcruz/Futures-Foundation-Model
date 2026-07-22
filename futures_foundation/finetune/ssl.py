"""Masked-modeling SSL pretraining of the Mantis backbone (orchestrator) — "BERT for
futures".

BERT-style masked modeling on raw OHLCV across the 9 futures tickers x {1,3,5,15}min:
mask a fraction of bars and reconstruct them from context (in _ssl_torch.train_ssl_mask),
so the encoder learns regime/volatility/structure. Produces an ADAPTED ENCODER CHECKPOINT
(saved to Drive on Colab) that downstream classifier finetuning starts from
(build_model(..., backbone_ckpt=...) / BACKBONE_CKPT=... in the WF/produce driver).

Torch-free at import (the GPU trainer in _ssl_torch + the probe's torch bits load lazily)
so data assembly, the generalization gate, the Optuna search wiring, and the contract are
testable without the torch/mantis stack.

Generalization is PROBE-GATED + OPTUNA-TUNED, mirroring WF/produce:
  * TIME-SPLIT val reconstruction early-stop  (generalize forward; 2026 EXCLUDED)
  * GATE = a linear PROBE shows the frozen embedding predicts regime / vol / structure +
    forward buy/sell move BETTER than vanilla Mantis (the classification-relevance test)
  * if it doesn't pass -> OPTUNA tunes lr/reg/capacity/mask_ratio to MAXIMIZE the probe
  * REAL vs SHUFFLE vs RANDOM = probe-based diagnostic (did temporal order contribute)

Colab usage: see colab/mantis_ssl_pretrain.py.
"""
import argparse
import hashlib
import json
import os

import numpy as np

from . import ssl_data
from .model_identity import evaluation_environment
from .pretext import PRETEXTS, PretextTask, get_pretext   # noqa: F401 (pluggable pretext registry)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def assemble(streams, *, seq, max_jitter, val_frac, holdout_start, forecast_parent=0,
             sampling_mode='bar_proportional', verbose=True, return_related_layout=False):
    """Concatenate all stream OHLCV into one big [T, 5] array + global parent-window start
    positions for the (leak-safe, 2026-excluded) train/val split. Each window reserves enough
    bars for its consumers: seq+max_jitter (probe/mask) OR forecast_parent (stage-2 = max context
    length + max horizon), whichever is larger — so context+future stay in-stream.

    ``bar_proportional`` returns the historical flat training-start array. Opt-in
    ``uniform_stream`` attaches compact source ranges so the trainer chooses a non-empty
    ticker/timeframe stream uniformly before choosing a legal window from that stream.
    Validation starts always remain a flat chronological array.
    """
    if sampling_mode not in ssl_data.SAMPLING_MODES:
        raise ValueError(f'unsupported sampling_mode={sampling_mode!r}; '
                         f'expected one of {ssl_data.SAMPLING_MODES}')
    parent_len = max(seq + max_jitter, int(forecast_parent))
    bigs, tr_starts, tr_groups, train_stream_ids = [], [], [], []
    va_starts, layout_records, base = [], [], 0
    for s in streams:
        oh = s['ohlcv']
        tr_idx, va_idx = ssl_data.time_split(s['ts'], val_frac, holdout_start)
        # time_split already prevents train/val windows from referencing the holdout. Also
        # truncate the resident array at that boundary so held-out bars cannot enter training
        # through a future sampler/control change and do not consume accelerator memory. Since
        # load_ohlcv sorts timestamps, the usable train+val region is always a causal prefix.
        usable_end = (int(va_idx[-1]) + 1 if len(va_idx)
                      else int(tr_idx[-1]) + 1 if len(tr_idx) else 0)
        oh = oh[:usable_end]
        ts = ssl_data.window_starts(tr_idx, parent_len)
        vs = ssl_data.window_starts(va_idx, parent_len)
        if len(ts):
            tr_starts.append(ts + base)
            tr_groups.append(np.full(len(ts), len(train_stream_ids), dtype=np.int32))
            train_stream_ids.append(s['sid'])
        if len(vs):
            va_starts.append(vs + base)
        layout_records.append({**{k: s[k] for k in ('sid', 'ticker', 'tf')},
                               'base': base, 'ts': s['ts'][:usable_end]})
        bigs.append(oh)
        base += len(oh)
        if verbose:
            print(f"  [assemble] {s['sid']} train_win={len(ts)} val_win={len(vs)}",
                  flush=True)
    big = np.concatenate(bigs, 0).astype(np.float32)
    tr_values = np.concatenate(tr_starts) if tr_starts else np.array([], np.int64)
    if sampling_mode == 'uniform_stream':
        groups = np.concatenate(tr_groups) if tr_groups else np.array([], np.int32)
        tr = ssl_data.WindowStartPool(tr_values, groups, tuple(train_stream_ids))
    else:
        tr = tr_values
    va = np.concatenate(va_starts) if va_starts else np.array([], np.int64)
    if return_related_layout:
        from .related_series import RelatedSeriesLayout
        return big, tr, va, RelatedSeriesLayout.from_assembled(layout_records)
    return big, tr, va


# --------------------------------------------------------------------------- train + probe
# Pretext tasks (Mask / Forecast / Contrastive) live in the pluggable `pretext` package —
# futures_foundation/finetune/pretext/. Add a new pretrain experiment there, not here. The
# orchestrator only resolves a task via get_pretext(...) and calls reserve/train/gate/finalize.
def _train(big, tr, va, cfg, control='real'):
    """Train one config under a control via its pretext task -> (best_encoder_state, history)."""
    return get_pretext(cfg.get('pretext', 'mask')).train(big, tr, va, cfg, control)


def _probe_state(big, va, seq, state, *, model_id, device, seed, folds=1,
                 group_names=None, verbose=True):
    """Probe a trained encoder state vs vanilla -> the probe dict (regime/vol/structure).
    Saves to a temp ckpt so ssl_probe can load it through the normal path. folds>1 -> k-fold CV
    per probe (robust deltas for ranking candidates)."""
    import tempfile
    import torch
    from . import ssl_probe
    fd, tmp = tempfile.mkstemp(suffix='.pt'); os.close(fd)
    from .pretext._torch.related_series import plain_encoder_state
    torch.save(plain_encoder_state(state), tmp)
    try:
        return ssl_probe.run_probe(big, va, seq, tmp, model_id=model_id, device=device,
                                   seed=seed, folds=folds, group_names=group_names,
                                   verbose=verbose)
    finally:
        os.remove(tmp)


def _passes(probe_res, std, margin=0.0, dir_margin=0.0, pretext='mask',
            forecast_skill=None):
    """Report-only gate on the PROBE (representation content), delegated to the pretext task.
    Each task (MaskTask/ForecastTask/ContrastiveTask) owns its own pass/fail rule — see
    PretextTask.gate + `_decide`. Kept as a thin function for callers/tests."""
    return get_pretext(pretext).gate(
        probe_res, std, margin, dir_margin, forecast_skill=forecast_skill)


def _control_cfg(cfg):
    """Bound diagnostic controls independently from REAL optimization."""
    out = dict(cfg)
    out['epochs'] = min(int(cfg['epochs']), int(cfg.get('control_epochs', 8)))
    out['patience'] = min(int(cfg['patience']), out['epochs'])
    out['resume'] = False
    return out


def _selected_history_row(history):
    """Return metrics for the encoder that was actually checkpointed.

    The minimum observed validation loss is not always saved: a collapse guard
    may reject that epoch. New trainers mark the selected row explicitly; old
    histories fall back to the historical minimum for compatibility.
    """
    selected = [row for row in history if row.get('checkpoint_selected')]
    return selected[-1] if selected else min(history, key=lambda row: row['val_loss'])


def revalidate_saved_report(report_path):
    """Reapply the current task gate to a completed checkpoint report without retraining.

    This is intentionally limited to metrics already frozen in the report. It cannot turn a bad
    training run into a good one; it only lets evaluation-policy fixes recover a fully trained,
    crash-safe checkpoint. The caller still runs the independent Probe Atlas retention gate.
    """
    report_path = os.fspath(report_path)
    report = json.load(open(report_path))
    cfg = report.get('config', {})
    pretext = cfg.get('pretext', 'mask')
    history = report.get('history') or []
    if not history:
        return report.get('verdict', {})
    summary = history[0]
    fc_skill = summary.get('forecast_skill')
    ok, detail = _passes(
        report.get('probe'), float(summary.get('std', 0.0)), pretext=pretext,
        forecast_skill=fc_skill)
    summary.update({'gate_ok': bool(ok), **detail})
    verdict = dict(report.get('verdict', {}))
    verdict['representation_pass'] = bool(ok)
    verdict['all_pass'] = bool(ok and verdict.get('beats_controls', False))
    get_pretext(pretext).finalize_verdict(verdict, fc_skill, report.get('probe'))
    report['verdict'] = verdict
    tmp = report_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    os.replace(tmp, report_path)
    return verdict


def _stream_names(streams):
    """Stable probe labels for production streams and minimal test fixtures."""
    return [s.get('sid', f"{s['ticker']}@{s['tf']}") for s in streams]


# ------------------------------------------------------------------------------- save/probe
def _finalize(big, tr, va, state, probe_res, cfg, *, out_path, controls, holdout_start,
              val_frac, streams, history, verbose):
    """Save the chosen encoder + report and evaluate objective-aware corruption controls.

    Representation stages compare Probe Atlas lift. Forecasting tasks can compare their own
    held-out temporal targets so inherited parent knowledge cannot make shuffled adapters appear
    equivalent to REAL. The independent representation gate remains mandatory for every task.
    """
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    torch.save(state, out_path)                          # adapted ENCODER state_dict
    with open(out_path + '.real_complete.json', 'w') as marker:
        json.dump({
            'schema': 'ffm_ssl_real_complete_v1',
            'checkpoint_sha256': _file_sha256(out_path),
            'best_val': float(history[0]['best_val']),
            'embedding_std': float(history[0]['std']),
        }, marker, indent=2)

    task = get_pretext(cfg.get('pretext', 'mask'))
    ctrl_delta = {}
    ctrl_task = {}
    for ctrl in controls:
        if ctrl == 'real':
            continue
        if verbose:
            print(f"\n=== control={ctrl} (probe-based diagnostic) ===", flush=True)
        ctrl_cfg = _control_cfg(cfg)
        if verbose:
            print(f"  [control-budget] {ctrl}: max_epochs={ctrl_cfg['epochs']} "
                  f"patience={ctrl_cfg['patience']}", flush=True)
        st, ctrl_history = _train(big, tr, va, ctrl_cfg, ctrl)
        r = _probe_state(big, va, cfg['seq'], st, model_id=cfg['model_id'],
                         device=cfg['device'], seed=cfg['seed'],
                         folds=cfg.get('probe_folds', 1),
                         group_names=_stream_names(streams), verbose=verbose)
        ctrl_delta[ctrl] = float(r['mean_core_delta'])
        ctrl_best = _selected_history_row(ctrl_history) if ctrl_history else {}
        ctrl_task[ctrl] = task.control_evidence(ctrl_best, r)

    real_delta = (None if probe_res is None else float(probe_res['mean_core_delta']))
    real_task = task.control_evidence(history[0] if history else {}, probe_res)
    beats_controls, task_margins, temporal = task.compare_control_evidence(
        real_task, ctrl_task)
    representation_pass = bool(history and history[0].get('gate_ok', False))
    task_control = {
        'contract': task.control_contract,
        'real': real_task,
        'controls': ctrl_task,
        'margins': task_margins,
        'beats_controls': beats_controls,
    }
    verdict = {
        # Fail closed: the task-specific representation gate must pass and REAL temporal
        # learning must beat every corrupted-input control. The old verdict only checked that
        # REAL beat vanilla, allowing checkpoints that lost to shuffle/random to advance.
        'all_pass': bool(representation_pass and beats_controls),
        'representation_pass': representation_pass,
        'beats_controls': beats_controls,
        'learns_regime_vol_structure': (None if probe_res is None
                                        else bool(probe_res['learns_regime_vol_structure'])),
        'real_delta': real_delta,
        'control_delta': ctrl_delta,
        'task_control': task_control,
        'temporal_signal': temporal,
    }
    report = {'verdict': verdict, 'probe': probe_res, 'control_delta': ctrl_delta,
              'task_control': task_control,
              'evaluation_environment': evaluation_environment(),
              'config': {k: cfg[k] for k in cfg if not k.startswith('_') and k not in
                         ('verbose', 'device', 'model_id', 'compile_model')},
              'holdout_start': holdout_start, 'val_frac': val_frac, 'bars': int(len(big)),
              'tickers': sorted({s['ticker'] for s in streams}),
              'tfs': sorted({s['tf'] for s in streams}), 'history': history, 'ckpt': out_path}
    with open(out_path + '.report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    if verbose:
        print(f"\n[ssl] saved encoder -> {out_path}\n[ssl] VERDICT: {verdict}", flush=True)
    return verdict


# ------------------------------------------------------------------------------- entrypoints
def _load_assemble(data_dir, tickers, tfs, seq, max_jitter, val_frac, holdout_start, verbose,
                   forecast_parent=0, sampling_mode='bar_proportional', related=False):
    streams = ssl_data.load_ohlcv(data_dir, tickers, tfs, verbose=verbose)
    assembled = assemble(streams, seq=seq, max_jitter=max_jitter, val_frac=val_frac,
                         holdout_start=holdout_start, forecast_parent=forecast_parent,
                         sampling_mode=sampling_mode, verbose=verbose,
                         return_related_layout=related)
    big, tr, va = assembled[:3]
    if verbose:
        print(f"[ssl] bars={len(big)} train_win={len(tr)} val_win={len(va)} "
              f"streams={len(streams)} sampling={sampling_mode}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("no train/val windows — check seq/max_jitter vs data length")
    return (streams, big, tr, va, assembled[3]) if related else (streams, big, tr, va)


def _base_cfg(**kw):
    """Default SSL config (one place). seq = the probe/embed window; max_jitter reserves the
    probe's forward horizon. Stage-2 forecast knobs: horizons (multi-horizon candle prediction),
    context_lengths (variable input). Only known keys are kept."""
    d = dict(seq=64, max_jitter=16, new_channels=8, mask_ratio=0.4, epochs=60,
             steps_per_epoch=200, batch=1024, lr=1e-4, weight_decay=0.05, patience=8,
             control_epochs=8,
             model_id='paris-noah/Mantis-8M', compile_model=False, device=None,
             seed=0, verbose=True, backbone_ckpt=None,
             sampling_mode='bar_proportional',
             pretext='mask',                                  # 'mask' (1) | 'forecast' (2) | 'forecast_dist' (2.5) | 'contrastive' (3)
             # mask SpanBERT mode (shared by the mask pretext): span_mean>0 = corrupt CONTIGUOUS
             # multi-bar spans (geometric mean span_mean, clipped span_max); 0 = single-bar masking.
             span_mean=0.0, span_max=10,
             # stage-2 multi-horizon / variable-context candle forecasting:
             horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
             grad_clip=1.0, clamp=10.0,
             # forecast supervision OBJECTIVE (pluggable, no if-chains): 'candle_mse' (original) |
             # 'candle_direction' (candle MSE + BCE on sign(fwd close move) via dir_weight). The Optuna
             # sweep searches this + the knobs below to maximize downstream WR.
             objective='candle_mse',
             # OPTIONAL forecast direction-head squeeze (0 = off / backward-compat; >0 adds BCE on
             # sign of the forward close move -> trains the encoder to be direction-aware for WR):
             dir_weight=0.0, dir_close_ch=3,
             # stage-2.5 forecast_dist faithfulness knobs (defaults = the original refine-study
             # behavior): mse_weight 0 = PURE Chronos loss (no MSE anchor); quantile_taus 'bolt9'
             # = the full 9-level quantile head; bins_k = bin-classification resolution.
             mse_weight=1.0, quantile_taus='lohi', bins_k=41, balance_w=0.02,
             # Contrastive regime geometry. `kaufman` uses only the completed context's
             # efficiency ratio and direction; `temporal` preserves the historical offset-based
             # ablation. Both corrupt inputs only for honest shuffle/random controls.
             temperature=0.1, crop_max=0.2, proj_dim=128,
             pos_deltas=(2, 16, 64), far_min=512, aug_noise=0.10, aug_scale=0.20,
             aug_tmask=0.15, vol_weight=1.0, w_clip=4.0, metrics_n=768,
             # Optional causal regime key. Kaufman ER uses only the completed context window;
             # the transition band remains instance-only instead of receiving a hard label.
             regime_key='temporal', kaufman_chop=0.25, kaufman_trend=0.50,
             # stage-4 TURN-ELECTRA (replaced-TURN detection — the discriminative slot): spans are
             # CENTERED ON DETECTED TURNS (local swing highs/lows, neighborhood ±turn_w) with prob
             # turn_bias (0 = uniform span-ELECTRA, the placement ablation); a weak generator
             # (gen_width) fills each masked turn = a SYNTHETIC FAKE TURN; the encoder labels every
             # bar real/replaced (rtd_weight) while the encoder-side recon anchor (recon_weight)
             # keeps the embedding tied to the data (0 = pure discrimination / drift risk).
             # span_mean/span_max (shared above) set span lengths; electra coerces span_mean<=0 to 4.
             rtd_weight=5.0, recon_weight=1.0, gen_width=48, turn_w=3, turn_bias=0.85,
             # stage-2.6 NEXT-LEG forecasting (bars; pure-fractal pivots, NO ATR):
             leg_cap=256, leg_w=1.0, leg_k=2,
             # stage-2.8 NEXT-LEG-RACE (future-only ordered candle path; no ATR/R/strategy data):
             race_w=0.25, race_cap=2.0, race_levels=(0.25, 0.50, 0.75, 1.00),
             # std_guard: IN-LOOP drift halt — training stops (without saving that epoch)
             # the moment emb_std exceeds it; 0 = off. Guards the anchored-discrimination
             # runs against slow drift that val loss rewards (val micro-improves while the
             # representation walks off the data).
             std_guard=1.6,
             # crash-safe progressive best-save + resume + anti-forgetting layer-freeze (ALL pretexts,
             # real run only; controls never touch the ckpt). ckpt_path is set to out_path by loop_ssl.
             ckpt_path=None, resume=False, freeze_encoder_layers=0,
             lora_r=0, lora_alpha=16.0, lora_dropout=0.0,
             log_every_steps=25,
             # Chronos-2-inspired, Mantis-native related-series experiment. These settings are
             # inert unless pretext='related_nextleg'. Timestamps are causally aligned by bar CLOSE.
             related_tfs=('1min', '3min', '5min', '15min'),
             related_siblings='default', related_heads=4, related_dropout=0.0,
             related_max_gap_factor=2.0, related_control='real',
             probe_folds=1)                                   # k-fold CV per probe (robust)
    d.update({k: v for k, v in kw.items() if v is not None and k in d})
    return d


def loop_ssl(data_dir=None, *, tickers=None, tfs=None, controls=('shuffle', 'random'),
             out_path='mantis_ssl_ohlcv.pt', probe=True, probe_margin=0.0, dir_margin=0.0,
             holdout_start='2026-01-01', val_frac=0.1,
             reuse_real_checkpoint=False, **cfg_over):
    """Train the SSL encoder ONCE and save it (no Optuna). pretext='mask' = stage-1 masked
    modeling; pretext='forecast' = stage-2 multi-horizon / variable-context candle seq2seq
    (warm-started from stage-1 via backbone_ckpt). Then PROBE vs vanilla + shuffle/random controls
    as diagnostics (gate = report-only), and write the encoder + report."""
    cfg = _base_cfg(**cfg_over)
    if controls and int(cfg.get('control_epochs', 8)) < 1:
        raise ValueError('control_epochs must be >= 1 when controls are enabled')
    cfg['ckpt_path'] = out_path              # progressive best-save target (crash-safe); real run only
    verbose = cfg['verbose']
    pretext = cfg.get('pretext', 'mask')
    # each pretext task declares how much window to reserve (forecast: ctx+horizon;
    # contrastive: ctx; mask: none) — no pretext if-chain here.
    task = get_pretext(pretext)
    fc_reserve = task.reserve(cfg)
    # Trainers may gather a much shorter GPU tensor than the split-safety
    # window. Preserve the actual legal-window reserve so target-side leak
    # guards validate the split contract, not the batch allocation width.
    cfg['target_reserve'] = fc_reserve
    loaded = _load_assemble(data_dir, tickers, tfs, cfg['seq'], cfg['max_jitter'],
                            val_frac, holdout_start, verbose, forecast_parent=fc_reserve,
                            sampling_mode=cfg.get('sampling_mode', 'bar_proportional'),
                            related=task.requires_related_series)
    if task.requires_related_series:
        streams, big, tr, va, cfg['_related_layout'] = loaded
    else:
        streams, big, tr, va = loaded
    if reuse_real_checkpoint:
        if not os.path.isfile(out_path):
            raise FileNotFoundError(
                f'cannot reuse REAL checkpoint; file missing: {out_path}')
        if not probe:
            raise ValueError('checkpoint-only finalization requires probe=True')
        marker_path = out_path + '.real_complete.json'
        if os.path.isfile(marker_path):
            marker = json.loads(open(marker_path).read())
            if marker.get('checkpoint_sha256') != _file_sha256(out_path):
                raise RuntimeError(
                    f'REAL-complete marker does not match checkpoint: {out_path}')
        elif verbose:
            print('[ssl] WARNING: explicit recovery of a pre-marker checkpoint; '
                  'the representation probe will revalidate it', flush=True)
        import torch
        state = torch.load(out_path, map_location='cpu')
        meta_path = out_path + '.meta.json'
        meta = json.loads(open(meta_path).read()) if os.path.isfile(meta_path) else {}
        recovered_best = {}
        trainer_path = out_path + '.trainer.pt'
        if os.path.isfile(trainer_path):
            trainer_payload = torch.load(trainer_path, map_location='cpu')
            recovered_best = dict(trainer_payload.get('best_history_row') or {})
        hist = [{**recovered_best,
            'epoch': int(meta.get('epoch', -1)),
            'train_loss': None,
            'val_loss': float(meta.get('best_val', float('nan'))),
            'resumed_finalization_only': True,
        }]
        if verbose:
            print(f"[ssl] reusing saved REAL checkpoint -> {out_path}; "
                  "REAL optimization skipped", flush=True)
    else:
        state, hist = _train(big, tr, va, cfg, 'real')
    probe_res = (_probe_state(big, va, cfg['seq'], state, model_id=cfg['model_id'],
                              device=cfg['device'], seed=cfg['seed'],
                              folds=cfg.get('probe_folds', 1),
                              group_names=_stream_names(streams),
                              verbose=verbose) if probe else None)
    if reuse_real_checkpoint:
        std = float(probe_res['embedding_std'])
        hist[0]['std'] = std
    best_ep = _selected_history_row(hist)
    if not reuse_real_checkpoint:
        # Bind the gate to the exact encoder returned/saved by the trainer, not
        # the final attempted epoch (which may be worse or collapse-guarded).
        std = float(best_ep['std'])
    fc_skill = best_ep.get('skill')                       # forecast skill vs copy-now (None for mask)
    ok, detail = _passes(probe_res, std, probe_margin, dir_margin, pretext,
                         forecast_skill=fc_skill)
    task_metrics = {key: value for key, value in best_ep.items()
                    if key not in ('epoch', 'train_loss', 'val_loss', 'std',
                                   'checkpoint_selected')}
    history = [{'source': 'default', 'best_epoch': int(best_ep.get('epoch', -1)),
                'best_val': float(best_ep['val_loss']), 'std': std,
                'forecast_skill': fc_skill, 'gate_ok': bool(ok),
                **task_metrics, **detail}]
    verdict = _finalize(big, tr, va, state, probe_res, cfg, out_path=out_path, controls=controls,
                        holdout_start=holdout_start, val_frac=val_frac, streams=streams,
                        history=history, verbose=verbose)
    verdict['history'] = history
    verdict['epochs'] = hist                 # per-epoch trainer history (val_loss + task extras,
    #                                          e.g. electra rtd_bal_acc) — learning verification
    get_pretext(pretext).finalize_verdict(verdict, fc_skill, probe_res)   # pretext-specific fields
    # _finalize writes the core report before task-specific fields are attached. Persist the final
    # verdict atomically so a restarted pipeline sees the same decision returned to this process.
    report_path = out_path + '.report.json'
    if os.path.isfile(report_path):
        report = json.load(open(report_path))
        # `history`/`epochs` are returned to the caller for live diagnostics; the report already
        # stores its compact validation history and must not duplicate the full epoch trace here.
        report['verdict'] = {
            key: value for key, value in verdict.items() if key not in ('history', 'epochs')}
        tmp_report = report_path + '.tmp'
        with open(tmp_report, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        os.replace(tmp_report, report_path)
    return verdict


def run_ssl(data_dir=None, *, controls=('shuffle', 'random'),
            out_path='mantis_ssl_ohlcv.pt', probe=True, holdout_start='2026-01-01',
            val_frac=0.1, tickers=None, tfs=None, **cfg_over):
    """Thin alias of loop_ssl (kept for callers/tests). loop_ssl trains once and saves."""
    return loop_ssl(data_dir, tickers=tickers, tfs=tfs, controls=controls, out_path=out_path,
                    probe=probe, holdout_start=holdout_start, val_frac=val_frac, **cfg_over)


def main():
    p = argparse.ArgumentParser(description="Mantis OHLCV SSL — masked (stage 1) or multi-horizon seq2seq (stage 2)")
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR'))
    p.add_argument('--out', default=os.environ.get('OUT_PATH', 'mantis_ssl_ohlcv.pt'))
    p.add_argument('--tickers', default=os.environ.get('TICKERS'))
    p.add_argument('--tfs', default=os.environ.get('TFS', '1min,3min,5min,15min'))
    p.add_argument('--pretext', default=os.environ.get('PRETEXT', 'mask'), choices=['mask', 'forecast'])
    p.add_argument('--backbone-ckpt', default=os.environ.get('BACKBONE_CKPT'))  # warm-start (stage 2)
    p.add_argument('--horizons', default=os.environ.get('HORIZONS', '5,10,20,25,50'))
    p.add_argument('--context-lengths', default=os.environ.get('CONTEXT_LENGTHS', '64,100,150,200'))
    p.add_argument('--seq', type=int, default=int(os.environ.get('SEQ', '64')))
    p.add_argument('--max-jitter', type=int, default=int(os.environ.get('MAX_JITTER', '16')))
    p.add_argument('--new-channels', type=int, default=int(os.environ.get('NEW_C', '8')))
    p.add_argument('--batch', type=int, default=int(os.environ.get('BATCH', '1024')))
    p.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', '60')))
    p.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '200')))
    p.add_argument('--lr', type=float, default=float(os.environ.get('LR', '1e-4')))
    p.add_argument('--val-frac', type=float, default=float(os.environ.get('VAL_FRAC', '0.1')))
    p.add_argument('--holdout-start', default=os.environ.get('HOLDOUT_START', '2026-01-01'))
    p.add_argument('--controls', default=os.environ.get('CONTROLS', 'shuffle,random'))
    p.add_argument('--control-epochs', type=int,
                   default=int(os.environ.get('CONTROL_EPOCHS', '8')))
    p.add_argument('--reuse-real-checkpoint', action='store_true',
                   default=os.environ.get('REUSE_REAL_CHECKPOINT') == '1')
    p.add_argument('--no-probe', action='store_true', default=os.environ.get('NO_PROBE') == '1')
    p.add_argument('--device', default=os.environ.get('DEVICE'))
    p.add_argument('--compile', action='store_true', default=os.environ.get('COMPILE') == '1')
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')))
    a = p.parse_args()
    loop_ssl(data_dir=a.data_dir, out_path=a.out,
             tickers=(a.tickers.split(',') if a.tickers else None), tfs=a.tfs.split(','),
             controls=tuple(a.controls.split(',')), probe=not a.no_probe,
             holdout_start=a.holdout_start, val_frac=a.val_frac, seq=a.seq, max_jitter=a.max_jitter,
             new_channels=a.new_channels, batch=a.batch, epochs=a.epochs,
             control_epochs=a.control_epochs, steps_per_epoch=a.steps,
             lr=a.lr, device=a.device, compile_model=a.compile, seed=a.seed, pretext=a.pretext,
             reuse_real_checkpoint=a.reuse_real_checkpoint,
             backbone_ckpt=a.backbone_ckpt,
             horizons=tuple(int(x) for x in a.horizons.split(',')),
             context_lengths=tuple(int(x) for x in a.context_lengths.split(',')))


if __name__ == '__main__':
    main()
