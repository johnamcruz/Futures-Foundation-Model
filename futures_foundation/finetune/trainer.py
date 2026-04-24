"""
Walk-forward fine-tuning trainer.

Entry points:
    run_labeling()        — Cell 3: label all tickers, save parquet cache
    run_walk_forward()    — Cell 4: train all folds, return results dict
    print_eval_summary()  — Cell 5: print threshold/fold/baseline tables
"""

import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..config import FFMConfig
from ..features import get_model_feature_columns
from .base import StrategyLabeler
from .config import TrainingConfig
from .dataset import HybridStrategyDataset
from .losses import FocalLoss
from .model import HybridStrategyModel


# ── Labeling ─────────────────────────────────────────────────────────────────

def run_labeling(
    labeler: StrategyLabeler,
    tickers: list,
    raw_dir: str,
    ffm_dir: str,
    cache_dir: str,
    micro_to_full: dict = None,
    force: bool = False,
) -> None:
    """
    For each ticker: load raw CSV + FFM parquet, call labeler.run(),
    save strategy_features and labels to cache_dir.

    Skips a ticker if cached files already exist (unless force=True).
    Raw data is expected at {raw_dir}/{data_ticker}_5min.csv.
    FFM features at {ffm_dir}/{data_ticker}_features.parquet.
    """
    os.makedirs(cache_dir, exist_ok=True)
    micro_to_full = micro_to_full or {}
    total_signals = total_bars = 0

    print(f"\n{'='*60}")
    print(f'  LABELING — {labeler.name.upper()} ({len(tickers)} tickers)')
    print(f"{'='*60}")

    for ticker in tickers:
        feat_path  = os.path.join(cache_dir, f'{ticker}_strategy_features.parquet')
        label_path = os.path.join(cache_dir, f'{ticker}_strategy_labels.parquet')

        if not force and os.path.exists(feat_path) and os.path.exists(label_path):
            cached = pd.read_parquet(label_path)
            sigs   = (cached['signal_label'] > 0).sum()
            print(f'  {ticker}: ⚡ cached — {len(cached):,} bars, {sigs} signals')
            total_signals += sigs
            total_bars    += len(cached)
            continue

        data_ticker   = micro_to_full.get(ticker, ticker)
        csv_path      = os.path.join(raw_dir, f'{data_ticker}_5min.csv')
        ffm_feat_path = os.path.join(ffm_dir,  f'{data_ticker}_features.parquet')

        if not os.path.exists(csv_path) or not os.path.exists(ffm_feat_path):
            print(f'  ⚠ Skip {ticker} — missing data'); continue

        print(f"\n{'─'*60}\n  {ticker}\n{'─'*60}")
        t0 = time.time()

        ffm_df = pd.read_parquet(ffm_feat_path)
        ffm_dt = pd.to_datetime(ffm_df['_datetime'])
        if ffm_dt.dt.tz is None:
            ffm_dt = ffm_dt.dt.tz_localize('UTC').tz_convert('America/New_York')
        ffm_df.index = ffm_dt

        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        if 'date' in df_raw.columns and 'datetime' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'date': 'datetime'})
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw.set_index('datetime', inplace=True)
        df_raw.sort_index(inplace=True)
        try:
            df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('America/New_York')
        except TypeError:
            if df_raw.index.tz is not None:
                df_raw.index = df_raw.index.tz_convert('America/New_York')
        print(f'  Loaded {len(df_raw):,} 5m bars')

        strategy_feats, labels_df = labeler.run(df_raw, ffm_df, ticker)

        strategy_feats.to_parquet(feat_path,  index=False)
        labels_df.to_parquet(label_path, index=False)

        sigs = (labels_df['signal_label'] > 0).sum()
        total_signals += sigs
        total_bars    += len(labels_df)
        print(f'  ✓ {ticker}: {sigs} signals | ({time.time() - t0:.1f}s)')

    print(f"\n{'='*60}")
    print(f'  ✅ LABELING COMPLETE — {total_bars:,} bars | {total_signals} signals')
    print(f'  {"✅ density OK" if total_signals >= 500 else "⚠️  density LOW (<500)"}')
    print(f"{'='*60}")


# ── DataLoader ────────────────────────────────────────────────────────────────

def _make_balanced_loader(
    dataset,
    batch_size: int,
    sig_per_batch: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """WeightedRandomSampler that delivers ~sig_per_batch signals per batch."""
    if not dataset.signal_indices or len(dataset.signal_indices) < sig_per_batch:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    n_total  = len(dataset)
    n_signal = len(dataset.signal_indices)
    n_noise  = n_total - n_signal
    target_frac = sig_per_batch / batch_size
    w_signal = target_frac / (n_signal / n_total) if n_signal > 0 else 1.0
    w_noise  = (1.0 - target_frac) / (n_noise / n_total) if n_noise > 0 else 1.0
    if hasattr(dataset, 'window_starts'):
        labels = [dataset._labels[s + dataset.seq_len - 1] for s in dataset.window_starts]
    else:
        labels = list(dataset._labels)  # ConcatDataset: _labels already per-window
    weights = [w_signal if l > 0 else w_noise for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                     num_workers=num_workers, pin_memory=True, drop_last=True)


def _load_fold_data(
    fold: dict,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    strategy_feature_cols: list,
    seq_len: int,
    micro_to_full: dict = None,
):
    """Load and time-slice FFM + strategy data for a single fold."""
    micro_to_full = micro_to_full or {}
    train_end = pd.Timestamp(fold['train_end'], tz='America/New_York')
    val_end   = pd.Timestamp(fold['val_end'],   tz='America/New_York')
    test_end  = pd.Timestamp(fold['test_end'],  tz='America/New_York')

    train_dsets = []; val_dsets = []; test_dsets = []

    for ticker in tickers:
        data_ticker = micro_to_full.get(ticker, ticker)
        ffm_path    = os.path.join(ffm_dir,      f'{data_ticker}_features.parquet')
        feat_path   = os.path.join(strategy_dir, f'{ticker}_strategy_features.parquet')
        label_path  = os.path.join(strategy_dir, f'{ticker}_strategy_labels.parquet')

        if not all(os.path.exists(p) for p in [ffm_path, feat_path, label_path]):
            print(f'  ⚠ Skip {ticker} — missing parquet'); continue

        ffm_df  = pd.read_parquet(ffm_path)
        strat_f = pd.read_parquet(feat_path)
        strat_l = pd.read_parquet(label_path)

        dt_col = pd.to_datetime(ffm_df['_datetime'])
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize('UTC').tz_convert('America/New_York')

        tr_mask   = dt_col < train_end
        val_mask  = (dt_col >= train_end) & (dt_col < val_end)
        test_mask = (dt_col >= val_end)   & (dt_col < test_end)

        for mask, dset_list, tag in [
            (tr_mask,   train_dsets, 'train'),
            (val_mask,  val_dsets,   'val'),
            (test_mask, test_dsets,  'test'),
        ]:
            idx = np.where(mask.values)[0]
            if len(idx) < seq_len + 1:
                continue
            lo, hi = idx[0], idx[-1] + 1
            ds = HybridStrategyDataset(
                ffm_df.iloc[lo:hi].reset_index(drop=True),
                strat_f.iloc[lo:hi].reset_index(drop=True),
                strat_l.iloc[lo:hi].reset_index(drop=True),
                strategy_feature_cols=strategy_feature_cols,
                seq_len=seq_len,
            )
            if len(ds.signal_indices) == 0:
                print(f'  ⚠ {ticker} {tag}: 0 signals — skipping')
                continue
            dset_list.append(ds)
            print(f'  {ticker} {tag}: {len(ds):,} windows, {len(ds.signal_indices)} signals')

    return train_dsets, val_dsets, test_dsets


def _concat_with_meta(dsets: list, seq_len: int):
    """Combine a list of HybridStrategyDatasets into a ConcatDataset with
    the per-window _labels and signal_indices that the balanced loader needs."""
    ds = ConcatDataset(dsets)
    ds.seq_len = seq_len
    ds._labels = np.concatenate([
        d._labels[d.window_starts[i] + d.seq_len - 1:
                  d.window_starts[i] + d.seq_len]
        for d in dsets for i in range(len(d))
    ])
    ds.signal_indices = []
    offset = 0
    for d in dsets:
        for local_i in d.signal_indices:
            ds.signal_indices.append(offset + local_i)
        offset += len(d)
    return ds


# ── Train / evaluate ──────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0; n_batches = 0
    correct = total = sig_correct = sig_total = 0

    for batch in loader:
        feats   = batch['features'].to(device)
        strat   = batch['strategy_features'].to(device)
        candles = batch['candle_types'].to(device)
        inst    = batch['instrument_ids'].to(device)
        sess    = batch['session_ids'].to(device)
        tod     = batch['time_of_day'].to(device)
        dow     = batch['day_of_week'].to(device)
        labels  = batch['signal_label'].to(device)
        max_rr  = batch['max_rr'].to(device)

        optimizer.zero_grad()
        out = model(features=feats, strategy_features=strat, candle_types=candles,
                    time_of_day=tod, day_of_week=dow,
                    instrument_ids=inst, session_ids=sess)

        cls_loss  = loss_fn(out['signal_logits'], labels)
        risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
        loss      = cls_loss + model.risk_weight * risk_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item(); n_batches += 1
        preds = out['signal_logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        sig_mask    = labels > 0
        sig_correct += (preds[sig_mask] == labels[sig_mask]).sum().item()
        sig_total   += sig_mask.sum().item()

    return {
        'loss': total_loss / max(n_batches, 1),
        'acc':  correct / max(total, 1),
        'sig_acc': sig_correct / max(sig_total, 1),
    }


def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0; n_batches = 0
    correct = total = tp = fp = fn = 0
    all_conf = []; all_labels = []; all_preds = []; all_max_rr = []

    with torch.no_grad():
        for batch in loader:
            feats   = batch['features'].to(device)
            strat   = batch['strategy_features'].to(device)
            candles = batch['candle_types'].to(device)
            inst    = batch['instrument_ids'].to(device)
            sess    = batch['session_ids'].to(device)
            tod     = batch['time_of_day'].to(device)
            dow     = batch['day_of_week'].to(device)
            labels  = batch['signal_label'].to(device)
            max_rr  = batch['max_rr'].to(device)

            out = model(features=feats, strategy_features=strat, candle_types=candles,
                        time_of_day=tod, day_of_week=dow,
                        instrument_ids=inst, session_ids=sess)

            cls_loss  = loss_fn(out['signal_logits'], labels)
            risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
            loss = cls_loss + model.risk_weight * risk_loss
            total_loss += loss.item(); n_batches += 1

            preds = out['signal_logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
            tp += ((preds > 0) & (labels > 0)).sum().item()
            fp += ((preds > 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels > 0)).sum().item()
            all_conf.extend(out['confidence'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_max_rr.extend(batch['max_rr'].tolist())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'loss':      total_loss / max(n_batches, 1),
        'acc':       correct / max(total, 1),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': all_conf, 'all_labels': all_labels,
        'all_preds': all_preds, 'all_max_rr': all_max_rr,
    }


# ── Fold ─────────────────────────────────────────────────────────────────────

def _config_hash(training_cfg: TrainingConfig) -> str:
    d = {k: v for k, v in training_cfg.__dict__.items() if k != 'baseline_wr'}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _train_fold(
    fold: dict,
    ffm_config: FFMConfig,
    training_cfg: TrainingConfig,
    num_strategy_features: int,
    strategy_feature_cols: list,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    output_dir: str,
    backbone_path: str,
    config_hash: str,
    micro_to_full: dict = None,
    warm_start_state: dict = None,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_name    = fold['name']
    ckpt_path    = os.path.join(output_dir, f'{fold_name}_{config_hash}_loss.pt')

    print(f"\n{'='*60}")
    print(f'  FOLD {fold_name} | train<{fold["train_end"]} val<{fold["val_end"]}')
    print(f'  {"Warm start from prev fold" if warm_start_state is not None else "Cold start"}')
    print(f"{'='*60}")

    # ── Datasets ──
    train_dsets, val_dsets, test_dsets = _load_fold_data(
        fold, tickers, ffm_dir, strategy_dir, strategy_feature_cols,
        training_cfg.seq_len, micro_to_full)

    if not train_dsets or not val_dsets:
        print(f'  ⚠ {fold_name}: insufficient data — skipping')
        return None

    train_ds = _concat_with_meta(train_dsets, training_cfg.seq_len)
    val_ds   = ConcatDataset(val_dsets)
    test_ds  = ConcatDataset(test_dsets) if test_dsets else None

    n_sig   = len(train_ds.signal_indices)
    n_total = len(train_ds)
    print(f'\n  Train: {n_total:,} windows, {n_sig} signals ({n_sig/n_total*100:.2f}%)')
    print(f'  Val:   {len(val_ds):,} windows')

    train_loader = _make_balanced_loader(train_ds, training_cfg.batch_size,
                                         training_cfg.sig_per_batch)
    val_loader   = DataLoader(val_ds, batch_size=training_cfg.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ──
    model = HybridStrategyModel(ffm_config, num_strategy_features,
                                training_cfg.num_labels,
                                training_cfg.risk_weight).to(device)

    if warm_start_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in warm_start_state.items()})
        print(f'  ✅ Warm start from prev fold')
    elif os.path.exists(backbone_path):
        print(f'  Loading backbone: {backbone_path}')
        model.load_backbone(backbone_path)
    else:
        print(f'  ⚠ Backbone not found — training from scratch')

    model.freeze_backbone(training_cfg.freeze_ratio)

    # ── Loss + optimiser ──
    class_weights = torch.tensor(
        [training_cfg.false_penalty, training_cfg.miss_penalty],
        dtype=torch.float32).to(device)
    loss_fn   = FocalLoss(gamma=training_cfg.focal_gamma, weight=class_weights,
                          label_smoothing=training_cfg.focal_smoothing)
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=training_cfg.lr, weight_decay=0.01)
    total_steps = training_cfg.epochs * len(train_loader)
    warmup      = min(500, total_steps // 10)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=training_cfg.lr, total_steps=total_steps,
        pct_start=warmup / total_steps, anneal_strategy='cos')

    # ── Resume ──
    start_epoch    = 0
    best_val_loss  = float('inf')
    best_val_epoch = -1
    best_signal_f1 = 0.0
    best_f1_epoch  = -1
    best_f1_state  = None
    patience_ctr   = 0
    ratio_bad_ctr  = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get('config_hash') == config_hash:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            start_epoch    = ckpt['epoch'] + 1
            best_val_loss  = ckpt['val_loss']
            best_val_epoch = ckpt['epoch']
            best_signal_f1 = ckpt.get('best_signal_f1', 0.0)
            best_f1_epoch  = ckpt.get('best_f1_epoch', -1)
            patience_ctr   = ckpt.get('patience_ctr', 0)
            print(f'  ▶ Resumed from epoch {start_epoch} (val_loss={best_val_loss:.4f})')
        else:
            print(f'  ℹ Config changed — starting fresh')

    # ── Training loop ──
    for epoch in range(start_epoch, training_cfg.epochs):
        tr = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = _evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        ratio     = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        improved  = va['loss'] < best_val_loss
        f1_better = va['f1'] > best_signal_f1
        save_str  = ''

        if improved:
            best_val_loss  = va['loss']
            best_val_epoch = epoch
            patience_ctr   = 0
            ratio_bad_ctr  = 0
            torch.save({
                'config_hash': config_hash, 'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': best_val_loss, 'patience_ctr': patience_ctr,
                'best_signal_f1': best_signal_f1, 'best_f1_epoch': best_f1_epoch,
            }, ckpt_path)
            save_str += ' 💾L'

        if f1_better:
            best_signal_f1 = va['f1']
            best_f1_epoch  = epoch
            best_f1_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_str += ' 📈F'

        if ratio > training_cfg.max_ratio:
            ratio_bad_ctr += 1
            ratio_str = f'🚨 {ratio:.2f} ({ratio_bad_ctr}/{training_cfg.ratio_patience})'
        else:
            ratio_bad_ctr = 0
            ratio_str = f'OK {ratio:.2f}'

        if not improved:
            patience_ctr += 1

        print(f'  {fold_name} E{epoch+1:2d}/{training_cfg.epochs} | '
              f'TrL:{tr["loss"]:.4f} VL:{va["loss"]:.4f} | '
              f'P:{va["precision"]:.3f} R:{va["recall"]:.3f} F1:{va["f1"]:.3f} | '
              f'{ratio_str}{save_str}')

        if patience_ctr >= training_cfg.patience:
            print(f'  ⏹ Early stop — patience exhausted'); break
        if ratio_bad_ctr >= training_cfg.ratio_patience:
            print(f'  ⏹ Early stop — overfitting'); break

    # ── Load best F1 for test eval ──
    print(f'\n  Checkpoint summary:')
    print(f'    val_loss  : epoch={best_val_epoch+1} score={best_val_loss:.4f}')
    print(f'    signal_f1 : epoch={best_f1_epoch+1}  score={best_signal_f1:.4f}')
    print(f'  ✅ Loading best signal_f1 (epoch {best_f1_epoch+1}) for test')

    if best_f1_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_f1_state.items()})
    elif os.path.exists(ckpt_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])

    next_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Test eval ──
    test_metrics = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=training_cfg.batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        test_metrics = _evaluate(model, test_loader, loss_fn, device)
        n_test = len(test_ds)
        n_sig  = test_metrics['tp'] + test_metrics['fn']
        print(f'\n  {fold_name} test: {n_test:,} bars | {n_sig} actual signals')
        print(f'  {"Thresh":>6}  {"Predicted":>9}  {"Correct":>7}  {"Precision":>9}  {"Recall":>6}  {"Rate"}')
        conf_arr = np.array(test_metrics['all_conf'])
        lab_arr  = np.array(test_metrics['all_labels'])
        pred_arr = np.array(test_metrics['all_preds'])
        for thresh in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            m = conf_arr >= thresh
            if m.sum() == 0: continue
            htp = ((pred_arr[m] > 0) & (lab_arr[m] > 0)).sum()
            hfp = ((pred_arr[m] > 0) & (lab_arr[m] == 0)).sum()
            hfn = ((pred_arr[m] == 0) & (lab_arr[m] > 0)).sum()
            hp  = htp / max(htp + hfp, 1)
            hr  = htp / max(htp + hfn, 1)
            ok  = ' ✅' if hp >= 0.40 else ''
            print(f'  {thresh:>6.2f}  {htp+hfp:>9}  {htp:>7}  {hp:>9.3f}  {hr:>6.3f}  '
                  f'{(htp+hfp)/max(len(lab_arr),1)*100:.1f}%{ok}')

    return model, test_metrics, next_fold_state


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(
    folds: list,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    output_dir: str,
    backbone_path: str,
    ffm_config: FFMConfig,
    training_cfg: TrainingConfig,
    num_strategy_features: int,
    strategy_feature_cols: list,
    micro_to_full: dict = None,
    device=None,
):
    """
    Train all walk-forward folds and return per-fold test metrics.

    Args:
        folds:                 List of dicts with keys name/train_end/val_end/test_end.
        tickers:               Instruments to include.
        ffm_dir:               Directory with {ticker}_features.parquet (FFM prepared).
        strategy_dir:          Directory with {ticker}_strategy_*.parquet (labeler output).
        output_dir:            Where to save checkpoints and ONNX model.
        backbone_path:         Path to best_backbone.pt.
        ffm_config:            FFMConfig used for the backbone.
        training_cfg:          TrainingConfig with all hyperparameters.
        num_strategy_features: Size of the strategy feature vector.
        strategy_feature_cols: Ordered column names for the strategy features.
        micro_to_full:         Optional ticker → data_ticker mapping (e.g. MES → ES).
        device:                torch.device (auto-detected if None).

    Returns:
        dict: fold_name → test_metrics dict (or None if fold was skipped).
              Also returns the last trained model as fold_results['_model'].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f'Backbone not found: {backbone_path}')

    config_hash = _config_hash(training_cfg)
    print(f'\n{"="*60}')
    print(f'  WALK-FORWARD — {len(folds)} folds | config hash: {config_hash}')
    print(f'  Backbone: {backbone_path}')
    print(f'{"="*60}')

    fold_results    = {}
    last_model      = None
    prev_fold_state = None

    for fold in folds:
        result = _train_fold(
            fold=fold,
            ffm_config=ffm_config,
            training_cfg=training_cfg,
            num_strategy_features=num_strategy_features,
            strategy_feature_cols=strategy_feature_cols,
            tickers=tickers,
            ffm_dir=ffm_dir,
            strategy_dir=strategy_dir,
            output_dir=output_dir,
            backbone_path=backbone_path,
            config_hash=config_hash,
            micro_to_full=micro_to_full,
            warm_start_state=prev_fold_state,
            device=device,
        )
        if result is not None:
            last_model, test_metrics, prev_fold_state = result
            fold_results[fold['name']] = test_metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_results['_model'] = last_model
    return fold_results


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(
    model: HybridStrategyModel,
    output_path: str,
    seq_len: int,
    num_ffm_features: int,
    num_strategy_features: int,
) -> None:
    """Export the fine-tuned model to ONNX for production inference."""
    import torch.onnx

    model.eval().cpu()
    dummy = {
        'features':          torch.randn(1, seq_len, num_ffm_features),
        'strategy_features': torch.randn(1, num_strategy_features),
        'candle_types':      torch.zeros(1, seq_len, dtype=torch.long),
        'instrument_ids':    torch.zeros(1, dtype=torch.long),
        'session_ids':       torch.zeros(1, seq_len, dtype=torch.long),
        'time_of_day':       torch.zeros(1, seq_len),
        'day_of_week':       torch.zeros(1, seq_len, dtype=torch.long),
    }
    torch.onnx.export(
        model,
        (dummy['features'], dummy['strategy_features'], dummy['candle_types'],
         dummy['time_of_day'], dummy['day_of_week'],
         dummy['instrument_ids'], dummy['session_ids']),
        output_path,
        input_names=['features', 'strategy_features', 'candle_types',
                     'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'],
        output_names=['signal_logits', 'risk_predictions', 'confidence'],
        dynamic_axes={
            'features':          {0: 'batch', 1: 'seq'},
            'strategy_features': {0: 'batch'},
            'candle_types':      {0: 'batch', 1: 'seq'},
            'time_of_day':       {0: 'batch', 1: 'seq'},
            'day_of_week':       {0: 'batch', 1: 'seq'},
            'instrument_ids':    {0: 'batch'},
            'session_ids':       {0: 'batch', 1: 'seq'},
            'signal_logits':     {0: 'batch'},
            'risk_predictions':  {0: 'batch'},
            'confidence':        {0: 'batch'},  # max(softmax(signal_logits))
        },
        opset_version=17,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  ✅ ONNX exported: {output_path} ({size_mb:.1f} MB)')


# ── Evaluation summary ────────────────────────────────────────────────────────

def print_eval_summary(
    fold_results: dict,
    baseline_wr: dict = None,
    output_dir: str = None,
) -> None:
    """
    Print the full walk-forward evaluation table (confidence thresholds,
    per-fold breakdown, and learning verification vs mechanical baseline).
    Equivalent to Cell 5 in the strategy notebook.
    """
    all_conf_c = []; all_labels_c = []; all_preds_c = []; all_rr_c = []
    for fname, metrics in fold_results.items():
        if fname == '_model' or metrics is None:
            continue
        all_conf_c.extend(metrics['all_conf'])
        all_labels_c.extend(metrics['all_labels'])
        all_preds_c.extend(metrics['all_preds'])
        all_rr_c.extend(metrics['all_max_rr'])

    if not all_labels_c:
        print('No fold results available.')
        return

    all_conf   = np.array(all_conf_c)
    all_labels = np.array(all_labels_c)
    all_preds  = np.array(all_preds_c)
    all_rr     = np.array(all_rr_c)

    n_sig = (all_labels > 0).sum()
    print(f'\n📊 Combined ({len(all_labels):,} bars): {n_sig} signals | '
          f'{len(all_labels)-n_sig} noise')
    print(f'   Signal rate: {n_sig/len(all_labels)*100:.2f}%')

    print(f'\n🎯 CONFIDENCE THRESHOLDS')
    print('='*72)
    print(f'   {"Thresh":>6}  {"Trades":>6}  {"Correct":>7}  {"Prec":>6}  '
          f'{"Recall":>6}  {"AvgRR":>6}  {"PF":>7}  Verdict')
    print(f'  {"-"*66}')

    for thresh in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = all_conf >= thresh
        if mask.sum() == 0:
            print(f'  {thresh:.2f}{"":>12}  — no trades'); continue
        t_labels = all_labels[mask]; t_preds = all_preds[mask]; t_rr = all_rr[mask]
        called   = t_preds > 0
        if called.sum() == 0: continue
        tp = (called & (t_labels > 0)).sum()
        fp = (called & (t_labels == 0)).sum()
        fn = (~called & (t_labels > 0)).sum()
        trades  = int(tp + fp)
        prec    = tp / max(trades, 1)
        rec     = tp / max(tp + fn, 1)
        wins_rr = t_rr[called & (t_labels > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:>7.2f}' if pf < 9999 else '    ∞  '
        verdict = '✅ EDGE' if prec >= 0.40 and trades >= 5 else ('⚠️ LOW' if trades < 5 else '❌')
        print(f'  {thresh:.2f}  {trades:>6}  {int(tp):>7}  {prec:>6.3f}  {rec:>6.3f}  '
              f'{avg_rr:>6.2f}  {pf_str}  {verdict}')

    print(f'\n{"="*72}')
    print(f'  📊 PER-FOLD BREAKDOWN (conf ≥ 0.90)')
    print(f'{"="*72}')
    for fname, metrics in fold_results.items():
        if fname == '_model': continue
        if metrics is None: print(f'  {fname}: no data'); continue
        ca = np.array(metrics['all_conf']); la = np.array(metrics['all_labels'])
        pa = np.array(metrics['all_preds']); ra = np.array(metrics['all_max_rr'])
        m  = ca >= 0.90; called = pa[m] > 0
        if called.sum() == 0: print(f'  {fname}: 0 trades at 0.90'); continue
        tp = (called & (la[m] > 0)).sum(); fp = (called & (la[m] == 0)).sum()
        wins_rr = ra[m][called & (la[m] > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:.2f}' if pf < 9999 else '∞'
        pl_r    = wins_rr.sum() - fp
        print(f'  {fname}: {int(tp+fp)} trades | Prec:{tp/max(tp+fp,1):.3f} | '
              f'AvgRR:{avg_rr:.2f} | PF:{pf_str} | {pl_r:+.1f}R')

    if baseline_wr:
        baseline_avg = np.mean(list(baseline_wr.values()))
        print(f'\n{"="*72}')
        print(f'  🧠 LEARNING VERIFICATION (vs mechanical baseline)')
        print(f'{"="*72}')
        print(f'  Mechanical baseline avg: {baseline_avg*100:.1f}%')
        print(f'  {"Thresh":>6}  {"Trades":>6}  {"Prec":>6}  {"vs Base":>8}  Verdict')
        print(f'  {"-"*48}')
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            mask   = all_conf >= thresh
            called = all_preds[mask] > 0
            if called.sum() < 3: continue
            tp_n = (called & (all_labels[mask] > 0)).sum()
            fp_n = (called & (all_labels[mask] == 0)).sum()
            prec  = tp_n / max(tp_n + fp_n, 1)
            delta = prec - baseline_avg
            ok    = '✅ LEARNING' if delta > 0 else '❌ BELOW'
            print(f'  {thresh:.2f}  {int(tp_n+fp_n):>6}  {prec:>6.3f}  {delta:>+8.1%}  {ok}')

    if output_dir:
        onnx_path = os.path.join(output_dir, 'strategy_hybrid.onnx')
        print(f'\n  ONNX model: {onnx_path}')
        print(f'  Checkpoints: {output_dir}')
