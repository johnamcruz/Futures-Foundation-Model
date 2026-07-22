"""Structural NextLeg SSL: causal HH/HL/LH/LL plus future BOS/CHOCH prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_structural import (
    BREAK_NAMES, MASKED_BREAK, STRUCTURE_NAMES, structural_targets_by_segments,
    structural_span_bounds, validate_structural_reserve,
)
from .common import _apply_control, _gather_batch
from .forecast import _ForecastTrainer
from .nextleg import NextLegNet


def _balanced_accuracy(truth, pred, classes):
    values = []
    for cls in range(int(classes)):
        rows = truth == cls
        if rows.any():
            values.append(float((pred[rows] == cls).float().mean()))
    return float(np.mean(values)) if values else 0.0


def _correlation(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if len(pred) < 2 or pred.std() <= 1e-12 or truth.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, truth)[0, 1])


def _load_warm_nextleg_heads(net, path):
    """Load only parent NextLeg adapter/forecast/duration tensors from a trainer sidecar.

    Structural heads are intentionally new.  Requiring every inherited tensor and exact shape
    prevents silently starting from a partial or unrelated sidecar.
    """
    payload = torch.load(path, map_location="cpu")
    saved = payload.get("model_state") or {}
    own = net.state_dict()
    prefixes = ("adapter.", "decoder.", "leg_head.")
    required = [key for key in own if key.startswith(prefixes)]
    missing = [key for key in required if key not in saved]
    if missing:
        raise RuntimeError(f"warm NextLeg trainer is missing task tensors: {missing}")
    for key in required:
        if own[key].shape != saved[key].shape:
            raise RuntimeError(f"warm task tensor shape mismatch: {key}")
        own[key].copy_(saved[key])
    net.load_state_dict(own)
    return tuple(required)


class StructuralNextLegNet(NextLegNet):
    """NextLeg anchor heads plus generic structural/event heads; checkpoint remains encoder-only."""

    def __init__(self, *args, span_width=5, **kwargs):
        super().__init__(*args, **kwargs)
        emb = self.decoder[0].in_features
        hidden = emb // 4
        self.structure_head = nn.Sequential(
            nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, 2 * len(STRUCTURE_NAMES)))
        self.excursion_head = nn.Sequential(
            nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.break_head = nn.Sequential(
            nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, len(BREAK_NAMES)))
        self.break_delay_head = nn.Sequential(
            nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.span_width = int(span_width)
        self.span_decoder = nn.Sequential(
            nn.Linear(emb, hidden), nn.GELU(), nn.Linear(hidden, self.C * self.span_width))

    def forward_structural(self, context):
        embedding = self.embed(context)
        candles = self.decoder(embedding).view(-1, self.C, self.nH)
        durations = self.leg_head(embedding)
        structure = self.structure_head(embedding).view(-1, 2, len(STRUCTURE_NAMES))
        excursions = self.excursion_head(embedding)
        breaks = self.break_head(embedding)
        break_delay = self.break_delay_head(embedding).squeeze(-1)
        span = self.span_decoder(embedding).view(-1, self.C, self.span_width)
        return candles, durations, structure, excursions, breaks, break_delay, span


class _StructuralNextLegTrainer(_ForecastTrainer):
    """Stream-separated target construction and deterministic multi-task validation."""

    def __init__(self, big, tr, va, *, leg_cap=256, leg_w=1.0, leg_k=2,
                 mse_weight=1.0, structure_current_w=.25, structure_next_w=.75,
                 excursion_w=.25, structure_event_w=.75, structure_event_horizon=128,
                 structure_span_w=.25, structure_span_width=5, structure_span_prob=.5,
                 target_reserve=None, _stream_layout=None, head_lr=None,
                 warm_trainer_ckpt=None, **base):
        if _stream_layout is None:
            raise ValueError("nextleg_structural requires exact assembled stream layout")
        super().__init__(big, tr, va, **base)
        self.leg_w, self.mse_weight = float(leg_w), float(mse_weight)
        self.structure_current_w = float(structure_current_w)
        self.structure_next_w = float(structure_next_w)
        self.excursion_w, self.structure_event_w = float(excursion_w), float(structure_event_w)
        self.event_horizon = int(structure_event_horizon)
        self.structure_span_w = float(structure_span_w)
        self.span_width = int(structure_span_width)
        self.span_prob = float(structure_span_prob)
        objective_weights = {
            "mse_weight": self.mse_weight, "leg_w": self.leg_w,
            "structure_current_w": self.structure_current_w,
            "structure_next_w": self.structure_next_w, "excursion_w": self.excursion_w,
            "structure_event_w": self.structure_event_w,
            "structure_span_w": self.structure_span_w,
        }
        if any(value < 0 for value in objective_weights.values()):
            raise ValueError(f"structural objective weights must be non-negative: {objective_weights}")
        if self.event_horizon < 1:
            raise ValueError("structure_event_horizon must be positive")
        if not (0.0 < self.span_prob <= 1.0):
            raise ValueError("structure_span_prob must lie in (0,1]")
        self.leg_k = int(leg_k)
        # Validate the narrowest configured context now; every wider context is then safe too.
        structural_span_bounds(
            min(int(value) for value in self.clens), confirmation_lag=self.leg_k,
            span_width=self.span_width)
        self.head_lr = float(head_lr) if head_lr is not None else 10.0 * float(self.lr)
        if self.head_lr <= 0:
            raise ValueError("head_lr must be positive")
        self.warm_trainer_ckpt = warm_trainer_ckpt
        segments = tuple((stream.base, stream.size) for stream in _stream_layout.streams)
        targets = structural_targets_by_segments(
            np.asarray(big, np.float32), segments, k=int(leg_k), leg_cap=int(leg_cap),
            event_horizon=self.event_horizon)
        reserve = (self.max_ctx + max(2 * int(leg_cap) + int(leg_k), self.event_horizon)
                   if target_reserve is None else int(target_reserve))
        validate_structural_reserve(
            targets.confirms, targets.future_ends, max_ctx=self.max_ctx,
            target_reserve=reserve)
        starts = targets.confirms - self.max_ctx + 1
        tr_values, va_values = np.asarray(tr), np.asarray(va)
        in_train, in_val = np.isin(starts, tr_values), np.isin(starts, va_values)
        if in_train.sum() < 1000 or in_val.sum() < 200:
            raise ValueError("nextleg_structural: too few split-safe resolved anchors "
                             f"(train={int(in_train.sum())}, val={int(in_val.sum())})")
        values = targets.values
        self._replace_start_pool("tr", starts[in_train])
        self._replace_start_pool("va", starts[in_val])
        self._tgt_tr = torch.as_tensor(values[in_train], device=self.dev)
        self._tgt_va = torch.as_tensor(values[in_val], device=self.dev)
        event = targets.break_event[in_train]
        counts = np.bincount(event[event >= 0], minlength=len(BREAK_NAMES)).astype(np.float64)
        weights = np.ones(len(BREAK_NAMES), np.float32)
        present = counts > 0
        weights[present] = 1.0 / np.sqrt(counts[present])
        if present.any():
            weights[present] /= weights[present].mean()
        self.event_weights = torch.as_tensor(weights, device=self.dev)
        if self.verbose:
            state_counts = np.bincount(targets.current_state[in_train],
                                       minlength=len(STRUCTURE_NAMES))
            print(f"  [structural] anchors train={len(self.tr):,} val={len(self.va):,} "
                  f"states={state_counts.tolist()} events={counts.astype(int).tolist()} "
                  f"event_horizon={self.event_horizon}", flush=True)

    def build_net(self):
        net = StructuralNextLegNet(
            C=self.C, new_channels=self.new_channels, horizons=self.hlist,
            model_id=self.model_id, aux_dim=0, span_width=self.span_width).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(self.backbone_ckpt, map_location="cpu"))
        if self.warm_trainer_ckpt:
            _load_warm_nextleg_heads(net, self.warm_trainer_ckpt)
        self.net = net

    def make_optimizer(self):
        encoder_ids = {id(parameter) for parameter in self._encoder().parameters()}
        encoder = [p for p in self.net.parameters() if p.requires_grad and id(p) in encoder_ids]
        heads = [p for p in self.net.parameters() if p.requires_grad and id(p) not in encoder_ids]
        return torch.optim.AdamW(
            [{"params": encoder, "lr": self.lr}, {"params": heads, "lr": self.head_lr}],
            weight_decay=self.weight_decay)

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
        target = self._tgt_tr if starts is self.tr else self._tgt_va
        batch_index = self.sample_indices(starts, generator=gen)
        window = _gather_batch(self.big_t, starts, batch_index, self.parent)
        length = int(self.clens_t[torch.randint(
            0, len(self.clens_t), (1,), device=self.dev, generator=gen)].item())
        context_raw = window[:, :, self.max_ctx - length:self.max_ctx]
        future_raw = window[:, :, self.max_ctx:]
        mean = context_raw.mean(2, keepdim=True)
        std = context_raw.std(2, keepdim=True) + 1e-6
        context = ((context_raw - mean) / std).clamp(-self.clamp, self.clamp)
        future = ((future_raw - mean) / std).clamp(-self.clamp, self.clamp)
        candle_target = future[:, :, self.h_off] - context[:, :, -1:]
        lo, hi = structural_span_bounds(
            length, confirmation_lag=self.leg_k, span_width=self.span_width)
        span_target = context[:, :, lo:hi].clone()
        source = _apply_control(context, self.control)
        active = torch.rand(len(source), device=self.dev, generator=gen) < self.span_prob
        if active.any():
            mask = active[:, None, None].expand(-1, self.C, self.span_width)
            replacement = torch.randn(
                (len(source), self.C, self.span_width), device=self.dev, generator=gen,
                dtype=source.dtype)
            source[:, :, lo:hi] = torch.where(mask, replacement, source[:, :, lo:hi])
        return source, candle_target, target[batch_index], span_target, active

    def _losses(self, batch):
        context, candle_target, target, span_target, span_active = batch
        candles, duration, structure, excursion, breaks, break_delay, span = \
            self.net.forward_structural(context)
        candle_loss = F.mse_loss(candles.float(), candle_target)
        duration_loss = F.smooth_l1_loss(duration.float(), target[:, :2])
        current_loss = F.cross_entropy(structure[:, 0].float(), target[:, 2].long())
        next_loss = F.cross_entropy(structure[:, 1].float(), target[:, 3].long())
        excursion_loss = F.smooth_l1_loss(excursion.float(), target[:, 4:6])
        event_truth = target[:, 6].long()
        event_rows = event_truth >= 0
        event_loss = (F.cross_entropy(
            breaks[event_rows].float(), event_truth[event_rows], weight=self.event_weights)
            if event_rows.any() else breaks.sum() * 0.0)
        happened = event_truth > 0
        delay_loss = (F.smooth_l1_loss(break_delay[happened].float(), target[happened, 7])
                      if happened.any() else break_delay.sum() * 0.0)
        span_loss = (F.mse_loss(span[span_active].float(), span_target[span_active])
                     if span_active.any() else span.sum() * 0.0)
        total = (self.mse_weight * candle_loss + self.leg_w * duration_loss
                 + self.structure_current_w * current_loss + self.structure_next_w * next_loss
                 + self.excursion_w * excursion_loss
                 + self.structure_event_w * (event_loss + .25 * delay_loss)
                 + self.structure_span_w * span_loss)
        return total, (candles, duration, structure, excursion, breaks, break_delay, span), {
            "candle": candle_loss, "duration": duration_loss, "current": current_loss,
            "next": next_loss, "excursion": excursion_loss, "event": event_loss,
            "delay": delay_loss, "span": span_loss}

    def compute_loss(self, batch):
        return self._losses(batch)[0]

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        generator = torch.Generator(device=self.dev); generator.manual_seed(20260722)
        totals = {key: 0.0 for key in
                  ("total", "candle", "persist", "duration", "current", "next",
                   "excursion", "event", "delay", "span", "span_baseline")}
        duration_pred, duration_true, excursion_pred, excursion_true = [], [], [], []
        current_pred, current_true, next_pred, next_true, event_pred, event_true = [], [], [], [], [], []
        embed_context = None
        batches = min(20, max(1, len(self.va) // self.batch))
        for _ in range(batches):
            batch = self.make_batch(self.va, gen=generator)
            if embed_context is None:
                embed_context = batch[0]
            total, output, parts = self._losses(batch)
            candles, durations, structures, excursions, breaks, _delay, _span = output
            totals["total"] += float(total)
            totals["persist"] += float((batch[1] ** 2).mean())
            if batch[4].any():
                totals["span_baseline"] += float((batch[3][batch[4]] ** 2).mean())
            for key in parts:
                totals[key] += float(parts[key])
            duration_pred.append(durations.float().cpu()); duration_true.append(batch[2][:, :2].cpu())
            excursion_pred.append(excursions.float().cpu()); excursion_true.append(batch[2][:, 4:6].cpu())
            current_pred.append(structures[:, 0].argmax(1).cpu()); current_true.append(batch[2][:, 2].long().cpu())
            next_pred.append(structures[:, 1].argmax(1).cpu()); next_true.append(batch[2][:, 3].long().cpu())
            event_pred.append(breaks.argmax(1).cpu()); event_true.append(batch[2][:, 6].long().cpu())
        dp, dt = torch.cat(duration_pred), torch.cat(duration_true)
        xp, xt = torch.cat(excursion_pred), torch.cat(excursion_true)
        cp, ct, npred, nt = map(torch.cat, (current_pred, current_true, next_pred, next_true))
        ep, et = torch.cat(event_pred), torch.cat(event_true)
        event_mask = et >= 0
        skill = 1.0 - totals["candle"] / max(totals["persist"], 1e-12)
        extra = {
            "skill": skill,
            "leg_corr1": _correlation(dp[:, 0], dt[:, 0]),
            "leg_corr2": _correlation(dp[:, 1], dt[:, 1]),
            "excursion_corr": float(np.mean([
                _correlation(xp[:, index], xt[:, index]) for index in (0, 1)])),
            "current_structure_bal_acc": _balanced_accuracy(ct, cp, len(STRUCTURE_NAMES)),
            "next_structure_bal_acc": _balanced_accuracy(nt, npred, len(STRUCTURE_NAMES)),
            "break_bal_acc": (_balanced_accuracy(et[event_mask], ep[event_mask], len(BREAK_NAMES))
                              if event_mask.any() else 0.0),
            "span_skill": 1.0 - totals["span"] / max(totals["span_baseline"], 1e-12),
            "std": float(self.net.embed(embed_context).std(0).mean()),
        }
        self.net.train()
        return totals["total"] / batches, extra

    def log_line(self, epoch, train_loss, val_loss, extra, improved):
        if self.verbose:
            print(f"  ep{epoch:>3} train={train_loss:.4f} val={val_loss:.4f} "
                  f"skill={extra['skill']:+.3f} "
                  f"legR={extra['leg_corr1']:+.3f}/{extra['leg_corr2']:+.3f} "
                  f"structBA={extra['current_structure_bal_acc']:.3f}/"
                  f"{extra['next_structure_bal_acc']:.3f} "
                  f"breakBA={extra['break_bal_acc']:.3f} "
                  f"span={extra['span_skill']:+.3f} excR={extra['excursion_corr']:+.3f} "
                  f"std={extra['std']:.4f}"
                  f"{'  *' if improved else ''}", flush=True)


def train_ssl_nextleg_structural(
        big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
        context_lengths=(64, 100, 150, 200), new_channels=8, epochs=20,
        steps_per_epoch=50, batch=512, lr=1e-5, head_lr=1e-4, weight_decay=0.0,
        patience=8, device=None, model_id="paris-noah/Mantis-8M", backbone_ckpt=None,
        warm_trainer_ckpt=None, control="real", seed=0, clamp=10.0, grad_clip=1.0,
        verbose=True, ckpt_path=None, resume=False, freeze_encoder_layers=2, std_guard=1.6,
        leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0,
        structure_current_w=.25, structure_next_w=.75, excursion_w=.25,
        structure_event_w=.75, structure_event_horizon=128, target_reserve=None,
        structure_span_w=.25, structure_span_width=5, structure_span_prob=.5,
        _stream_layout=None, lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        log_every_steps=25, **_ignore):
    trainer = _StructuralNextLegTrainer(
        big, train_starts, val_starts, horizons=horizons, context_lengths=context_lengths,
        new_channels=new_channels, model_id=model_id, backbone_ckpt=backbone_ckpt,
        warm_trainer_ckpt=warm_trainer_ckpt, clamp=clamp, leg_cap=leg_cap, leg_w=leg_w,
        leg_k=leg_k, mse_weight=mse_weight, structure_current_w=structure_current_w,
        structure_next_w=structure_next_w, excursion_w=excursion_w,
        structure_event_w=structure_event_w, structure_event_horizon=structure_event_horizon,
        structure_span_w=structure_span_w, structure_span_width=structure_span_width,
        structure_span_prob=structure_span_prob,
        target_reserve=target_reserve, _stream_layout=_stream_layout, epochs=epochs,
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, head_lr=head_lr,
        weight_decay=weight_decay, patience=patience, device=device, seed=seed,
        grad_clip=grad_clip, verbose=verbose, control=control, ckpt_path=ckpt_path,
        resume=resume, freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        log_every_steps=log_every_steps)
    return trainer.fit()


__all__ = ["StructuralNextLegNet", "train_ssl_nextleg_structural",
           "_load_warm_nextleg_heads"]
