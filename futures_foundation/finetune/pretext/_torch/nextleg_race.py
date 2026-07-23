"""Stage-2.8 v2 NEXT-LEG causal-range path-race trainer."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_race import RACE_LEVELS, scaled_path_race
from .common import _apply_control, _gather_batch
from .forecast import _ForecastTrainer
from .nextleg import NextLegNet, _alternating_fractals


def _correlation(prediction, truth) -> float:
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if len(prediction) < 2 or prediction.std() <= 1e-12 or truth.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(prediction, truth)[0, 1])


def _binary_auc(truth, score) -> float:
    """Dependency-free tie-aware binary ROC AUC."""
    truth, score = np.asarray(truth, np.int8), np.asarray(score, np.float64)
    positive, negative = truth == 1, truth == 0
    if not positive.any() or not negative.any():
        return 0.5
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _leg_race_targets(big, k, leg_cap, *, race_levels=RACE_LEVELS,
                      race_scale_lookback=64, race_cap=8.0):
    """Resolved NextLeg duration plus fixed-scale reach/adverse/delay targets for one stream."""
    high, low, close = big[:, 1], big[:, 2], big[:, 3]
    sequence = _alternating_fractals(high, low, k)
    confirms, targets, valid = [], [], []
    for index in range(len(sequence) - 2):
        _origin, confirm, direction = sequence[index]
        next_origin, _, _ = sequence[index + 1]
        following_origin, _, _ = sequence[index + 2]
        first, second = next_origin - confirm, following_origin - next_origin
        race = scaled_path_race(
            high, low, close, confirm, next_origin, direction,
            levels=race_levels, lookback=race_scale_lookback, cap=race_cap)
        ok = ((first > 0) and (second > 0)
              and (first <= leg_cap) and (second <= leg_cap)
              and bool(np.isfinite(race).all()))
        confirms.append(confirm)
        targets.append((
            np.log1p(max(first, 0)), np.log1p(max(second, 0)),
            *np.nan_to_num(race, nan=0.0).reshape(-1).tolist(),
        ))
        valid.append(ok)
    width = 2 + 3 * len(tuple(race_levels))
    return (
        np.asarray(confirms, np.int64),
        np.asarray(targets, np.float32).reshape(-1, width),
        np.asarray(valid, bool),
    )


def _leg_race_targets_by_segments(big, segments, k, leg_cap, **race):
    """Construct pivots independently inside every ticker/timeframe stream."""
    confirms, targets = [], []
    bars = np.asarray(big, np.float32)
    for base, size in segments:
        base, size = int(base), int(size)
        if base < 0 or size < 0 or base + size > len(bars):
            raise ValueError("stream segment lies outside assembled bars")
        local_confirm, local_target, valid = _leg_race_targets(
            bars[base:base + size], k, leg_cap, **race)
        if valid.any():
            confirms.append(local_confirm[valid] + base)
            targets.append(local_target[valid])
    if not targets:
        levels = tuple(race.get("race_levels", RACE_LEVELS))
        return np.empty(0, np.int64), np.empty((0, 2 + 3 * len(levels)), np.float32)
    return np.concatenate(confirms), np.concatenate(targets)


class NextLegRaceNet(NextLegNet):
    """NextLeg anchor heads plus monotone reach/adverse/delay forecasts."""

    def __init__(self, *args, race_levels=RACE_LEVELS, **kwargs):
        super().__init__(*args, **kwargs)
        self.race_levels = tuple(float(value) for value in race_levels)
        embedding = self.decoder[0].in_features
        self.race_head = nn.Sequential(
            nn.Linear(embedding, embedding // 4), nn.GELU(),
            nn.Linear(embedding // 4, 3 * len(self.race_levels)))

    def readouts(self, embedding):
        """Decode every task head from one already-computed embedding."""
        candles = self.decoder(embedding).view(-1, self.C, self.nH)
        durations = self.leg_head(embedding)
        raw = self.race_head(embedding).view(-1, 3, len(self.race_levels))
        # Higher levels cannot be more reachable.  Adverse excursion and elapsed time cannot
        # decrease as the target level moves farther away.
        reach = torch.cat((
            raw[:, 0, :1],
            raw[:, 0, :1] - torch.cumsum(F.softplus(raw[:, 0, 1:]), dim=1),
        ), dim=1)
        adverse = torch.cumsum(F.softplus(raw[:, 1]), dim=1)
        delay = torch.cumsum(F.softplus(raw[:, 2]), dim=1)
        return candles, durations, reach, adverse, delay

    def forward_race(self, context):
        return self.readouts(self.embed(context))


def _validate_race_target_reserve(targets, *, max_ctx, target_reserve,
                                  confirmation_lag, batch_parent):
    """Prove every resolved two-leg target and its final pivot confirmation fit its split.

    Durations end at the second future pivot *origin*, but that pivot is not knowable until
    ``confirmation_lag`` later bars close.  Reserving only the two durations would therefore let
    boundary anchors use a pivot whose confirmation lies in validation or the 2026 holdout.
    """
    horizon = np.expm1(targets[:, 0]) + np.expm1(targets[:, 1])
    target_future = horizon + int(confirmation_lag)
    max_future = float(target_future.max()) if len(target_future) else 0.0
    reserved_future = int(target_reserve) - int(max_ctx)
    assert max_future < reserved_future + 1, (
        f"TEMPORAL LEAK: race target/confirmation reads {max_future:.0f} bars ahead but only "
        f"{reserved_future} split-safe future bars were reserved "
        f"(target_reserve={target_reserve}, ctx={max_ctx}, "
        f"confirmation_lag={confirmation_lag}, batch_parent={batch_parent}).")


def _load_warm_heads(net, path):
    """Warm the common adapter/candle/duration heads from a matched trainer sidecar."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(saved, dict):
        raise ValueError(f"warm trainer sidecar has no model_state: {path}")
    own = net.state_dict()
    prefixes = ("adapter.", "decoder.", "leg_head.")
    required = [key for key in own if key.startswith(prefixes)]
    missing = [key for key in required if key not in saved]
    shapes = [key for key in required if key in saved and own[key].shape != saved[key].shape]
    if missing or shapes:
        raise ValueError(f"warm trainer mismatch: missing={missing[:8]} shape={shapes[:8]}")
    with torch.no_grad():
        for key in required:
            own[key].copy_(saved[key])
    net.load_state_dict(own)


class _NextLegRaceTrainer(_ForecastTrainer):
    def __init__(self, big, tr, va, *, leg_cap=256, leg_w=1.0, leg_k=2,
                 mse_weight=1.0, race_w=0.5, race_cap=8.0,
                 race_levels=RACE_LEVELS, race_scale_lookback=64,
                 target_reserve=None, _stream_layout=None, head_lr=None,
                 warm_trainer_ckpt=None, **forecast):
        if _stream_layout is None:
            raise ValueError("nextleg_race requires exact assembled stream layout")
        _ForecastTrainer.__init__(self, big, tr, va, **forecast)
        self.leg_w, self.mse_weight, self.race_w = (
            float(leg_w), float(mse_weight), float(race_w))
        self.race_cap = float(race_cap)
        self.race_levels = tuple(float(value) for value in race_levels)
        self.race_scale_lookback = int(race_scale_lookback)
        self.head_lr = float(head_lr) if head_lr is not None else 10.0 * float(self.lr)
        self.warm_trainer_ckpt = warm_trainer_ckpt
        if self.race_w <= 0 or self.head_lr <= 0:
            raise ValueError("race_w and head_lr must be positive")
        if self.race_scale_lookback > min(self.clens):
            raise ValueError("race_scale_lookback must fit the shortest context")

        segments = tuple((stream.base, stream.size) for stream in _stream_layout.streams)
        confirms, targets = _leg_race_targets_by_segments(
            np.asarray(big, np.float32), segments, int(leg_k), int(leg_cap),
            race_levels=self.race_levels, race_scale_lookback=self.race_scale_lookback,
            race_cap=self.race_cap)
        reserve = (self.max_ctx + 2 * int(leg_cap) + int(leg_k)
                   if target_reserve is None else int(target_reserve))
        _validate_race_target_reserve(
            targets, max_ctx=self.max_ctx, target_reserve=reserve,
            confirmation_lag=int(leg_k), batch_parent=self.parent)
        starts = confirms - self.max_ctx + 1
        train_values, val_values = np.asarray(tr), np.asarray(va)
        in_train, in_val = np.isin(starts, train_values), np.isin(starts, val_values)
        if in_train.sum() < 1000 or in_val.sum() < 200:
            raise ValueError(
                f"nextleg_race: too few split-safe anchors "
                f"(train={int(in_train.sum())}, val={int(in_val.sum())})")
        self._replace_start_pool("tr", starts[in_train])
        self._replace_start_pool("va", starts[in_val])
        self._tgt_tr = torch.as_tensor(targets[in_train], device=self.dev)
        self._tgt_va = torch.as_tensor(targets[in_val], device=self.dev)
        levels = len(self.race_levels)
        reach = targets[in_train, 2:2 + levels]
        positives = reach.sum(0)
        negatives = len(reach) - positives
        self.reach_pos_weight = torch.as_tensor(
            np.clip(negatives / np.maximum(positives, 1), 0.25, 20.0).astype(np.float32),
            device=self.dev)
        if self.verbose:
            print(
                f"  [nextleg_race:v2] anchors train={len(self.tr):,} val={len(self.va):,} "
                f"levels={self.race_levels} causal_range_lookback={self.race_scale_lookback} "
                f"reach_rate={np.round(reach.mean(0), 3).tolist()}",
                flush=True)

    def build_net(self):
        net = NextLegRaceNet(
            C=self.C, new_channels=self.new_channels, horizons=self.hlist,
            model_id=self.model_id, aux_dim=0, race_levels=self.race_levels).to(self.dev)
        if self.backbone_ckpt:
            net.encoder.load_state_dict(torch.load(
                self.backbone_ckpt, map_location="cpu", weights_only=False))
        if self.warm_trainer_ckpt:
            _load_warm_heads(net, self.warm_trainer_ckpt)
        self.net = net

    def make_optimizer(self):
        encoder_ids = {id(parameter) for parameter in self._encoder().parameters()}
        encoder = [parameter for parameter in self.net.parameters()
                   if parameter.requires_grad and id(parameter) in encoder_ids]
        heads = [parameter for parameter in self.net.parameters()
                 if parameter.requires_grad and id(parameter) not in encoder_ids]
        return torch.optim.AdamW(
            ({"params": encoder, "lr": self.lr}, {"params": heads, "lr": self.head_lr}),
            weight_decay=self.weight_decay)

    def make_batch(self, starts, gen=None):
        """Gather a causal input/candle target plus its identically sampled race target."""
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
        return _apply_control(context, self.control), candle_target, target[batch_index]

    def _losses(self, batch):
        context, candle_target, target = batch
        candles, durations, reach, adverse, delay = self.net.forward_race(context)
        levels = len(self.race_levels)
        reach_truth = target[:, 2:2 + levels]
        adverse_truth = target[:, 2 + levels:2 + 2 * levels]
        delay_truth = target[:, 2 + 2 * levels:2 + 3 * levels]
        parts = {
            "candle": F.mse_loss(candles.float(), candle_target),
            "duration": F.smooth_l1_loss(durations.float(), target[:, :2]),
            "reach": F.binary_cross_entropy_with_logits(
                reach.float(), reach_truth, pos_weight=self.reach_pos_weight),
            "adverse": F.smooth_l1_loss(adverse.float(), adverse_truth),
            "delay": F.smooth_l1_loss(delay.float(), delay_truth),
        }
        total = (self.mse_weight * parts["candle"] + self.leg_w * parts["duration"]
                 + self.race_w * (parts["reach"] + 0.5 * parts["adverse"]
                                  + 0.25 * parts["delay"]))
        return total, (candles, durations, reach, adverse, delay), parts

    def compute_loss(self, batch):
        return self._losses(batch)[0]

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        generator = torch.Generator(device=self.dev); generator.manual_seed(20260718)
        totals = {key: 0.0 for key in
                  ("total", "candle", "persist", "duration", "reach", "adverse", "delay")}
        predictions = {key: [] for key in ("duration", "reach", "adverse", "delay")}
        truths = {key: [] for key in predictions}
        batches = min(20, max(1, len(self.va) // self.batch))
        embed_context = None
        for _ in range(batches):
            batch = self.make_batch(self.va, gen=generator)
            if embed_context is None:
                embed_context = batch[0]
            total, output, parts = self._losses(batch)
            _candles, duration, reach, adverse, delay = output
            levels = len(self.race_levels)
            target = batch[2]
            totals["total"] += float(total)
            totals["persist"] += float((batch[1] ** 2).mean())
            for key, value in parts.items():
                totals[key] += float(value)
            for key, prediction, truth in (
                ("duration", duration, target[:, :2]),
                ("reach", reach.sigmoid(), target[:, 2:2 + levels]),
                ("adverse", adverse, target[:, 2 + levels:2 + 2 * levels]),
                ("delay", delay, target[:, 2 + 2 * levels:2 + 3 * levels]),
            ):
                predictions[key].append(prediction.float().cpu())
                truths[key].append(truth.cpu())
        prediction = {key: torch.cat(value).numpy() for key, value in predictions.items()}
        truth = {key: torch.cat(value).numpy() for key, value in truths.items()}
        reach_auc = [_binary_auc(truth["reach"][:, index], prediction["reach"][:, index])
                     for index in range(len(self.race_levels))]
        adverse_corr = [_correlation(prediction["adverse"][:, index], truth["adverse"][:, index])
                        for index in range(len(self.race_levels))]
        delay_corr = [_correlation(prediction["delay"][:, index], truth["delay"][:, index])
                      for index in range(len(self.race_levels))]
        duration_corr = [_correlation(prediction["duration"][:, index],
                                     truth["duration"][:, index]) for index in (0, 1)]
        extra = {
            "skill": 1.0 - totals["candle"] / max(totals["persist"], 1e-12),
            "leg_corr1": duration_corr[0], "leg_corr2": duration_corr[1],
            "race_reach_auc": float(np.mean(reach_auc)),
            "race_reach_aucs": tuple(float(value) for value in reach_auc),
            "race_adverse_corr": float(np.mean(adverse_corr)),
            "race_adverse_corrs": tuple(float(value) for value in adverse_corr),
            "race_delay_corr": float(np.mean(delay_corr)),
            "race_delay_corrs": tuple(float(value) for value in delay_corr),
            "std": float(self.net.embed(embed_context).std(0).mean()),
        }
        self.net.train()
        return totals["total"] / batches, extra

    def log_line(self, epoch, train_loss, val_loss, extra, improved):
        if self.verbose:
            print(
                f"  ep{epoch:>3} train={train_loss:.4f} val={val_loss:.4f} "
                f"skill={extra['skill']:+.3f} "
                f"legR={extra['leg_corr1']:+.3f}/{extra['leg_corr2']:+.3f} "
                f"reachAUC={extra['race_reach_auc']:.3f} "
                f"adverseR={extra['race_adverse_corr']:+.3f} "
                f"delayR={extra['race_delay_corr']:+.3f} "
                f"std={extra['std']:.4f}{'  *' if improved else ''}",
                flush=True)


def train_ssl_nextleg_race(
        big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
        context_lengths=(64, 100, 150, 200), new_channels=8, epochs=60,
        steps_per_epoch=50, batch=512, lr=1e-5, head_lr=1e-4, weight_decay=0.0,
        patience=8, device=None, model_id="paris-noah/Mantis-8M", backbone_ckpt=None,
        warm_trainer_ckpt=None, control="real", seed=0, clamp=10.0, grad_clip=1.0,
        verbose=True, ckpt_path=None, resume=False, freeze_encoder_layers=2, std_guard=1.6,
        leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0, target_reserve=None,
        race_w=0.5, race_cap=8.0, race_levels=RACE_LEVELS, race_scale_lookback=64,
        _stream_layout=None, lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        log_every_steps=25, **_ignore):
    """Train NextLeg plus the causal fixed-scale reach/adverse/time path objective."""
    trainer = _NextLegRaceTrainer(
        big, train_starts, val_starts, horizons=horizons, context_lengths=context_lengths,
        new_channels=new_channels, model_id=model_id, backbone_ckpt=backbone_ckpt,
        warm_trainer_ckpt=warm_trainer_ckpt, clamp=clamp, leg_cap=leg_cap, leg_w=leg_w,
        leg_k=leg_k, mse_weight=mse_weight, race_w=race_w, race_cap=race_cap,
        race_levels=race_levels, race_scale_lookback=race_scale_lookback,
        target_reserve=target_reserve, _stream_layout=_stream_layout,
        epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, head_lr=head_lr,
        weight_decay=weight_decay, patience=patience, device=device, seed=seed,
        grad_clip=grad_clip, verbose=verbose, control=control, ckpt_path=ckpt_path,
        resume=resume, freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        log_every_steps=log_every_steps)
    return trainer.fit()


__all__ = [
    "NextLegRaceNet", "_leg_race_targets", "_leg_race_targets_by_segments",
    "_validate_race_target_reserve", "train_ssl_nextleg_race",
]
