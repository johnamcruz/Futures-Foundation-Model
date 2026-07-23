"""Stage-2.8 v2 NEXT-LEG causal-range path-race trainer."""
from __future__ import annotations

import time
import weakref

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nextleg_race import RACE_LEVELS, scaled_path_race
from .common import _apply_control, _gather_batch
from .forecast import _ForecastTrainer
from .nextleg import NextLegNet


_RACE_TARGET_CACHE = {}


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


def _alternating_fractals_vectorized(high, low, k):
    """Vectorized exact equivalent of NextLeg's pure keep-first fractal sequence."""
    high, low, k = np.asarray(high, np.float64), np.asarray(low, np.float64), int(k)
    if len(high) < 2 * k + 2:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0, np.int8)
    from numpy.lib.stride_tricks import sliding_window_view

    high_window = sliding_window_view(high, 2 * k + 1)[:-1]
    low_window = sliding_window_view(low, 2 * k + 1)[:-1]
    center_high, center_low = high_window[:, k], low_window[:, k]
    is_low = ((center_low == low_window.min(1))
              & ((low_window == center_low[:, None]).sum(1) == 1))
    is_high = ((center_high == high_window.max(1))
               & ((high_window == center_high[:, None]).sum(1) == 1))
    direction_all = np.where(is_low, 1, np.where(is_high, -1, 0)).astype(np.int8)
    rows = np.flatnonzero(direction_all)
    if not len(rows):
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0, np.int8)
    directions = direction_all[rows]
    # NextLeg keep-first semantics skip every repeated same-direction pivot until the opposite
    # type appears. In chronological order this is precisely run-length compression.
    keep = np.r_[True, directions[1:] != directions[:-1]]
    origins = rows[keep].astype(np.int64) + k
    directions = directions[keep]
    return origins, origins + k, directions


def _leg_race_targets(big, k, leg_cap, *, race_levels=RACE_LEVELS,
                      race_scale_lookback=64, race_cap=8.0, chunk_size=8192):
    """Vectorized NextLeg duration and causal-range race targets for one stream."""
    high, low, close = big[:, 1], big[:, 2], big[:, 3]
    levels = np.asarray(tuple(float(value) for value in race_levels), np.float64)
    if not len(levels) or (levels <= 0).any() or (np.diff(levels) <= 0).any():
        raise ValueError("race_levels must be non-empty, positive, and strictly increasing")
    if int(race_scale_lookback) < 2 or int(chunk_size) < 1:
        raise ValueError("race_scale_lookback >= 2 and chunk_size >= 1 are required")
    origins, pivot_confirms, directions = _alternating_fractals_vectorized(high, low, k)
    count, width = max(0, len(origins) - 2), 2 + 3 * len(levels)
    if not count:
        return (
            np.empty(0, np.int64), np.empty((0, width), np.float32),
            np.empty(0, bool))

    confirms = pivot_confirms[:-2]
    next_origins, following_origins = origins[1:-1], origins[2:]
    directions = directions[:-2]
    first = next_origins - confirms
    second = following_origins - next_origins
    valid = ((first > 0) & (second > 0)
             & (first <= int(leg_cap)) & (second <= int(leg_cap))
             & (confirms >= int(race_scale_lookback) - 1))
    targets = np.zeros((count, width), np.float32)
    targets[:, 0] = np.log1p(np.maximum(first, 0))
    targets[:, 1] = np.log1p(np.maximum(second, 0))
    candidate_rows = np.flatnonzero(valid)
    candle_range = np.asarray(high - low, np.float64)
    step_axis = np.arange(int(leg_cap), dtype=np.int64)
    history_axis = np.arange(int(race_scale_lookback) - 1, -1, -1, dtype=np.int64)

    for start in range(0, len(candidate_rows), int(chunk_size)):
        rows = candidate_rows[start:start + int(chunk_size)]
        confirmation = confirms[rows]
        duration = first[rows]
        scale_index = confirmation[:, None] - history_axis[None, :]
        scale = np.median(candle_range[scale_index], axis=1)
        finite_scale = np.isfinite(scale) & (scale > np.finfo(np.float32).eps)
        valid[rows[~finite_scale]] = False
        rows = rows[finite_scale]
        if not len(rows):
            continue
        confirmation = confirms[rows]
        duration = first[rows]
        scale = scale[finite_scale]
        positions = confirmation[:, None] + 1 + step_axis[None, :]
        active = step_axis[None, :] < duration[:, None]
        # Clip only inactive positions so every gather remains in-stream. Active positions end at
        # the next pivot origin by construction.
        positions = np.minimum(positions, next_origins[rows, None])
        reference = close[confirmation, None]
        segment_high, segment_low = high[positions], low[positions]
        long = directions[rows, None] == 1
        favourable = np.where(long, segment_high - reference, reference - segment_low)
        adverse = np.where(long, reference - segment_low, segment_high - reference)
        finite_path = (
            np.where(active, np.isfinite(favourable), True).all(1)
            & np.where(active, np.isfinite(adverse), True).all(1))
        valid[rows[~finite_path]] = False
        rows = rows[finite_path]
        if not len(rows):
            continue
        favourable = favourable[finite_path]
        adverse = np.maximum(adverse[finite_path], 0.0)
        active = active[finite_path]
        duration = duration[finite_path]
        scale = scale[finite_path]
        favourable = np.where(active, favourable, -np.inf)
        adverse = np.where(active, adverse, 0.0)
        hit = favourable[:, :, None] >= scale[:, None, None] * levels[None, None, :]
        reached = hit.any(1)
        first_hit = hit.argmax(1)
        stops = np.where(reached, first_hit, duration[:, None] - 1)
        cumulative_adverse = np.maximum.accumulate(adverse, axis=1)
        risk = np.take_along_axis(cumulative_adverse, stops, axis=1) / scale[:, None]
        delay = np.log1p(stops + 1)
        offset = 2
        targets[rows, offset:offset + len(levels)] = reached.astype(np.float32)
        offset += len(levels)
        targets[rows, offset:offset + len(levels)] = np.minimum(risk, race_cap)
        offset += len(levels)
        targets[rows, offset:offset + len(levels)] = delay
    return confirms.astype(np.int64), targets, valid


def _target_cache_key(big, segments, k, leg_cap, race):
    bars = np.asarray(big)
    levels = tuple(float(value) for value in race.get("race_levels", RACE_LEVELS))
    lookback = int(race.get("race_scale_lookback", 64))
    cap = float(race.get("race_cap", 8.0))
    return (
        int(bars.__array_interface__["data"][0]), bars.shape, bars.strides,
        tuple((int(base), int(size)) for base, size in segments),
        int(k), int(leg_cap), levels, lookback, cap,
    )


def _leg_race_targets_by_segments(big, segments, k, leg_cap, *, verbose=False, **race):
    """Construct pivots independently inside every ticker/timeframe stream."""
    key = _target_cache_key(big, segments, k, leg_cap, race)
    entry = _RACE_TARGET_CACHE.get(key)
    if entry is not None and entry[0]() is np.asarray(big):
        cached = entry[1]
        if verbose:
            print(f"  [nextleg_race:targets] cache hit anchors={len(cached[0]):,}", flush=True)
        return cached
    _RACE_TARGET_CACHE.pop(key, None)
    confirms, targets = [], []
    bars = np.asarray(big, np.float32)
    total_start = time.perf_counter()
    for stream_index, (base, size) in enumerate(segments, start=1):
        base, size = int(base), int(size)
        if base < 0 or size < 0 or base + size > len(bars):
            raise ValueError("stream segment lies outside assembled bars")
        stream_start = time.perf_counter()
        local_confirm, local_target, valid = _leg_race_targets(
            bars[base:base + size], k, leg_cap, **race)
        if valid.any():
            confirms.append(local_confirm[valid] + base)
            targets.append(local_target[valid])
        if verbose:
            print(
                f"  [nextleg_race:targets] stream={stream_index}/{len(segments)} "
                f"bars={size:,} anchors={int(valid.sum()):,} "
                f"elapsed={time.perf_counter() - stream_start:.1f}s",
                flush=True)
    if not targets:
        levels = tuple(race.get("race_levels", RACE_LEVELS))
        return np.empty(0, np.int64), np.empty((0, 2 + 3 * len(levels)), np.float32)
    result = np.concatenate(confirms), np.concatenate(targets)
    def _evict(reference, cache_key=key):
        current = _RACE_TARGET_CACHE.get(cache_key)
        if current is not None and current[0] is reference:
            _RACE_TARGET_CACHE.pop(cache_key, None)

    bars_ref = weakref.ref(bars, _evict)
    _RACE_TARGET_CACHE[key] = (bars_ref, result)
    if verbose:
        print(
            f"  [nextleg_race:targets] complete anchors={len(result[0]):,} "
            f"elapsed={time.perf_counter() - total_start:.1f}s",
            flush=True)
    return result


class NextLegRaceNet(NextLegNet):
    """NextLeg anchor heads plus monotone reach/adverse/delay forecasts."""

    def __init__(self, *args, race_levels=RACE_LEVELS, **kwargs):
        super().__init__(*args, **kwargs)
        self.race_levels = tuple(float(value) for value in race_levels)
        embedding = self.decoder[0].in_features
        self.race_head = nn.Sequential(
            nn.Linear(embedding, embedding // 4), nn.GELU(),
            nn.Linear(embedding // 4, 3 * len(self.race_levels)))

    def task_outputs(self, embedding):
        """Decode disposable SSL task outputs from one already-computed embedding."""
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
        return self.task_outputs(self.embed(context))


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


class _NextLegRaceTrainer(_ForecastTrainer):
    def __init__(self, big, tr, va, *, leg_cap=256, leg_w=1.0, leg_k=2,
                 mse_weight=1.0, race_w=0.5, race_cap=8.0,
                 race_levels=RACE_LEVELS, race_scale_lookback=64,
                 target_reserve=None, _stream_layout=None, head_lr=None, **forecast):
        if _stream_layout is None:
            raise ValueError("nextleg_race requires exact assembled stream layout")
        _ForecastTrainer.__init__(self, big, tr, va, **forecast)
        self.leg_w, self.mse_weight, self.race_w = (
            float(leg_w), float(mse_weight), float(race_w))
        self.race_cap = float(race_cap)
        self.race_levels = tuple(float(value) for value in race_levels)
        self.race_scale_lookback = int(race_scale_lookback)
        self.head_lr = float(head_lr) if head_lr is not None else 10.0 * float(self.lr)
        if self.race_w <= 0 or self.head_lr <= 0:
            raise ValueError("race_w and head_lr must be positive")
        if self.race_scale_lookback > min(self.clens):
            raise ValueError("race_scale_lookback must fit the shortest context")

        segments = tuple((stream.base, stream.size) for stream in _stream_layout.streams)
        confirms, targets = _leg_race_targets_by_segments(
            np.asarray(big, np.float32), segments, int(leg_k), int(leg_cap),
            race_levels=self.race_levels, race_scale_lookback=self.race_scale_lookback,
            race_cap=self.race_cap, verbose=self.verbose)
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
        control="real", seed=0, clamp=10.0, grad_clip=1.0,
        verbose=True, ckpt_path=None, resume=False, freeze_encoder_layers=2, std_guard=1.6,
        leg_cap=256, leg_w=1.0, leg_k=2, mse_weight=1.0, target_reserve=None,
        race_w=0.5, race_cap=8.0, race_levels=RACE_LEVELS, race_scale_lookback=64,
        _stream_layout=None, lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        log_every_steps=25, **_ignore):
    """Train NextLeg plus the causal fixed-scale reach/adverse/time path objective."""
    trainer = _NextLegRaceTrainer(
        big, train_starts, val_starts, horizons=horizons, context_lengths=context_lengths,
        new_channels=new_channels, model_id=model_id, backbone_ckpt=backbone_ckpt,
        clamp=clamp, leg_cap=leg_cap, leg_w=leg_w,
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
