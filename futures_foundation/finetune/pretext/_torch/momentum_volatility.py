"""Torch trainer for the causal momentum-volatility coupling SSL objective."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _apply_control, _gather_batch
from .forecast import MultiHorizonForecastNet, _ForecastTrainer
from .nextleg_race import _binary_auc, _correlation


class MomentumVolatilityNet(MultiHorizonForecastNet):
    """Shared FFM embedding with candle, momentum, volatility, and coupling readouts."""

    def __init__(self, *args, coupling_classes=4, **kwargs):
        super().__init__(*args, **kwargs)
        embedding = self.decoder[0].in_features
        self.coupling_classes = int(coupling_classes)
        self.mv_head = nn.Sequential(
            nn.Linear(embedding, embedding // 4),
            nn.GELU(),
            nn.Linear(
                embedding // 4,
                self.nH * (2 + self.coupling_classes),
            ),
        )

    def forward_mv(self, context):
        embedding = self.embed(context)
        candles = self.decoder(embedding).view(-1, self.C, self.nH)
        raw = self.mv_head(embedding).view(
            -1, self.nH, 2 + self.coupling_classes)
        return candles, raw[:, :, 0], raw[:, :, 1], raw[:, :, 2:]


def _momentum_volatility_targets_torch(
        context_raw, future_raw, horizon_offsets, *,
        scale_lookback=64, momentum_lookback=20,
        momentum_threshold=.5, expansion_threshold=1.1):
    """Vectorized targets; every scale/input term ends on the decision candle."""
    close_channel = min(3, context_raw.shape[1] - 1)
    high_channel = min(1, context_raw.shape[1] - 1)
    low_channel = min(2, context_raw.shape[1] - 1)
    if scale_lookback > context_raw.shape[-1]:
        raise ValueError("scale_lookback must fit the sampled context")
    if momentum_lookback >= context_raw.shape[-1]:
        raise ValueError("momentum_lookback must be shorter than sampled context")

    def median(values):
        ordered = values.sort(dim=1).values
        width = ordered.shape[1]
        if width % 2:
            return ordered[:, width // 2]
        return .5 * (ordered[:, width // 2 - 1] + ordered[:, width // 2])

    causal_ranges = (
        context_raw[:, high_channel, -scale_lookback:]
        - context_raw[:, low_channel, -scale_lookback:])
    scale = median(causal_ranges).clamp_min(1e-6)
    now = context_raw[:, close_channel, -1]
    past = (now - context_raw[:, close_channel, -1 - momentum_lookback]) / scale
    momentum, volatility, coupling = [], [], []
    for offset in horizon_offsets.tolist():
        endpoint = future_raw[:, close_channel, offset]
        move = (endpoint - now) / scale
        future_ranges = (
            future_raw[:, high_channel, :offset + 1]
            - future_raw[:, low_channel, :offset + 1])
        ratio = median(future_ranges).clamp_min(1e-6) / scale
        expanding = ratio >= float(expansion_threshold)
        directional = move.abs() >= float(momentum_threshold)
        has_past = past.abs() >= float(momentum_threshold)
        same = torch.sign(move) == torch.sign(past)
        state = torch.zeros_like(move, dtype=torch.long)
        state = torch.where(expanding & directional & ~has_past, 3, state)
        state = torch.where(expanding & directional & has_past & same, 1, state)
        state = torch.where(expanding & directional & has_past & ~same, 2, state)
        momentum.append(move)
        volatility.append(ratio.log())
        coupling.append(state)
    return (
        torch.stack(momentum, dim=1),
        torch.stack(volatility, dim=1),
        torch.stack(coupling, dim=1),
    )


class _MomentumVolatilityTrainer(_ForecastTrainer):
    def __init__(
            self, big, tr, va, *, scale_lookback=64, momentum_lookback=20,
            momentum_threshold=.5, expansion_threshold=1.1,
            candle_weight=.25, momentum_weight=1.0, volatility_weight=.5,
            coupling_weight=.5, head_lr=None, **forecast):
        super().__init__(big, tr, va, **forecast)
        self.scale_lookback = int(scale_lookback)
        self.momentum_lookback = int(momentum_lookback)
        if self.scale_lookback > min(self.clens):
            raise ValueError("scale_lookback must fit the shortest context")
        if self.momentum_lookback >= min(self.clens):
            raise ValueError("momentum_lookback must be shorter than the shortest context")
        self.momentum_threshold = float(momentum_threshold)
        self.expansion_threshold = float(expansion_threshold)
        self.weights = tuple(float(value) for value in (
            candle_weight, momentum_weight, volatility_weight, coupling_weight))
        if any(value < 0 for value in self.weights) or sum(self.weights) <= 0:
            raise ValueError("loss weights must be non-negative and not all zero")
        self.head_lr = float(head_lr) if head_lr is not None else 10.0 * self.lr

    def build_net(self):
        net = MomentumVolatilityNet(
            C=self.C, new_channels=self.new_channels, horizons=self.hlist,
            model_id=self.model_id, aux_dim=0).to(self.dev)
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
            ({"params": encoder, "lr": self.lr},
             {"params": heads, "lr": self.head_lr}),
            weight_decay=self.weight_decay)

    def make_batch(self, starts, gen=None):
        gen = gen or self.gen
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
        momentum, volatility, coupling = _momentum_volatility_targets_torch(
            context_raw, future_raw, self.h_off,
            scale_lookback=self.scale_lookback,
            momentum_lookback=self.momentum_lookback,
            momentum_threshold=self.momentum_threshold,
            expansion_threshold=self.expansion_threshold)
        return (
            _apply_control(context, self.control),
            candle_target, momentum, volatility, coupling)

    def _losses(self, batch):
        context, candle_truth, momentum_truth, volatility_truth, coupling_truth = batch
        candles, momentum, volatility, coupling = self.net.forward_mv(context)
        parts = {
            "candle": F.mse_loss(candles.float(), candle_truth),
            "momentum": F.smooth_l1_loss(momentum.float(), momentum_truth),
            "volatility": F.smooth_l1_loss(volatility.float(), volatility_truth),
            "coupling": F.cross_entropy(
                coupling.float().reshape(-1, 4), coupling_truth.reshape(-1)),
        }
        total = sum(weight * parts[name] for weight, name in zip(
            self.weights, ("candle", "momentum", "volatility", "coupling")))
        return total, (candles, momentum, volatility, coupling), parts

    def compute_loss(self, batch):
        return self._losses(batch)[0]

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        generator = torch.Generator(device=self.dev); generator.manual_seed(20260723)
        prediction = {"momentum": [], "volatility": [], "coupling": []}
        truth = {key: [] for key in prediction}
        total = candle = persist = 0.0
        batches = min(20, max(1, len(self.va) // self.batch))
        embedding_context = None
        for _ in range(batches):
            batch = self.make_batch(self.va, gen=generator)
            if embedding_context is None:
                embedding_context = batch[0]
            loss, output, parts = self._losses(batch)
            _candles, momentum, volatility, coupling = output
            total += float(loss)
            candle += float(parts["candle"])
            persist += float((batch[1] ** 2).mean())
            for name, pred, target in (
                    ("momentum", momentum, batch[2]),
                    ("volatility", volatility, batch[3]),
                    ("coupling", coupling.softmax(-1), batch[4])):
                prediction[name].append(pred.float().cpu())
                truth[name].append(target.cpu())
        prediction = {key: torch.cat(value).numpy() for key, value in prediction.items()}
        truth = {key: torch.cat(value).numpy() for key, value in truth.items()}
        momentum_corr = np.mean([
            _correlation(prediction["momentum"][:, i], truth["momentum"][:, i])
            for i in range(len(self.hlist))])
        volatility_corr = np.mean([
            _correlation(prediction["volatility"][:, i], truth["volatility"][:, i])
            for i in range(len(self.hlist))])
        aucs = []
        for horizon in range(len(self.hlist)):
            for label in range(4):
                aucs.append(_binary_auc(
                    truth["coupling"][:, horizon] == label,
                    prediction["coupling"][:, horizon, label]))
        extra = {
            "skill": 1.0 - candle / max(persist, 1e-12),
            "mv_momentum_corr": float(momentum_corr),
            "mv_volatility_corr": float(volatility_corr),
            "mv_coupling_auc": float(np.mean(aucs)),
            "std": float(self.net.embed(embedding_context).std(0).mean()),
        }
        self.net.train()
        return total / batches, extra

    def log_line(self, epoch, train_loss, val_loss, extra, improved):
        if self.verbose:
            print(
                f"  ep{epoch:>3} train={train_loss:.4f} val={val_loss:.4f} "
                f"skill={extra['skill']:+.3f} "
                f"momentumR={extra['mv_momentum_corr']:+.3f} "
                f"volatilityR={extra['mv_volatility_corr']:+.3f} "
                f"couplingAUC={extra['mv_coupling_auc']:.3f} "
                f"std={extra['std']:.4f}{'  *' if improved else ''}",
                flush=True)


def train_ssl_momentum_volatility(
        big, train_starts, val_starts, *, horizons=(5, 10, 20, 25),
        context_lengths=(64, 100, 150, 200), new_channels=3, epochs=60,
        steps_per_epoch=50, batch=512, lr=1e-5, head_lr=1e-4,
        weight_decay=0.0, patience=8, device=None,
        model_id="paris-noah/Mantis-8M", backbone_ckpt=None,
        control="real", seed=0, clamp=10.0, grad_clip=1.0, verbose=True,
        ckpt_path=None, resume=False, freeze_encoder_layers=2, std_guard=1.6,
        scale_lookback=64, momentum_lookback=20,
        momentum_threshold=.5, expansion_threshold=1.1,
        candle_weight=.25, momentum_weight=1.0, volatility_weight=.5,
        coupling_weight=.5, lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        log_every_steps=25, **_ignore):
    """Refine a warm FFM encoder on causal momentum-volatility coupling."""
    return _MomentumVolatilityTrainer(
        big, train_starts, val_starts, horizons=horizons,
        context_lengths=context_lengths, new_channels=new_channels,
        model_id=model_id, backbone_ckpt=backbone_ckpt,
        scale_lookback=scale_lookback, momentum_lookback=momentum_lookback,
        momentum_threshold=momentum_threshold, expansion_threshold=expansion_threshold,
        candle_weight=candle_weight, momentum_weight=momentum_weight,
        volatility_weight=volatility_weight, coupling_weight=coupling_weight,
        epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
        lr=lr, head_lr=head_lr, weight_decay=weight_decay, patience=patience,
        device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        clamp=clamp, log_every_steps=log_every_steps).fit()


__all__ = [
    "MomentumVolatilityNet",
    "_momentum_volatility_targets_torch",
    "train_ssl_momentum_volatility",
]
