"""Torch trainer for causal momentum-strength/volatility transition SSL."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _apply_control, _encode_channels, _gather_batch, load_mantis
from .forecast import _ForecastTrainer
from .nextleg_race import _binary_auc, _correlation
from ..momentum_volatility import COUPLING_CLASSES


class MomentumVolatilityNet(nn.Module):
    """Disposable linear supervision attached directly to the production encoder path.

    There is intentionally no trainable channel adapter or nonlinear task tower. The same
    per-channel encoder representation exported for downstream inference receives every MV
    gradient; temporary heads cannot hide task learning in discarded parameters.
    """

    def __init__(self, C=5, new_channels=3, horizons=(5, 10, 20, 25),
                 model_id="paris-noah/Mantis-8M", aux_dim=0, coupling_classes=4):
        super().__init__()
        del new_channels, aux_dim
        self.encoder = load_mantis(model_id)
        hidden = getattr(self.encoder, "hidden_dim", 256)
        self.C = int(C)
        self.horizons = tuple(int(value) for value in horizons)
        self.nH = len(self.horizons)
        embedding = hidden * self.C
        self.coupling_classes = int(coupling_classes)
        self.decoder = nn.Linear(embedding, self.C * self.nH)
        self.mv_head = nn.Linear(
            embedding, self.nH * (2 + self.coupling_classes))

    def embed(self, context):
        return _encode_channels(self.encoder, context)

    def forward_mv(self, context, *, return_embedding=False):
        embedding = self.embed(context)
        candles = self.decoder(embedding).view(-1, self.C, self.nH)
        raw = self.mv_head(embedding).view(
            -1, self.nH, 2 + self.coupling_classes)
        output = (candles, raw[:, :, 0], raw[:, :, 1], raw[:, :, 2:])
        return (*output, embedding) if return_embedding else output


def _transition_contrastive_loss(embedding, state, temperature=.1):
    """Supervised contrastive loss applied to exported encoder embeddings directly."""
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    features = F.normalize(embedding.float(), dim=1)
    logits = features @ features.T / float(temperature)
    count = logits.shape[0]
    eye = torch.eye(count, dtype=torch.bool, device=logits.device)
    positive = (state[:, None] == state[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    logits = logits.masked_fill(eye, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_probability.masked_fill(~positive, 0.0).sum(dim=1)
                   / positive.sum(dim=1).clamp_min(1))
    counts = torch.bincount(state, minlength=4).float().clamp_min(1)
    anchor_weight = counts[state].reciprocal()
    anchor_weight = anchor_weight * valid
    return (per_anchor * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-12)


def _balanced_transition_cross_entropy(logits, truth, classes=4):
    """Give every MV state equal batch influence without resampling time."""
    flat_truth = truth.reshape(-1)
    counts = torch.bincount(flat_truth, minlength=int(classes)).float()
    weights = torch.where(
        counts > 0,
        counts.sum() / (counts.clamp_min(1) * int(classes)),
        torch.zeros_like(counts))
    return F.cross_entropy(
        logits.float().reshape(-1, int(classes)), flat_truth, weight=weights)


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
    momentum_strength, volatility, coupling = [], [], []
    for offset in horizon_offsets.tolist():
        endpoint = future_raw[:, close_channel, offset]
        future_close = future_raw[:, close_channel, :offset + 1]
        path = torch.cat((now[:, None], future_close), dim=1)
        steps = path[:, 1:] - path[:, :-1]
        strength = (endpoint - now).abs() / steps.abs().sum(dim=1).clamp_min(1e-6)
        future_ranges = (
            future_raw[:, high_channel, :offset + 1]
            - future_raw[:, low_channel, :offset + 1])
        ratio = median(future_ranges).clamp_min(1e-6) / scale
        expanding = ratio >= float(expansion_threshold)
        directional = strength >= float(momentum_threshold)
        state = torch.full_like(strength, 3, dtype=torch.long)
        state = torch.where(~directional & expanding, 2, state)
        state = torch.where(directional & ~expanding, 1, state)
        state = torch.where(directional & expanding, 0, state)
        momentum_strength.append(strength)
        volatility.append(ratio.log())
        coupling.append(state)
    return (
        torch.stack(momentum_strength, dim=1),
        torch.stack(volatility, dim=1),
        torch.stack(coupling, dim=1),
    )


class _MomentumVolatilityTrainer(_ForecastTrainer):
    def __init__(
            self, big, tr, va, *, scale_lookback=64, momentum_lookback=20,
            momentum_threshold=.5, expansion_threshold=1.1,
            candle_weight=.25, momentum_weight=1.0, volatility_weight=.5,
            coupling_weight=.5, transition_contrastive_weight=.1,
            parent_retention_weight=.5, contrastive_temperature=.1,
            head_lr=None, **forecast):
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
            candle_weight, momentum_weight, volatility_weight, coupling_weight,
            transition_contrastive_weight, parent_retention_weight))
        if any(value < 0 for value in self.weights) or sum(self.weights) <= 0:
            raise ValueError("loss weights must be non-negative and not all zero")
        self.contrastive_temperature = float(contrastive_temperature)
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        self.head_lr = float(head_lr) if head_lr is not None else self.lr
        self.parent_encoder = None

    def build_net(self):
        if not self.backbone_ckpt:
            raise ValueError(
                "momentum-volatility v3 requires a frozen parent encoder checkpoint")
        parent_state = torch.load(
            self.backbone_ckpt, map_location="cpu", weights_only=False)
        net = MomentumVolatilityNet(
            C=self.C, new_channels=self.new_channels, horizons=self.hlist,
            model_id=self.model_id, aux_dim=0).to(self.dev)
        net.encoder.load_state_dict(parent_state)
        teacher = load_mantis(self.model_id).to(self.dev)
        teacher.load_state_dict(parent_state)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        self.parent_encoder = teacher
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
        momentum_strength, volatility, coupling = _momentum_volatility_targets_torch(
            context_raw, future_raw, self.h_off,
            scale_lookback=self.scale_lookback,
            momentum_lookback=self.momentum_lookback,
            momentum_threshold=self.momentum_threshold,
            expansion_threshold=self.expansion_threshold)
        return (
            _apply_control(context, self.control),
            candle_target, momentum_strength, volatility, coupling)

    def _losses(self, batch):
        context, candle_truth, strength_truth, volatility_truth, coupling_truth = batch
        candles, strength, volatility, coupling, embedding = self.net.forward_mv(
            context, return_embedding=True)
        if self.parent_encoder is None:
            raise RuntimeError("frozen parent encoder was not initialized")
        with torch.no_grad():
            parent_embedding = _encode_channels(self.parent_encoder, context)
        parent_energy = parent_embedding.float().square().mean().clamp_min(1e-6)
        parent_relative_mse = F.mse_loss(
            embedding.float(), parent_embedding.float()) / parent_energy
        parts = {
            "candle": F.mse_loss(candles.float(), candle_truth),
            "momentum_strength": F.smooth_l1_loss(
                strength.float(), strength_truth),
            "volatility": F.smooth_l1_loss(volatility.float(), volatility_truth),
            "coupling": _balanced_transition_cross_entropy(
                coupling, coupling_truth),
            "contrastive": _transition_contrastive_loss(
                embedding, coupling_truth[:, -1],
                temperature=self.contrastive_temperature),
            "retention": parent_relative_mse,
            "parent_cosine": F.cosine_similarity(
                embedding.float(), parent_embedding.float(), dim=1).mean(),
        }
        total = sum(weight * parts[name] for weight, name in zip(
            self.weights,
            ("candle", "momentum_strength", "volatility", "coupling",
             "contrastive", "retention")))
        return total, (candles, strength, volatility, coupling), parts

    def compute_loss(self, batch):
        return self._losses(batch)[0]

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        generator = torch.Generator(device=self.dev); generator.manual_seed(20260723)
        prediction = {"momentum_strength": [], "volatility": [], "coupling": []}
        truth = {key: [] for key in prediction}
        total = candle = persist = retention = parent_cosine = 0.0
        batches = min(20, max(1, len(self.va) // self.batch))
        embedding_context = None
        for _ in range(batches):
            batch = self.make_batch(self.va, gen=generator)
            if embedding_context is None:
                embedding_context = batch[0]
            loss, output, parts = self._losses(batch)
            _candles, momentum_strength, volatility, coupling = output
            total += float(loss)
            candle += float(parts["candle"])
            persist += float((batch[1] ** 2).mean())
            retention += float(parts["retention"])
            parent_cosine += float(parts["parent_cosine"])
            for name, pred, target in (
                    ("momentum_strength", momentum_strength, batch[2]),
                    ("volatility", volatility, batch[3]),
                    ("coupling", coupling.softmax(-1), batch[4])):
                prediction[name].append(pred.float().cpu())
                truth[name].append(target.cpu())
        prediction = {key: torch.cat(value).numpy() for key, value in prediction.items()}
        truth = {key: torch.cat(value).numpy() for key, value in truth.items()}
        momentum_strength_corr = np.mean([
            _correlation(
                prediction["momentum_strength"][:, i],
                truth["momentum_strength"][:, i])
            for i in range(len(self.hlist))])
        volatility_corr = np.mean([
            _correlation(prediction["volatility"][:, i], truth["volatility"][:, i])
            for i in range(len(self.hlist))])
        per_class_auc = {}
        per_class_rate = {}
        for label, name in enumerate(COUPLING_CLASSES):
            per_class_auc[name] = float(np.mean([
                _binary_auc(
                    truth["coupling"][:, horizon] == label,
                    prediction["coupling"][:, horizon, label])
                for horizon in range(len(self.hlist))]))
            per_class_rate[name] = float(np.mean(truth["coupling"] == label))
        extra = {
            "skill": 1.0 - candle / max(persist, 1e-12),
            "mv_momentum_strength_corr": float(momentum_strength_corr),
            "mv_volatility_corr": float(volatility_corr),
            "mv_transition_auc": float(np.mean(list(per_class_auc.values()))),
            "mv_transition_worst_auc": float(min(per_class_auc.values())),
            "mv_transition_auc_per_class": per_class_auc,
            "mv_transition_class_rate": per_class_rate,
            "mv_parent_relative_rmse": float(
                np.sqrt(max(retention / batches, 0.0))),
            "mv_parent_cosine": float(parent_cosine / batches),
            "std": float(self.net.embed(embedding_context).std(0).mean()),
        }
        self.net.train()
        return total / batches, extra

    def log_line(self, epoch, train_loss, val_loss, extra, improved):
        if self.verbose:
            print(
                f"  ep{epoch:>3} train={train_loss:.4f} val={val_loss:.4f} "
                f"skill={extra['skill']:+.3f} "
                f"strengthR={extra['mv_momentum_strength_corr']:+.3f} "
                f"volatilityR={extra['mv_volatility_corr']:+.3f} "
                f"transitionAUC={extra['mv_transition_auc']:.3f}/"
                f"{extra['mv_transition_worst_auc']:.3f}worst "
                f"parent={extra['mv_parent_cosine']:.4f}/"
                f"{extra['mv_parent_relative_rmse']:.4f}rmse "
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
        coupling_weight=.5, transition_contrastive_weight=.1,
        parent_retention_weight=.5, contrastive_temperature=.1,
        lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
        log_every_steps=25, **_ignore):
    """Refine a warm FFM encoder on causal momentum-strength/volatility transitions."""
    return _MomentumVolatilityTrainer(
        big, train_starts, val_starts, horizons=horizons,
        context_lengths=context_lengths, new_channels=new_channels,
        model_id=model_id, backbone_ckpt=backbone_ckpt,
        scale_lookback=scale_lookback, momentum_lookback=momentum_lookback,
        momentum_threshold=momentum_threshold, expansion_threshold=expansion_threshold,
        candle_weight=candle_weight, momentum_weight=momentum_weight,
        volatility_weight=volatility_weight, coupling_weight=coupling_weight,
        transition_contrastive_weight=transition_contrastive_weight,
        parent_retention_weight=parent_retention_weight,
        contrastive_temperature=contrastive_temperature,
        epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch,
        lr=lr, head_lr=head_lr, weight_decay=weight_decay, patience=patience,
        device=device, seed=seed, grad_clip=grad_clip, verbose=verbose,
        control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        clamp=clamp, log_every_steps=log_every_steps).fit()


__all__ = [
    "MomentumVolatilityNet",
    "_transition_contrastive_loss",
    "_balanced_transition_cross_entropy",
    "_momentum_volatility_targets_torch",
    "train_ssl_momentum_volatility",
]
