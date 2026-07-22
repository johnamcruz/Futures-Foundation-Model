"""NextLeg with causal multi-timeframe and sibling context fusion.

This is an opt-in experiment. The primary stream still owns every candle/leg target; related
series provide context only. All members share one compact Mantis encoder, then a small masked
attention block fuses their embeddings. Existing ``nextleg`` behavior and artifacts are untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _apply_control, _gather_batch, _time_shuffle
from .nextleg import _NextLegTrainer
from .related_series import (RelatedMantisEncoder, load_related_checkpoint,
                             plain_encoder_state, related_checkpoint_state)


def _gather_related(big, starts, length):
    """Global starts ``[B,R]`` -> related windows ``[B,R,C,L]``."""
    batch, related = starts.shape
    flat = starts.reshape(-1)
    rows = flat[:, None] + torch.arange(length, device=big.device)[None, :]
    windows = big[rows].reshape(batch, related, length, big.shape[1])
    return windows.permute(0, 1, 3, 2).contiguous()


def _apply_related_control(contexts, mask, control, *, generator=None):
    """Corrupt/drop related members only; primary slot zero remains byte-identical."""
    mode = str(control or "real").lower()
    if mode == "real":
        return contexts, mask
    output, output_mask = contexts.clone(), mask.clone()
    if mode == "drop":
        output_mask[:, 1:] = False
    elif mode == "shuffle":
        shape = output[:, 1:].shape
        output[:, 1:] = _time_shuffle(output[:, 1:].reshape(-1, shape[2], shape[3])).reshape(shape)
    elif mode == "random":
        output[:, 1:] = torch.randn(output[:, 1:].shape, device=output.device,
                                    dtype=output.dtype, generator=generator)
    else:
        raise ValueError(f"unknown related_control={control!r}; expected real/shuffle/random/drop")
    return output, output_mask


class RelatedNextLegNet(nn.Module):
    """Shared Mantis related encoder plus the incumbent candle and leg heads."""

    def __init__(self, *, C=5, horizons=(5, 10, 20, 25), model_id="paris-noah/Mantis-8M",
                 num_roles=6, related_heads=4, related_dropout=0.0):
        super().__init__()
        from .common import load_mantis
        mantis = load_mantis(model_id)
        self.related_encoder = RelatedMantisEncoder(
            mantis, channels=C, num_roles=num_roles, num_heads=related_heads,
            dropout=related_dropout)
        self.C, self.horizons = int(C), tuple(int(h) for h in horizons)
        self.nH = len(self.horizons)
        emb = self.related_encoder.embed_dim
        self.decoder = nn.Sequential(nn.Linear(emb, emb), nn.GELU(),
                                     nn.Linear(emb, self.C * self.nH))
        self.leg_head = nn.Sequential(nn.Linear(emb, emb // 4), nn.GELU(),
                                      nn.Linear(emb // 4, 2))

    @property
    def encoder(self):
        """Expose the underlying Mantis module to shared LoRA/freeze machinery."""
        return self.related_encoder.mantis

    def embed(self, contexts, mask, role_ids):
        return self.related_encoder(contexts, mask, role_ids, standardized=True)

    def forward_all(self, contexts, mask, role_ids):
        embedding = self.embed(contexts, mask, role_ids)
        candles = self.decoder(embedding).view(-1, self.C, self.nH)
        return candles, self.leg_head(embedding)


class _RelatedNextLegTrainer(_NextLegTrainer):
    def __init__(self, big, tr, va, *, _related_layout=None,
                 related_tfs=("1min", "3min", "5min", "15min"),
                 related_siblings="default", related_heads=4, related_dropout=0.0,
                 related_max_gap_factor=2.0, related_control="real", **kwargs):
        if _related_layout is None:
            raise ValueError("related_nextleg requires assembled related-series metadata")
        super().__init__(big, tr, va, **kwargs)
        self.related_heads = int(related_heads)
        self.related_dropout = float(related_dropout)
        self.related_tfs = tuple(related_tfs)
        self.related_siblings = related_siblings
        self.related_max_gap_factor = float(related_max_gap_factor)
        self.related_control = str(related_control)
        self.tr_plan = _related_layout.align(
            self.tr.detach().cpu().numpy(), self.max_ctx, related_tfs=self.related_tfs,
            siblings=self.related_siblings, max_gap_factor=self.related_max_gap_factor)
        self.va_plan = _related_layout.align(
            self.va.detach().cpu().numpy(), self.max_ctx, related_tfs=self.related_tfs,
            siblings=self.related_siblings, max_gap_factor=self.related_max_gap_factor)
        self.tr_related = torch.as_tensor(self.tr_plan.starts, device=self.dev)
        self.va_related = torch.as_tensor(self.va_plan.starts, device=self.dev)
        self.tr_mask = torch.as_tensor(self.tr_plan.mask, device=self.dev)
        self.va_mask = torch.as_tensor(self.va_plan.mask, device=self.dev)
        self.role_ids = torch.as_tensor(self.tr_plan.role_ids, device=self.dev)
        if self.verbose:
            coverage = " ".join(f"{name}={frac:.1%}"
                                for name, frac in self.tr_plan.valid_fraction().items())
            print(f"  [related] roles={self.tr_plan.num_roles} {coverage}", flush=True)

    def build_net(self):
        net = RelatedNextLegNet(
            C=self.C, horizons=self.hlist, model_id=self.model_id,
            num_roles=self.tr_plan.num_roles, related_heads=self.related_heads,
            related_dropout=self.related_dropout).to(self.dev)
        if self.backbone_ckpt:
            state = torch.load(self.backbone_ckpt, map_location="cpu")
            net.encoder.load_state_dict(plain_encoder_state(state))
        self.net = net

    def snapshot_state(self):
        state = related_checkpoint_state(self.net.related_encoder)
        state["config"].update({
            "role_names": self.tr_plan.role_names,
            "related_tfs": self.related_tfs,
            "related_siblings": self.related_siblings,
            "max_gap_factor": self.related_max_gap_factor,
            "related_control": self.related_control,
        })
        return state

    def load_snapshot_state(self, state):
        load_related_checkpoint(self.net.related_encoder, state)

    def make_batch(self, starts):
        is_train = starts is self.tr
        related = self.tr_related if is_train else self.va_related
        masks = self.tr_mask if is_train else self.va_mask
        targets = self._tgt_tr if is_train else self._tgt_va
        b_idx = self.sample_indices(starts)

        primary = _gather_batch(self.big_t, starts, b_idx, self.parent)
        grouped = _gather_related(self.big_t, related[b_idx], self.max_ctx)
        length = int(self.clens_t[torch.randint(
            0, len(self.clens_t), (1,), device=self.dev, generator=self.gen)].item())
        contexts = grouped[:, :, :, -length:]
        future = primary[:, :, self.max_ctx:]
        primary_context = primary[:, :, self.max_ctx - length:self.max_ctx]

        # Standardize each series/channel using only its own historical context. Targets retain
        # the incumbent primary-stream scale so the objective is directly comparable to NextLeg.
        mean = contexts.mean(3, keepdim=True)
        std = contexts.std(3, keepdim=True) + 1e-6
        contexts = ((contexts - mean) / std).clamp(-self.clamp, self.clamp)
        p_mean = primary_context.mean(2, keepdim=True)
        p_std = primary_context.std(2, keepdim=True) + 1e-6
        future_std = ((future - p_mean) / p_std).clamp(-self.clamp, self.clamp)
        primary_std = ((primary_context - p_mean) / p_std).clamp(-self.clamp, self.clamp)
        candle_target = future_std[:, :, self.h_off] - primary_std[:, :, -1:]

        contexts, batch_mask = _apply_related_control(
            contexts, masks[b_idx], self.related_control, generator=self.gen)
        flat = contexts.flatten(0, 1)
        contexts = _apply_control(flat, self.control).reshape_as(contexts)
        return contexts, batch_mask, candle_target, targets[b_idx]

    def compute_loss(self, batch):
        contexts, mask, candle_target, leg_target = batch
        candles, legs = self.net.forward_all(contexts, mask, self.role_ids)
        candle_loss = F.mse_loss(candles.float(), candle_target)
        leg_loss = F.smooth_l1_loss(legs.float(), leg_target)
        return self.mse_weight * candle_loss + self.leg_w * leg_loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval()
        total_candle = total_persist = total_leg = 0.0
        predictions, targets = [], []
        batches = min(20, max(1, len(self.va) // self.batch))
        for _ in range(batches):
            contexts, mask, candle_target, leg_target = self.make_batch(self.va)
            candles, legs = self.net.forward_all(contexts, mask, self.role_ids)
            total_candle += float(F.mse_loss(candles.float(), candle_target))
            total_persist += float((candle_target ** 2).mean())
            total_leg += float(F.smooth_l1_loss(legs.float(), leg_target))
            predictions.append(legs.float().cpu())
            targets.append(leg_target.cpu())
        pred, target = torch.cat(predictions), torch.cat(targets)
        corr = [float(np.corrcoef(pred[:, j].numpy(), target[:, j].numpy())[0, 1])
                for j in (0, 1)]
        sample = self.make_batch(self.va)
        emb_std = float(self.net.embed(sample[0], sample[1], self.role_ids).std(0).mean())
        skill = 1.0 - (total_candle / batches) / max(total_persist / batches, 1e-12)
        val_loss = (self.mse_weight * total_candle / batches
                    + self.leg_w * total_leg / batches)
        self.net.train()
        return val_loss, {"skill": skill, "leg_corr1": corr[0], "leg_corr2": corr[1],
                          "std": emb_std, "fusion_gate": float(torch.tanh(
                              self.net.related_encoder.fusion.gate).cpu())}

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"skill={extra['skill']:+.3f} legR={extra['leg_corr1']:+.3f}/"
                  f"{extra['leg_corr2']:+.3f} gate={extra['fusion_gate']:+.3f} "
                  f"emb_std={extra['std']:.4f}{'  *' if improved else ''}", flush=True)


def train_ssl_related_nextleg(big, train_starts, val_starts, *,
                              horizons=(5, 10, 20, 25),
                              context_lengths=(64, 100, 150, 200), new_channels=8,
                              epochs=60, steps_per_epoch=200, batch=512, lr=1e-4,
                              weight_decay=0.05, patience=8, device=None,
                              model_id="paris-noah/Mantis-8M", backbone_ckpt=None,
                              control="real", seed=0, clamp=10.0, grad_clip=1.0,
                              verbose=True, ckpt_path=None, resume=False,
                              freeze_encoder_layers=0, std_guard=1.6, leg_cap=256,
                              leg_w=1.0, leg_k=2, mse_weight=1.0,
                              target_reserve=None, lora_r=0,
                              lora_alpha=16.0, lora_dropout=0.0, log_every_steps=25,
                              _related_layout=None,
                              related_tfs=("1min", "3min", "5min", "15min"),
                              related_siblings="default", related_heads=4,
                              related_dropout=0.0, related_max_gap_factor=2.0,
                              related_control="real", **_ignore):
    if int(new_channels) < int(np.asarray(big).shape[1]):
        raise ValueError("related_nextleg preserves every OHLCV channel; new_channels must be >= C")
    trainer = _RelatedNextLegTrainer(
        big, train_starts, val_starts, horizons=horizons, context_lengths=context_lengths,
        new_channels=new_channels, model_id=model_id, backbone_ckpt=backbone_ckpt,
        clamp=clamp, leg_cap=leg_cap, leg_w=leg_w, leg_k=leg_k, mse_weight=mse_weight,
        target_reserve=target_reserve,
        epochs=epochs, steps_per_epoch=steps_per_epoch, batch=batch, lr=lr,
        weight_decay=weight_decay, patience=patience, device=device, seed=seed,
        grad_clip=grad_clip, verbose=verbose, control=control, ckpt_path=ckpt_path,
        resume=resume, freeze_encoder_layers=freeze_encoder_layers, std_guard=std_guard,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        log_every_steps=log_every_steps, _related_layout=_related_layout,
        related_tfs=related_tfs, related_siblings=related_siblings,
        related_heads=related_heads, related_dropout=related_dropout,
        related_max_gap_factor=related_max_gap_factor, related_control=related_control)
    return trainer.fit()


__all__ = ["RelatedNextLegNet", "train_ssl_related_nextleg", "_apply_related_control"]
