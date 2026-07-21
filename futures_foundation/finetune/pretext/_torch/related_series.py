"""Small Mantis-native related-series fusion.

The architectural idea mirrors Chronos-2 group attention while preserving the compact Mantis
encoder: every related OHLCV series is encoded by the *same* Mantis weights, then the primary
embedding attends only to members of its explicitly supplied group.  A zero-initialized residual
gate makes a fresh fusion block exactly equal to the incumbent primary-only embedding.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .common import _encode_channels, _standardize


RELATED_CHECKPOINT_FORMAT = "mantis-related-v1"


class RelatedSeriesFusion(nn.Module):
    """Fuse ``[primary, related...]`` embeddings with masked group attention.

    Unlike global batch attention, groups are explicit in the ``R`` dimension, so two unrelated
    training examples can never exchange information. ``mask=False`` contexts are excluded from
    keys/values. Role embeddings distinguish exact timeframe and sibling slots.
    """

    def __init__(self, embed_dim: int, *, num_roles: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        dim = int(embed_dim)
        heads = int(num_heads)
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if int(num_roles) < 1:
            raise ValueError("num_roles must be positive")
        self.embed_dim, self.num_roles, self.num_heads = dim, int(num_roles), heads
        self.norm = nn.LayerNorm(dim)
        self.role_embedding = nn.Embedding(self.num_roles, dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=float(dropout), batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, embeddings, mask=None, role_ids=None, *, return_attention=False):
        """Return the fused primary embedding, optionally with per-head attention weights."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, related, dim]")
        batch, related, dim = embeddings.shape
        if dim != self.embed_dim:
            raise ValueError(f"expected embedding dim {self.embed_dim}, got {dim}")
        if mask is None:
            mask = torch.ones(batch, related, dtype=torch.bool, device=embeddings.device)
        else:
            mask = mask.to(device=embeddings.device, dtype=torch.bool)
        if mask.shape != (batch, related):
            raise ValueError("mask must have shape [batch, related]")
        if not mask[:, 0].all():
            raise ValueError("the primary context in slot zero must always be valid")
        if role_ids is None:
            role_ids = torch.arange(related, device=embeddings.device)
        role_ids = role_ids.to(device=embeddings.device, dtype=torch.long)
        if role_ids.ndim == 1:
            if role_ids.shape != (related,):
                raise ValueError("1-D role_ids must have length related")
            roles = self.role_embedding(role_ids)[None, :, :]
        elif role_ids.shape == (batch, related):
            roles = self.role_embedding(role_ids)
        else:
            raise ValueError("role_ids must have shape [related] or [batch, related]")
        tokens = self.norm(embeddings + roles)
        query = tokens[:, :1]
        context, weights = self.attention(
            query, tokens, tokens, key_padding_mask=~mask,
            need_weights=bool(return_attention), average_attn_weights=False,
        )
        fused = embeddings[:, 0] + torch.tanh(self.gate) * self.dropout(self.out_norm(context[:, 0]))
        return (fused, weights) if return_attention else fused


class RelatedMantisEncoder(nn.Module):
    """Shared Mantis encoder followed by the lightweight related-series fusion block."""

    def __init__(self, mantis: nn.Module, *, channels: int = 5, num_roles: int = 6,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mantis = mantis
        self.channels = int(channels)
        hidden = int(getattr(mantis, "hidden_dim", 256))
        self.embed_dim = self.channels * hidden
        self.fusion = RelatedSeriesFusion(
            self.embed_dim, num_roles=num_roles, num_heads=num_heads, dropout=dropout)

    @property
    def seq_len(self):
        return getattr(self.mantis, "seq_len", 512)

    def encode_members(self, windows):
        """Encode standardized ``[B,R,C,L]`` windows with one shared Mantis call."""
        if windows.ndim != 4 or windows.shape[2] != self.channels:
            raise ValueError(f"windows must have shape [B,R,{self.channels},L]")
        batch, related, channels, length = windows.shape
        flat = windows.reshape(batch * related, channels, length)
        embeddings = _encode_channels(self.mantis, flat)
        return embeddings.reshape(batch, related, self.embed_dim)

    def forward(self, windows, mask=None, role_ids=None, *, standardized=False,
                return_attention=False):
        x = windows if standardized else _standardize(windows.flatten(0, 1)).reshape_as(windows)
        members = self.encode_members(x)
        return self.fusion(members, mask, role_ids, return_attention=return_attention)


def related_checkpoint_state(model: RelatedMantisEncoder) -> dict:
    """Materialize a portable composite checkpoint (plain Mantis weights + fusion weights)."""
    from .lora import merged_state_dict
    return {
        "format": RELATED_CHECKPOINT_FORMAT,
        "encoder": merged_state_dict(model.mantis),
        "fusion": {key: value.detach().cpu().clone()
                   for key, value in model.fusion.state_dict().items()},
        "config": {
            "channels": model.channels,
            "num_roles": model.fusion.num_roles,
            "num_heads": model.fusion.num_heads,
            "embed_dim": model.embed_dim,
        },
    }


def is_related_checkpoint(state) -> bool:
    return isinstance(state, dict) and state.get("format") == RELATED_CHECKPOINT_FORMAT


def plain_encoder_state(state):
    """Return the ordinary Mantis state from a plain or related checkpoint."""
    return state["encoder"] if is_related_checkpoint(state) else state


def load_related_checkpoint(model: RelatedMantisEncoder, state) -> None:
    if not is_related_checkpoint(state):
        raise ValueError(f"expected {RELATED_CHECKPOINT_FORMAT} checkpoint")
    cfg = state.get("config", {})
    expected = (model.channels, model.fusion.num_roles, model.fusion.num_heads, model.embed_dim)
    found = (cfg.get("channels"), cfg.get("num_roles"), cfg.get("num_heads"), cfg.get("embed_dim"))
    if found != expected:
        raise ValueError(f"related checkpoint architecture mismatch: expected={expected}, found={found}")
    from .lora import load_plain_state_dict
    load_plain_state_dict(model.mantis, state["encoder"])
    model.fusion.load_state_dict(state["fusion"])


@torch.no_grad()
def embed_related_windows(windows, mask, *, ckpt, role_ids=None,
                          model_id="paris-noah/Mantis-8M", device=None, batch=64):
    """Embed grouped raw OHLCV windows from a ``mantis-related-v1`` checkpoint.

    ``windows`` is ``[N,R,C,L]`` and ``mask`` is ``[N,R]``. This explicit shape keeps the
    downstream contract honest: related contexts cannot be silently flattened into channels or
    confused with the incumbent primary-only encoder input.
    """
    import numpy as np
    from .common import load_mantis

    state = (torch.load(ckpt, map_location="cpu")
             if isinstance(ckpt, (str, bytes, os.PathLike)) else ckpt)
    if not is_related_checkpoint(state):
        raise ValueError(f"embed_related_windows requires a {RELATED_CHECKPOINT_FORMAT} checkpoint")
    cfg = state["config"]
    dev = device or ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    model = RelatedMantisEncoder(
        load_mantis(model_id), channels=int(cfg["channels"]),
        num_roles=int(cfg["num_roles"]), num_heads=int(cfg["num_heads"])).to(dev).eval()
    load_related_checkpoint(model, state)
    values = torch.as_tensor(np.asarray(windows, np.float32))
    masks = torch.as_tensor(np.asarray(mask, bool))
    roles = torch.arange(values.shape[1]) if role_ids is None else torch.as_tensor(role_ids)
    output = []
    for start in range(0, len(values), int(batch)):
        output.append(model(values[start:start + batch].to(dev), masks[start:start + batch].to(dev),
                            roles.to(dev)).float().cpu())
    if not output:
        return np.zeros((0, int(cfg["embed_dim"])), np.float32)
    return torch.cat(output).numpy()


__all__ = ["RELATED_CHECKPOINT_FORMAT", "RelatedSeriesFusion", "RelatedMantisEncoder",
           "related_checkpoint_state", "is_related_checkpoint", "plain_encoder_state",
           "load_related_checkpoint", "embed_related_windows"]
