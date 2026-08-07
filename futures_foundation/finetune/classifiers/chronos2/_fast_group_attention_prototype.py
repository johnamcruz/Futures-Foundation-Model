"""PROTOTYPE: split Chronos-2 group attention into independent groups.

This is an opt-in port of the approach proposed in amazon-science/
chronos-forecasting#442.  It exists only to measure numerical parity and MPS
throughput for FFM's fixed five-variate OHLCV groups.  It is not enabled by
default and must not be treated as a promoted embedding implementation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _FastGroupMask:
    value: object
    groups: int
    group_size: int
    time_steps: int


_ENABLED = False


def enable_fast_group_attention(*, group_size: int = 5) -> None:
    """Patch the loaded Chronos-2 classes for uniform contiguous groups."""
    global _ENABLED
    if _ENABLED:
        return
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("fast group-attention size must be a positive integer")

    import torch
    from einops import rearrange
    from chronos.chronos2.layers import AttentionOutput, GroupSelfAttention
    from chronos.chronos2.model import Chronos2Encoder

    original_construct = Chronos2Encoder._construct_and_invert_group_time_mask
    original_forward = GroupSelfAttention.forward

    def construct(group_ids, attention_mask, floating_type):
        batch_size, time_steps = attention_mask.shape
        if batch_size % group_size:
            return original_construct(group_ids, attention_mask, floating_type)
        groups = batch_size // group_size
        valid_keys = rearrange(
            attention_mask,
            "(group series) time -> time group series",
            group=groups,
            series=group_size,
        )
        valid = valid_keys[:, :, None, :].expand(
            time_steps, groups, group_size, group_size)
        if torch.is_floating_point(valid):
            floating_type = valid.dtype
        mask = (1.0 - valid) * torch.finfo(floating_type).min
        mask = rearrange(
            mask,
            "time group query key -> (time group) 1 query key",
        )
        return _FastGroupMask(
            value=mask,
            groups=groups,
            group_size=group_size,
            time_steps=time_steps,
        )

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        if not isinstance(attention_mask, _FastGroupMask):
            return original_forward(
                self,
                hidden_states,
                attention_mask,
                output_attentions,
            )
        if output_attentions:
            raise NotImplementedError(
                "fast group-attention prototype does not materialize dense "
                "attention weights"
            )
        spec = attention_mask
        transposed = rearrange(hidden_states, "batch time d -> time batch d")
        normalized = self.layer_norm(transposed)
        grouped = rearrange(
            normalized,
            "time (group series) d -> (time group) series d",
            group=spec.groups,
            series=spec.group_size,
        )
        attended = self.self_attention(
            grouped,
            mask=spec.value,
            output_attentions=False,
        )[0]
        attended = rearrange(
            attended,
            "(time group) series d -> time (group series) d",
            time=spec.time_steps,
            group=spec.groups,
        )
        output = transposed + self.dropout(attended)
        output = rearrange(output, "time batch d -> batch time d")
        return AttentionOutput(hidden_states=output, attn_weights=None)

    Chronos2Encoder._construct_and_invert_group_time_mask = staticmethod(construct)
    GroupSelfAttention.forward = forward
    _ENABLED = True


__all__ = ["enable_fast_group_attention"]
