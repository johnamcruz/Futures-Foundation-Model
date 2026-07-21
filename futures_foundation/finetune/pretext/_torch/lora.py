"""Small, deployment-safe LoRA implementation for the Mantis encoder.

Amazon Chronos-2 adapts q/k/v/o attention projections with rank 8 and alpha 16.
Mantis fuses q/k/v into ``to_qkv`` and uses ``to_out.0`` for attention output, so
those are the equivalent targets. Training uses adapters, while snapshots are
materialized as ordinary Mantis state dictionaries; downstream loaders, ONNX
export, Pivot Trend, and AlgoTrader never need to know LoRA was used.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank delta, W + alpha/r * B@A."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False,
                               device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False,
                               device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # exact base-model parity at injection
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        delta = self.lora_B.weight @ self.lora_A.weight
        return self.base.weight + delta.to(self.base.weight.dtype) * self.scaling


def _set_child(parent: nn.Module, name: str, value: nn.Module) -> None:
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def inject_mantis_lora(encoder: nn.Module, *, rank: int = 8, alpha: float = 16.0,
                       dropout: float = 0.0) -> dict[str, int | float]:
    """Freeze Mantis and replace its six fused qkv + six output projections."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    targets = [(name, module) for name, module in encoder.named_modules()
               if isinstance(module, nn.Linear)
               and (name.endswith(".fn.to_qkv") or name.endswith(".fn.to_out.0"))]
    if len(targets) != 12:
        raise RuntimeError(f"expected 12 Mantis attention projections, found {len(targets)}: "
                           f"{[name for name, _ in targets]}")
    modules = dict(encoder.named_modules())
    for name, module in targets:
        parent_name, child_name = name.rsplit(".", 1)
        _set_child(modules[parent_name], child_name,
                   LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    return {"modules": len(targets), "trainable": trainable, "total": total,
            "percent": 100.0 * trainable / total}


def has_lora(encoder: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in encoder.modules())


def merged_state_dict(encoder: nn.Module) -> dict[str, torch.Tensor]:
    """Return a plain-Mantis state dict without adapter-specific keys."""
    state = encoder.state_dict()
    wrappers = {name: module for name, module in encoder.named_modules()
                if isinstance(module, LoRALinear)}
    if not wrappers:
        return {key: value.detach().cpu().clone() for key, value in state.items()}
    result: dict[str, torch.Tensor] = {}
    wrapper_prefixes = tuple(name + "." for name in wrappers)
    for key, value in state.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            continue
        if key.endswith(".base.weight"):
            name = key[:-len(".base.weight")]
            result[name + ".weight"] = wrappers[name].merged_weight().detach().cpu().clone()
        elif key.endswith(".base.bias"):
            name = key[:-len(".base.bias")]
            result[name + ".bias"] = value.detach().cpu().clone()
        elif not key.startswith(wrapper_prefixes):
            result[key] = value.detach().cpu().clone()
    return result


def load_plain_state_dict(encoder: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a merged/plain checkpoint into an injected encoder for crash resume."""
    wrappers = {name: module for name, module in encoder.named_modules()
                if isinstance(module, LoRALinear)}
    if not wrappers:
        encoder.load_state_dict(state)
        return
    mapped = dict(state)
    for name, module in wrappers.items():
        mapped[name + ".base.weight"] = mapped.pop(name + ".weight")
        if module.base.bias is not None:
            mapped[name + ".base.bias"] = mapped.pop(name + ".bias")
        nn.init.zeros_(module.lora_B.weight)
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    allowed = {name + suffix for name in wrappers
               for suffix in (".lora_A.weight", ".lora_B.weight")}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch: missing={missing}, unexpected={unexpected}")
