"""Frozen inference and compact readout export for Momentum-Volatility SSL."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..momentum_volatility import COUPLING_CLASSES, MOMENTUM_VOLATILITY_SCHEMA
from .common import _standardize
from .momentum_volatility import MomentumVolatilityNet


DEFAULT_HORIZONS = (5, 10, 20, 25)
READOUT_SCHEMA = "ffm_momentum_volatility_readout_v1"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_mv_readout(path, *, trainer_ckpt, encoder_ckpt,
                      horizons=DEFAULT_HORIZONS) -> str:
    """Export only the trained adapter, candle decoder, and MV head."""
    payload = torch.load(trainer_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"MV trainer sidecar has no model_state: {trainer_ckpt}")
    task_state = {key: value for key, value in state.items()
                  if not key.startswith("encoder.")}
    required = (
        "adapter.transformation.weight",
        "decoder.2.weight",
        "mv_head.2.weight",
    )
    missing = [key for key in required if key not in task_state]
    if missing:
        raise ValueError(f"MV trainer sidecar is missing task tensors: {missing}")
    channels = int(task_state["adapter.transformation.weight"].shape[1])
    new_channels = int(task_state["adapter.transformation.weight"].shape[0])
    horizons = tuple(int(value) for value in horizons)
    inferred_forecast = int(task_state["decoder.2.weight"].shape[0]) // channels
    output_width = int(task_state["mv_head.2.weight"].shape[0])
    output_per_horizon = 2 + len(COUPLING_CLASSES)
    if (inferred_forecast != len(horizons)
            or output_width != len(horizons) * output_per_horizon):
        raise ValueError(
            "MV readout metadata does not match trainer tensors: "
            f"horizons={len(horizons)}/{inferred_forecast}, "
            f"mv_width={output_width}/{len(horizons) * output_per_horizon}")
    output = {
        "schema": READOUT_SCHEMA,
        "mv_schema": MOMENTUM_VOLATILITY_SCHEMA,
        "encoder_sha256": _sha256(encoder_ckpt),
        "channels": channels,
        "new_channels": new_channels,
        "horizons": list(horizons),
        "coupling_classes": list(COUPLING_CLASSES),
        "model_state": task_state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(destination)
    return str(destination)


def load_mv_forecaster(*, encoder_ckpt, readout_ckpt,
                       model_id="paris-noah/Mantis-8M", device="cpu"):
    """Load a hash-matched encoder/readout pair and fail closed on drift."""
    payload = torch.load(readout_ckpt, map_location="cpu", weights_only=False)
    if (payload.get("schema") != READOUT_SCHEMA
            or payload.get("mv_schema") != MOMENTUM_VOLATILITY_SCHEMA):
        raise ValueError(f"unsupported MV readout: {readout_ckpt}")
    if payload.get("encoder_sha256") != _sha256(encoder_ckpt):
        raise ValueError("MV readout belongs to a different encoder checkpoint")
    if tuple(payload.get("coupling_classes", ())) != COUPLING_CLASSES:
        raise ValueError("MV coupling class ordering does not match the runtime")
    horizons = tuple(int(value) for value in payload["horizons"])
    net = MomentumVolatilityNet(
        C=int(payload["channels"]), new_channels=int(payload["new_channels"]),
        horizons=horizons, model_id=model_id, aux_dim=0)
    encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    net.encoder.load_state_dict(encoder_state, strict=True)
    own = net.state_dict()
    task_state = payload["model_state"]
    task_keys = [key for key in own if not key.startswith("encoder.")]
    missing = [key for key in task_keys if key not in task_state]
    shapes = [key for key in task_keys
              if key in task_state and own[key].shape != task_state[key].shape]
    if missing or shapes:
        raise ValueError(f"MV readout mismatch: missing={missing[:8]} shape={shapes[:8]}")
    with torch.no_grad():
        for key in task_keys:
            own[key].copy_(task_state[key])
    net.load_state_dict(own, strict=True)
    return net.to(device).eval()


def mv_readout_feature_names(channels=5, horizons=DEFAULT_HORIZONS):
    """Ordered deployable task outputs, excluding the intermediate embedding."""
    horizons = tuple(int(value) for value in horizons)
    names = [
        f"forecast_{field}_h{horizon}"
        for field in ("open", "high", "low", "close", "volume")[:int(channels)]
        for horizon in horizons
    ]
    names += [f"mv_momentum_h{horizon}" for horizon in horizons]
    names += [f"mv_log_range_ratio_h{horizon}" for horizon in horizons]
    names += [
        f"mv_coupling_{state}_h{horizon}_prob"
        for horizon in horizons
        for state in ("chop", "continuation", "reversal", "launch")
    ]
    return names


def mv_feature_names(hidden=256, adapted_channels=3, channels=5,
                     horizons=DEFAULT_HORIZONS):
    names = [f"mv_emb_{index}" for index in range(
        int(hidden) * int(adapted_channels))]
    names += mv_readout_feature_names(channels=channels, horizons=horizons)
    return names


class MomentumVolatilityFeatureEncoder(nn.Module):
    """Raw OHLCV windows to embedding plus frozen MV task readouts."""

    def __init__(self, net, *, include_embedding=True):
        super().__init__()
        self.net = net
        self.include_embedding = bool(include_embedding)

    def forward(self, windows):
        context = _standardize(windows).clamp(-10.0, 10.0)
        embedding = self.net.embed(context)
        candles = self.net.decoder(embedding).view(-1, self.net.C, self.net.nH)
        raw = self.net.mv_head(embedding).view(-1, self.net.nH, 6)
        values = (
            candles.flatten(1),
            raw[:, :, 0],
            raw[:, :, 1],
            raw[:, :, 2:].softmax(-1).flatten(1),
        )
        if self.include_embedding:
            values = (embedding,) + values
        return torch.cat(values, dim=1)


@torch.no_grad()
def mv_features(windows, *, encoder_ckpt, readout_ckpt,
                model_id="paris-noah/Mantis-8M", device=None, batch=512,
                include_embedding=True):
    """Extract causal MV features from raw ``[N,5,L]`` windows."""
    selected = device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    net = load_mv_forecaster(
        encoder_ckpt=encoder_ckpt, readout_ckpt=readout_ckpt,
        model_id=model_id, device=selected)
    module = MomentumVolatilityFeatureEncoder(
        net, include_embedding=include_embedding).to(selected).eval()
    values = np.asarray(windows, np.float32)
    output = []
    for start in range(0, len(values), int(batch)):
        tensor = torch.as_tensor(
            values[start:start + int(batch)], device=selected)
        output.append(module(tensor).float().cpu().numpy())
    width = len(
        mv_feature_names(adapted_channels=net.new_c, channels=net.C,
                         horizons=net.horizons)
        if include_embedding else
        mv_readout_feature_names(channels=net.C, horizons=net.horizons))
    return np.concatenate(output) if output else np.empty((0, width), np.float32)


__all__ = [
    "DEFAULT_HORIZONS", "READOUT_SCHEMA", "MomentumVolatilityFeatureEncoder",
    "export_mv_readout", "load_mv_forecaster", "mv_feature_names",
    "mv_features", "mv_readout_feature_names",
]
