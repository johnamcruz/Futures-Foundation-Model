"""Optional diagnostics for the frozen Momentum-Volatility task readout.

The promoted foundation contract remains encoder-only.  This module reconstructs the already
trained SSL head only when explicitly requested for debugging or a controlled ablation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..momentum_volatility import COUPLING_CLASSES, MOMENTUM_VOLATILITY_SCHEMA
from .common import _standardize
from .momentum_volatility import MomentumVolatilityNet


DEFAULT_HORIZONS = (5, 10, 20, 25)
READOUT_SCHEMA = "ffm_momentum_volatility_readout_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_mv_readout(path, *, trainer_ckpt, encoder_ckpt,
                      horizons=DEFAULT_HORIZONS):
    """Export task tensors without optimizer state or the transient LoRA encoder."""
    payload = torch.load(trainer_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"MV trainer sidecar has no model_state: {trainer_ckpt}")
    task = {key: value for key, value in state.items()
            if not key.startswith("encoder.")}
    required = ("adapter.transformation.weight", "decoder.2.weight",
                "mv_head.2.weight")
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(f"MV trainer sidecar is missing task tensors: {missing}")
    channels = int(task["adapter.transformation.weight"].shape[1])
    adapted = int(task["adapter.transformation.weight"].shape[0])
    horizons = tuple(int(value) for value in horizons)
    forecast_count = int(task["decoder.2.weight"].shape[0]) // channels
    mv_width = int(task["mv_head.2.weight"].shape[0])
    if forecast_count != len(horizons) or mv_width != 6 * len(horizons):
        raise ValueError("MV readout metadata does not match trainer tensors")
    output = {
        "schema": READOUT_SCHEMA,
        "mv_schema": MOMENTUM_VOLATILITY_SCHEMA,
        "encoder_sha256": _sha256(encoder_ckpt),
        "channels": channels,
        "new_channels": adapted,
        "horizons": list(horizons),
        "coupling_classes": list(COUPLING_CLASSES),
        "model_state": task,
        "usage": "optional_diagnostics_only",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(destination)
    return str(destination)


def load_mv_forecaster(*, encoder_ckpt, readout_ckpt,
                       model_id="paris-noah/Mantis-8M", device="cpu"):
    """Reconstruct a hash-matched frozen encoder and optional diagnostic head."""
    payload = torch.load(readout_ckpt, map_location="cpu", weights_only=False)
    if (payload.get("schema") != READOUT_SCHEMA
            or payload.get("mv_schema") != MOMENTUM_VOLATILITY_SCHEMA):
        raise ValueError(f"unsupported MV readout: {readout_ckpt}")
    if payload.get("encoder_sha256") != _sha256(encoder_ckpt):
        raise ValueError("MV readout belongs to a different encoder checkpoint")
    if tuple(payload.get("coupling_classes", ())) != COUPLING_CLASSES:
        raise ValueError("MV coupling class ordering does not match the runtime")
    net = MomentumVolatilityNet(
        C=int(payload["channels"]), new_channels=int(payload["new_channels"]),
        horizons=tuple(payload["horizons"]), model_id=model_id, aux_dim=0)
    net.encoder.load_state_dict(torch.load(
        encoder_ckpt, map_location="cpu", weights_only=False), strict=True)
    own, task = net.state_dict(), payload["model_state"]
    keys = [key for key in own if not key.startswith("encoder.")]
    missing = [key for key in keys if key not in task]
    shapes = [key for key in keys if key in task and own[key].shape != task[key].shape]
    if missing or shapes:
        raise ValueError(f"MV readout mismatch: missing={missing[:8]} shape={shapes[:8]}")
    with torch.no_grad():
        for key in keys:
            own[key].copy_(task[key])
    net.load_state_dict(own, strict=True)
    return net.to(device).eval()


def mv_readout_feature_names(channels=5, horizons=DEFAULT_HORIZONS):
    horizons = tuple(int(value) for value in horizons)
    names = [
        f"forecast_{field}_h{horizon}"
        for field in ("open", "high", "low", "close", "volume")[:int(channels)]
        for horizon in horizons]
    names += [f"mv_momentum_h{horizon}" for horizon in horizons]
    names += [f"mv_log_range_ratio_h{horizon}" for horizon in horizons]
    names += [
        f"mv_coupling_{state}_h{horizon}_prob"
        for horizon in horizons
        for state in ("chop", "continuation", "reversal", "launch")]
    return names


@torch.no_grad()
def mv_readout_features(windows, *, encoder_ckpt, readout_ckpt,
                        model_id="paris-noah/Mantis-8M", device=None, batch=512):
    """Explicitly opt-in diagnostic outputs from raw causal ``[N,5,L]`` windows."""
    selected = device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    net = load_mv_forecaster(
        encoder_ckpt=encoder_ckpt, readout_ckpt=readout_ckpt,
        model_id=model_id, device=selected)
    values, output = np.asarray(windows, np.float32), []
    for start in range(0, len(values), int(batch)):
        raw = torch.as_tensor(values[start:start + int(batch)], device=selected)
        context = _standardize(raw).clamp(-10.0, 10.0)
        embedding = net.embed(context)
        candles = net.decoder(embedding).view(-1, net.C, net.nH)
        mv = net.mv_head(embedding).view(-1, net.nH, 6)
        output.append(torch.cat((
            candles.flatten(1), mv[:, :, 0], mv[:, :, 1],
            mv[:, :, 2:].softmax(-1).flatten(1)), dim=1).float().cpu().numpy())
    width = len(mv_readout_feature_names(net.C, net.horizons))
    return np.concatenate(output) if output else np.empty((0, width), np.float32)


__all__ = [
    "READOUT_SCHEMA", "export_mv_readout", "load_mv_forecaster",
    "mv_readout_feature_names", "mv_readout_features",
]
