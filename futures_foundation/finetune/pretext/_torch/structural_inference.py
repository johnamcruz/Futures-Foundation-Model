"""Frozen Structural NextLeg feature inference.

The public ``mantis_ssl_structural_nextleg.pt`` artifact intentionally contains only the
transferable Mantis encoder.  The SSL trainer sidecar retains the learned channel adapter and
forecast readouts (HH/HL/LH/LL, BOS/CHOCH, leg duration, excursion, and candle forecasts).  A
downstream strategy that wants those forecasts must load both artifacts; using only the encoder
silently discards most of the structural objective.

This module turns that matched pair into one causal feature encoder.  Inputs are raw OHLCV windows
ending on the decision candle.  Per-window normalization is identical to Structural NextLeg
training.  Future labels are never read at inference time.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _standardize
from .nextleg_structural import StructuralNextLegNet


DEFAULT_HORIZONS = (5, 10, 20, 25)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Structural NextLeg sidecar has no model_state: {path}")
    return payload, state


def load_structural_forecaster(*, encoder_ckpt, trainer_ckpt,
                               model_id="paris-noah/Mantis-8M", device="cpu"):
    """Reconstruct the frozen task network from an encoder plus its matched trainer sidecar.

    The sidecar's encoder is stored in its transient LoRA form.  Production instead loads the
    already-merged public encoder checkpoint and copies only the trained task tensors.  Exact key
    and shape checks fail closed if an unrelated or partial sidecar is supplied.
    """
    payload, state = _task_payload(trainer_ckpt)
    matched = payload.get("matched_encoder_sha256")
    if matched is not None and matched != _sha256(encoder_ckpt):
        raise ValueError(
            "Structural NextLeg sidecar is bound to a different encoder: "
            f"expected={matched} actual={_sha256(encoder_ckpt)}")
    adapter = state.get("adapter.transformation.weight")
    decoder = state.get("decoder.2.weight")
    span = state.get("span_decoder.2.weight")
    if adapter is None or decoder is None or span is None:
        raise ValueError("Structural NextLeg sidecar is missing architecture-defining tensors")
    channels = int(adapter.shape[1])
    new_channels = int(adapter.shape[0])
    horizons = int(decoder.shape[0]) // channels
    span_width = int(span.shape[0]) // channels
    if horizons != len(DEFAULT_HORIZONS):
        raise ValueError(f"unsupported Structural NextLeg horizon count: {horizons}")
    net = StructuralNextLegNet(
        C=channels, new_channels=new_channels, horizons=DEFAULT_HORIZONS,
        model_id=model_id, aux_dim=0, span_width=span_width)
    encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    net.encoder.load_state_dict(encoder_state, strict=True)
    own = net.state_dict()
    task_keys = [key for key in own if not key.startswith("encoder.")]
    missing = [key for key in task_keys if key not in state]
    unexpected_shapes = [
        key for key in task_keys if key in state and own[key].shape != state[key].shape]
    if missing or unexpected_shapes:
        raise ValueError(
            "Structural NextLeg sidecar mismatch: "
            f"missing={missing[:8]} shape={unexpected_shapes[:8]}")
    with torch.no_grad():
        for key in task_keys:
            own[key].copy_(state[key])
    net.load_state_dict(own, strict=True)
    return net.to(device).eval()


def structural_feature_names(hidden=256, channels=5, adapted_channels=3,
                             horizons=DEFAULT_HORIZONS):
    """Stable column contract for :class:`StructuralFeatureEncoder`."""
    names = [f"struct_emb_{index}" for index in range(hidden * adapted_channels)]
    names += [
        f"forecast_{field}_h{horizon}"
        for field in ("open", "high", "low", "close", "volume")[:channels]
        for horizon in horizons]
    names += ["leg1_logbars", "leg2_logbars"]
    names += [f"{when}_{state}_logit" for when in ("current", "next") for state in (
        "uptrend_hh_hl", "downtrend_lh_ll", "expanding_hh_ll", "contracting_lh_hl")]
    names += ["leg1_log_bps", "leg2_log_bps"]
    names += [f"break_{event}_logit" for event in (
        "none", "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch")]
    names += ["break_delay_logbars"]
    return names


class StructuralFeatureEncoder(nn.Module):
    """Raw OHLCV window -> structural embedding plus frozen forecast readouts."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, windows):
        context = _standardize(windows).clamp(-10.0, 10.0)
        embedding = self.net.embed(context)
        candles = self.net.decoder(embedding)
        durations = self.net.leg_head(embedding)
        structures = self.net.structure_head(embedding)
        excursions = self.net.excursion_head(embedding)
        breaks = self.net.break_head(embedding)
        delay = self.net.break_delay_head(embedding)
        return torch.cat(
            (embedding, candles, durations, structures, excursions, breaks, delay), dim=1)


@torch.no_grad()
def structural_features(windows, *, encoder_ckpt, trainer_ckpt,
                        model_id="paris-noah/Mantis-8M", device=None, batch=512):
    """Extract causal Structural NextLeg features from ``[N,5,L]`` raw windows."""
    dev = device or ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    net = load_structural_forecaster(
        encoder_ckpt=encoder_ckpt, trainer_ckpt=trainer_ckpt,
        model_id=model_id, device=dev)
    encoder = StructuralFeatureEncoder(net).to(dev).eval()
    values = np.asarray(windows, np.float32)
    out = []
    for start in range(0, len(values), int(batch)):
        tensor = torch.as_tensor(values[start:start + int(batch)], device=dev)
        out.append(encoder(tensor).float().cpu().numpy())
    width = len(structural_feature_names())
    return np.concatenate(out) if out else np.empty((0, width), np.float32)


def export_structural_encoder_onnx(path, *, encoder_ckpt, trainer_ckpt, C=5, seq=128,
                                   model_id="paris-noah/Mantis-8M", device="cpu"):
    """Export the exact raw-window Structural feature graph used by downstream heads."""
    from .common import _ort_optimize_graph

    net = load_structural_forecaster(
        encoder_ckpt=encoder_ckpt, trainer_ckpt=trainer_ckpt,
        model_id=model_id, device=device)
    module = StructuralFeatureEncoder(net).to(device).eval()
    dummy = torch.randn(2, int(C), int(seq), device=device)
    original_diff = torch.diff

    def traceable_diff(x, n=1, dim=-1, *, axis=None, prepend=None, append=None):
        selected = axis if axis is not None else dim
        for _ in range(int(n)):
            x = (x.narrow(selected, 1, x.size(selected) - 1)
                 - x.narrow(selected, 0, x.size(selected) - 1))
        return x

    torch.diff = traceable_diff
    try:
        torch.onnx.export(
            module, dummy, path, input_names=["ohlcv"], output_names=["embedding"],
            dynamic_axes={"ohlcv": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=15, dynamo=False)
    finally:
        torch.diff = original_diff
    _ort_optimize_graph(path)
    return path


__all__ = [
    "DEFAULT_HORIZONS", "StructuralFeatureEncoder", "export_structural_encoder_onnx",
    "load_structural_forecaster", "structural_feature_names", "structural_features",
]
