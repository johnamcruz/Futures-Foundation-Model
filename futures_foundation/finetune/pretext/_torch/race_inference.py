"""Frozen inference and compact readout export for NextLeg race v2."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..nextleg_race import RACE_LEVELS, RACE_SCHEMA
from .common import _standardize
from .nextleg_race import NextLegRaceNet


DEFAULT_HORIZONS = (5, 10, 20, 25)
READOUT_SCHEMA = "ffm_nextleg_race_readout_v2"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_race_readout(path, *, trainer_ckpt, encoder_ckpt,
                        horizons=DEFAULT_HORIZONS, race_levels=RACE_LEVELS) -> str:
    """Write the selected adapter/forecast/race heads without optimizer or transient LoRA state."""
    payload = torch.load(trainer_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"race trainer sidecar has no model_state: {trainer_ckpt}")
    task_state = {key: value for key, value in state.items() if not key.startswith("encoder.")}
    required = ("adapter.transformation.weight", "decoder.2.weight", "leg_head.2.weight",
                "race_head.2.weight")
    missing = [key for key in required if key not in task_state]
    if missing:
        raise ValueError(f"race trainer sidecar is missing task tensors: {missing}")
    channels = int(task_state["adapter.transformation.weight"].shape[1])
    new_channels = int(task_state["adapter.transformation.weight"].shape[0])
    horizons = tuple(int(value) for value in horizons)
    race_levels = tuple(float(value) for value in race_levels)
    inferred_horizons = int(task_state["decoder.2.weight"].shape[0]) // channels
    inferred_levels = int(task_state["race_head.2.weight"].shape[0]) // 3
    if inferred_horizons != len(horizons) or inferred_levels != len(race_levels):
        raise ValueError(
            "race readout metadata does not match trainer tensors: "
            f"horizons={len(horizons)}/{inferred_horizons}, "
            f"levels={len(race_levels)}/{inferred_levels}")
    output = {
        "schema": READOUT_SCHEMA,
        "race_schema": RACE_SCHEMA,
        "encoder_sha256": _sha256(encoder_ckpt),
        "channels": channels,
        "new_channels": new_channels,
        "horizons": list(horizons),
        "race_levels": list(race_levels),
        "model_state": task_state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(destination)
    return str(destination)


def load_race_forecaster(*, encoder_ckpt, readout_ckpt,
                         model_id="paris-noah/Mantis-8M", device="cpu"):
    """Load one hash-matched encoder/readout pair and fail closed on architecture drift."""
    payload = torch.load(readout_ckpt, map_location="cpu", weights_only=False)
    if payload.get("schema") != READOUT_SCHEMA or payload.get("race_schema") != RACE_SCHEMA:
        raise ValueError(f"unsupported NextLeg race readout: {readout_ckpt}")
    if payload.get("encoder_sha256") != _sha256(encoder_ckpt):
        raise ValueError("NextLeg race readout belongs to a different encoder checkpoint")
    horizons = tuple(int(value) for value in payload["horizons"])
    levels = tuple(float(value) for value in payload["race_levels"])
    net = NextLegRaceNet(
        C=int(payload["channels"]), new_channels=int(payload["new_channels"]),
        horizons=horizons, race_levels=levels, model_id=model_id, aux_dim=0)
    encoder_state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    net.encoder.load_state_dict(encoder_state, strict=True)
    own = net.state_dict()
    task_state = payload["model_state"]
    task_keys = [key for key in own if not key.startswith("encoder.")]
    missing = [key for key in task_keys if key not in task_state]
    shapes = [key for key in task_keys
              if key in task_state and own[key].shape != task_state[key].shape]
    if missing or shapes:
        raise ValueError(f"NextLeg race readout mismatch: missing={missing[:8]} shape={shapes[:8]}")
    with torch.no_grad():
        for key in task_keys:
            own[key].copy_(task_state[key])
    net.load_state_dict(own, strict=True)
    return net.to(device).eval()


def race_feature_names(hidden=256, adapted_channels=3, channels=5,
                       horizons=DEFAULT_HORIZONS, levels=RACE_LEVELS):
    names = [f"race_emb_{index}" for index in range(int(hidden) * int(adapted_channels))]
    names += race_readout_feature_names(
        channels=channels, horizons=horizons, levels=levels)
    return names


def race_readout_feature_names(channels=5, horizons=DEFAULT_HORIZONS,
                               levels=RACE_LEVELS):
    """Ordered deployable task outputs, excluding the intermediate embedding."""
    names = [
        f"forecast_{field}_h{horizon}"
        for field in ("open", "high", "low", "close", "volume")[:int(channels)]
        for horizon in horizons
    ]
    names += ["leg1_logbars", "leg2_logbars"]
    names += [f"reach_{level:g}bar_range_prob" for level in levels]
    names += [f"adverse_before_{level:g}bar_range" for level in levels]
    names += [f"delay_to_{level:g}bar_range_logbars" for level in levels]
    return names


class NextLegRaceFeatureEncoder(nn.Module):
    """Raw causal OHLCV windows -> encoder embedding plus calibrated generic race readouts."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, windows):
        context = _standardize(windows).clamp(-10.0, 10.0)
        embedding = self.net.embed(context)
        candles, durations, reach, adverse, delay = self.net.readouts(embedding)
        return torch.cat((
            embedding, candles.flatten(1), durations, reach.sigmoid(), adverse, delay,
        ), dim=1)


class NextLegRaceReadoutEncoder(nn.Module):
    """Raw causal OHLCV windows -> compact candle, leg, and path-race forecasts."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, windows):
        context = _standardize(windows).clamp(-10.0, 10.0)
        embedding = self.net.embed(context)
        candles, durations, reach, adverse, delay = self.net.readouts(embedding)
        return torch.cat((
            candles.flatten(1), durations, reach.sigmoid(), adverse, delay,
        ), dim=1)


@torch.no_grad()
def race_features(windows, *, encoder_ckpt, readout_ckpt,
                  model_id="paris-noah/Mantis-8M", device=None, batch=512):
    """Extract causal race features from raw ``[N,5,L]`` windows."""
    selected = device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    net = load_race_forecaster(
        encoder_ckpt=encoder_ckpt, readout_ckpt=readout_ckpt,
        model_id=model_id, device=selected)
    module = NextLegRaceFeatureEncoder(net).to(selected).eval()
    values = np.asarray(windows, np.float32)
    output = []
    for start in range(0, len(values), int(batch)):
        tensor = torch.as_tensor(values[start:start + int(batch)], device=selected)
        output.append(module(tensor).float().cpu().numpy())
    width = len(race_feature_names(
        adapted_channels=net.new_c, channels=net.C,
        horizons=net.horizons, levels=net.race_levels))
    return np.concatenate(output) if output else np.empty((0, width), np.float32)


@torch.no_grad()
def race_readout_features(windows, *, encoder_ckpt, readout_ckpt,
                          model_id="paris-noah/Mantis-8M", device=None, batch=512):
    """Extract only the compact causal task readouts from raw ``[N,5,L]`` windows."""
    selected = device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    net = load_race_forecaster(
        encoder_ckpt=encoder_ckpt, readout_ckpt=readout_ckpt,
        model_id=model_id, device=selected)
    module = NextLegRaceReadoutEncoder(net).to(selected).eval()
    values = np.asarray(windows, np.float32)
    output = []
    for start in range(0, len(values), int(batch)):
        tensor = torch.as_tensor(values[start:start + int(batch)], device=selected)
        output.append(module(tensor).float().cpu().numpy())
    width = len(race_readout_feature_names(
        channels=net.C, horizons=net.horizons, levels=net.race_levels))
    return np.concatenate(output) if output else np.empty((0, width), np.float32)


__all__ = [
    "DEFAULT_HORIZONS", "READOUT_SCHEMA", "NextLegRaceFeatureEncoder",
    "NextLegRaceReadoutEncoder", "export_race_readout", "load_race_forecaster",
    "race_feature_names", "race_features", "race_readout_feature_names",
    "race_readout_features",
]
