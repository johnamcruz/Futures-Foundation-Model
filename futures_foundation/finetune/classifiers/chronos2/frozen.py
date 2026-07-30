"""Frozen Chronos-2 representation probe behind the generic Classifier seam.

This follows the Mantis frozen pattern:

``labeler.mv_contexts -> isolated encoder worker -> cached embeddings -> shallow head``

The head implementation, calibration, train-stat standardization, and optional
handcrafted-feature concatenation are inherited from ``MantisFrozenClassifier``;
only the backbone embedding/cache identity differs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from ...classifier import register_classifier
from ..mantis.frozen import MantisFrozenClassifier


MODEL_ID = "autogluon/chronos-2-small"
_TREE_FP_CACHE: dict[tuple[str, int], str] = {}


def _checkpoint_fingerprint(checkpoint: str | None, model_id: str) -> str:
    if not checkpoint:
        return f"remote:{model_id}"
    path = Path(checkpoint)
    if not path.exists():
        return f"remote:{checkpoint}"
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file()) \
        if path.is_dir() else [path]
    total_size = sum(candidate.stat().st_size for candidate in files)
    key = (str(path.resolve()), total_size)
    cached = _TREE_FP_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(path) if path.is_dir() else path.name).encode())
        with candidate.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    value = f"local:{digest.hexdigest()}:{total_size}"
    _TREE_FP_CACHE[key] = value
    return value


def _cache_path(cfg: dict, labeler, keys) -> Path | None:
    if os.environ.get("EMBED_CACHE", "1") != "1" or not keys:
        return None
    model_id = str(cfg.get("model_id", MODEL_ID))
    checkpoint = cfg.get("backbone_ckpt")
    sid = str(keys[0][0]) if isinstance(keys[0], (tuple, list)) else "unkeyed"
    seq = int(getattr(labeler, "MV_SEQ", 0))
    mode = str(getattr(labeler, "MV_MODE", "?"))
    try:
        ticker, timeframe = sid.split("@", 1)
        bars = labeler._b[(ticker, timeframe)]
        source_identity = bars.get("source_sha256") or len(bars["c"])
    except Exception:
        source_identity = "unknown"
    identity = {
        "schema": "chronos2-embed-v1",
        "model": _checkpoint_fingerprint(checkpoint, model_id),
        "model_id": model_id,
        "sid": sid,
        "source": source_identity,
        "seq": seq,
        "mode": mode,
        "pool": cfg.get("pool", "reg"),
        "device": str(cfg.get("device", "cpu")).lower(),
        "keys": [list(key) if isinstance(key, (tuple, list)) else key for key in keys],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:20]
    cache_dir = Path(os.environ.get("EMBED_CACHE_DIR", "temp/embed_cache"))
    return cache_dir / f"chronos2_{sid.replace('@', '_')}_{digest}.npy"


@register_classifier("chronos2")
@register_classifier("chronos2_frozen")
class Chronos2FrozenClassifier(MantisFrozenClassifier):
    """Chronos-2 encoder frozen; logistic/MLP downstream head trained per fold."""

    needs_standardize = True
    embed_once = True

    def __init__(self, **cfg):
        cfg = dict(cfg)
        cfg.setdefault("model_id", MODEL_ID)
        cfg.setdefault("device", "mps")
        cfg.setdefault("batch", 5)
        cfg.setdefault("pool", "reg")
        cfg.setdefault("with_features", True)
        super().__init__(**cfg)

    def _embed(self, labeler, keys):
        windows = np.asarray(labeler.mv_contexts(keys), np.float32)
        if windows.ndim != 3:
            raise ValueError(
                f"Chronos-2 classifier expects mv_contexts [N,C,T], got {windows.shape}")
        if windows.shape[1] != 5:
            raise ValueError(
                "Chronos-2 serving expects exactly one ticker with five channels "
                "ordered open, high, low, close, volume; "
                f"received {windows.shape[1]} channels")
        if not np.isfinite(windows).all():
            raise ValueError("Chronos-2 classifier received non-finite OHLCV windows")
        worker_cfg = {
            "ckpt": self.cfg.get("backbone_ckpt"),
            "model_id": self.cfg["model_id"],
            "device": self.cfg["device"],
            "batch": int(self.cfg["batch"]),
            "pool": self.cfg["pool"],
            "context_length": int(self.cfg.get("context_length") or windows.shape[-1]),
        }
        command = [
            sys.executable,
            "-u",
            "-m",
            "futures_foundation.finetune.classifiers.chronos2._embed_worker",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "windows.npy", windows)
            worker_cfg["_windows"] = str(directory / "windows.npy")
            (directory / "cfg.json").write_text(json.dumps(worker_cfg))
            result = subprocess.run(
                command + [str(directory)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(
                    "Chronos-2 embed worker failed:\n"
                    + (result.stdout + "\n" + result.stderr)[-3000:])
            embeddings = np.load(directory / "emb.npy")
        if len(embeddings) != len(keys) or not np.isfinite(embeddings).all():
            raise RuntimeError("Chronos-2 embedding output violates shape/finite contract")
        return embeddings

    def featurize(self, labeler, keys):
        path = _cache_path(self.cfg, labeler, keys)
        if path is not None and path.is_file():
            cached = np.load(path)
            if len(cached) == len(keys) and np.isfinite(cached).all():
                print(f"[embed-cache] HIT {path.name} ({len(cached)})", flush=True)
                return self._with_features(labeler, keys, cached)
        embeddings = self._embed(labeler, keys)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
            np.save(temporary, embeddings)
            os.replace(temporary, path)
            print(f"[embed-cache] WROTE {path.name} ({len(embeddings)})", flush=True)
        return self._with_features(labeler, keys, embeddings)

    def fit_predict(self, *args, **kwargs):
        if self.cfg.get("export_onnx_path"):
            raise NotImplementedError(
                "Chronos-2 encoder ONNX export is not yet part of the smoke contract; "
                "run walk-forward without export and add deployment export only after promotion")
        return super().fit_predict(*args, **kwargs)


__all__ = ["Chronos2FrozenClassifier", "MODEL_ID"]
