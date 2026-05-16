"""
Shared training-resume primitives — used by both pretrain and finetune.

A long GPU job (Colab/Spot) can disconnect mid-run. These helpers persist and
restore *full training state* so the job continues from the next epoch with the
optimiser / LR-schedule / counters intact instead of restarting from 0.

Design seam: this module owns the *mechanics* (data-inclusive identity hash,
atomic write, validated load, cleanup). The *state contents* stay
caller-defined — pretrain and finetune each build their own blob dict — so the
shared surface is small and generic (no trainer-specific knowledge here).

The identity hash deliberately includes a **data** term. Excluding it is the
root cause of the CRT stale-cache incident: a resume/skip key that ignores the
data silently continues (or replays) a run against a different/regenerated
parquet. Any consumer MUST pass a data identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

__all__ = ["resume_hash", "atomic_save_resume", "load_resume", "clear_resume"]


def resume_hash(cfg_fields: Dict[str, Any],
                arch: Dict[str, Any],
                data: Any) -> str:
    """Stable short identity for a training run.

    Args:
        cfg_fields: trajectory-determining hyperparameters (lr, epochs, seq_len,
            stride, warmup, patience, …). Exclude things that don't change the
            optimisation path (num_workers, device, file paths).
        arch: model architecture dims (hidden/layers/heads/features).
        data: data identity — e.g. sorted (instrument, bar_count) pairs, or a
            dataset/parquet fingerprint. NON-OPTIONAL by contract: omitting it
            reintroduces the CRT stale-cache class of bug.

    Returns:
        10-char hex digest.
    """
    payload = {
        "cfg": {k: cfg_fields[k] for k in sorted(cfg_fields)},
        "arch": {k: arch[k] for k in sorted(arch)},
        "data": data,
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def atomic_save_resume(path: Path, blob: Dict[str, Any]) -> None:
    """Persist `blob` to `path` atomically.

    Writes a sibling .tmp then os.replace()s it in — a torn write (disconnect
    mid-save, the very failure we're guarding against) can never leave a
    corrupt resume file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_resume(path: Path,
                expected_hash: str,
                map_location: Any = "cpu",
                hash_key: str = "cfg_hash") -> Optional[Dict[str, Any]]:
    """Load a resume blob iff it exists and its embedded hash matches.

    Returns the blob dict on a valid match, else None (caller starts fresh).
    A hash mismatch means config/arch/data changed — resuming would corrupt
    the run, so it is intentionally ignored rather than loaded.

    `hash_key` is the dict key the caller stored its hash under. pretrain uses
    the default "cfg_hash"; finetune stores "config_hash" — making this
    configurable lets both reuse this loader without either changing its
    on-disk checkpoint schema (no checkpoint-cache invalidation).
    """
    path = Path(path)
    if not path.exists():
        return None
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if blob.get(hash_key) != expected_hash:
        return None
    return blob


def clear_resume(path: Path) -> None:
    """Remove the resume file (call on natural completion so a later run with
    the same identity starts fresh rather than thinking it's mid-flight)."""
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
