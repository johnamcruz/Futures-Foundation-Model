#!/usr/bin/env python3
"""Run the public 9x4 Probe Atlas with a frozen Chronos-2 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ATLAS_SCRIPT = ROOT / "scripts" / "probe_atlas.py"
DEFAULT_RUN = ROOT / "temp" / "chronos2_small_36stream"
DEFAULT_CHECKPOINT = DEFAULT_RUN / "contrastive_kaufman_full" / "checkpoint"
DEFAULT_ATLAS_DIR = DEFAULT_RUN / "probe_atlas"
DEFAULT_LABELS = DEFAULT_ATLAS_DIR / "trend_lifecycle_labels_pre2026.npz"


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        if not path.is_dir():
            raise SystemExit(
                "Chronos-2 Probe Atlas expects a PEFT checkpoint directory")
        if not (path / "adapter_config.json").is_file():
            raise SystemExit(
                f"Chronos-2 adapter_config.json is missing: {path}")
        return _tree_sha256(path)
    if path.is_absolute() or value.startswith((".", "~")) or value.count("/") != 1:
        raise SystemExit(f"Chronos-2 checkpoint does not exist: {path}")
    return f"remote:{value}"


def _generate_labels(labels: Path, data_dir: Path) -> None:
    generator = ROOT / "scripts" / "generate_trend_labels.py"
    labels.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "FFM_ROOT": str(ROOT),
        "DATA_DIR": str(data_dir),
        "TREND_LABELS": str(labels),
        "TREND_LABEL_END": "2026-01-01",
    })
    print(
        f"[chronos2-atlas] generating shared pre-2026 lifecycle corpus -> {labels}",
        flush=True,
    )
    subprocess.run(
        [sys.executable, str(generator)],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    if not labels.is_file():
        raise RuntimeError("Probe Atlas label generator produced no artifact")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    value.add_argument("--name", default="chronos2_kaufman")
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    value.add_argument("--out-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    value.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    value.add_argument("--control", choices=("real", "shuffle", "random"), default="real")
    value.add_argument("--window", type=int, default=256)
    value.add_argument("--horizons", default="5,10,20,50")
    value.add_argument(
        "--pool", choices=("reg", "mean_context"), default="reg")
    value.add_argument(
        "--batch-series", type=int, default=320,
        help=(
            "Chronos batch size counts variates; 320 equals 64 five-channel "
            "OHLCV windows and is the verified M1 Atlas default"))
    value.add_argument("--chunk-windows", type=int, default=1024)
    value.add_argument("--train-per-stream", type=int, default=6000)
    value.add_argument("--eval-per-stream", type=int, default=3000)
    value.add_argument("--preflight-only", action="store_true")
    return value


def _load_atlas():
    specification = importlib.util.spec_from_file_location(
        "chronos2_public_probe_atlas", ATLAS_SCRIPT)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> dict:
    args = parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    checkpoint_hash = _checkpoint_identity(args.checkpoint)
    horizons = tuple(
        int(item) for item in args.horizons.split(",") if item.strip())
    if (
        not horizons
        or any(item <= 0 for item in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise SystemExit("--horizons must be a non-empty unique positive list")
    if args.window < 2:
        raise SystemExit("--window must be >=2")
    if args.batch_series < 5:
        raise SystemExit("--batch-series must be >=5 for one OHLCV group")
    if args.chunk_windows < 1:
        raise SystemExit("--chunk-windows must be positive")
    if not labels.is_file():
        _generate_labels(labels, data_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.name)
    stem = f"{safe_name}_{args.control}"
    result_path = out_dir / f"{stem}.json"
    cache_path = out_dir / f"{stem}_emb.npy"
    os.environ.update({
        "FFM_ROOT": str(ROOT),
        "DATA_DIR": str(data_dir),
        "TREND_LABELS": str(labels),
        "CKPT_NAME": args.name,
        "CKPT_PATH": args.checkpoint,
        "CKPT_SHA256": checkpoint_hash,
        "EMB_CACHE": str(cache_path),
        "ATLAS_OUT": str(result_path),
        "ATLAS_BACKBONE": "chronos2",
        "ATLAS_CONTROL": args.control,
        "ATLAS_WINDOW": str(args.window),
        "ATLAS_HORIZONS": ",".join(str(item) for item in horizons),
        "ATLAS_POOL": args.pool,
        "ATLAS_BATCH": str(args.batch_series),
        "ATLAS_CHUNK": str(args.chunk_windows),
        "ATLAS_TRAIN_PER_STREAM": str(args.train_per_stream),
        "ATLAS_EVAL_PER_STREAM": str(args.eval_per_stream),
        "DEVICE": args.device,
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    atlas = _load_atlas()
    if args.preflight_only:
        bars, keys, fields = atlas._load_pool()
        payload = {
            "schema": "ffm_chronos2_probe_atlas_preflight_v1",
            "checkpoint_sha256": checkpoint_hash,
            "pool_rows": len(keys),
            "pool_sha256": atlas._pool_sha256(keys),
            "source_sha256": atlas._source_sha256(bars),
            "window": args.window,
            "horizons": list(horizons),
            "streams": sorted(set(fields["stream"])),
        }
        destination = out_dir / f"{stem}_preflight.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"[chronos2-atlas] PREFLIGHT PASS rows={len(keys):,} -> {destination}",
            flush=True,
        )
        return payload
    return atlas.main()


if __name__ == "__main__":
    main()
