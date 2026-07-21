#!/usr/bin/env python3
"""Run the complete clean Mantis SSL lineage locally, in production order.

The historical stage scripts were Colab launchers. This driver is the local/MPS
replacement and uses the repository's sealed continuous-contract ``data/`` corpus:

    public Mantis-8M
      -> masked OHLCV
      -> temporal contrastive
      -> multi-horizon candle seq2seq
      -> NextLeg

Every stage uses the immutable 2026-01-01 holdout, writes a distinct checkpoint,
and records both data hashes and the exact parent-checkpoint hash. Stages execute
in separate processes so Apple unified memory is released between objectives.
Completed stages are skipped; an interrupted stage resumes its progressive best
checkpoint. Any failure stops the lineage before a child can use a bad parent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futures_foundation.data_provenance import seal_continuous_streams, sha256


TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
TIMEFRAMES = ("1min", "3min", "5min", "15min")
HOLDOUT_START = "2026-01-01"
STAGE_ORDER = ("mask", "contrastive", "seq2seq", "nextleg")


@dataclass(frozen=True)
class Stage:
    name: str
    filename: str
    parent: str | None
    epochs: int
    batch: dict[str, int]


STAGES = (
    Stage("mask", "mantis_ssl_ohlcv.pt", None, 60,
          {"cuda": 1024, "mps": 256, "cpu": 64}),
    # Contrastive materializes five views per anchor; 32 is intentionally conservative
    # on a 16-GB M1. CUDA keeps the historically validated batch of 128.
    Stage("contrastive", "mantis_ssl_regime_from_mask.pt", "mask", 60,
          {"cuda": 128, "mps": 32, "cpu": 16}),
    Stage("seq2seq", "mantis_ssl_ctr_seq2seq.pt", "contrastive", 120,
          {"cuda": 512, "mps": 128, "cpu": 32}),
    Stage("nextleg", "mantis_ssl_nextleg.pt", "seq2seq", 120,
          {"cuda": 512, "mps": 128, "cpu": 32}),
)


def _device() -> str:
    import torch

    requested = os.environ.get("DEVICE")
    if requested:
        if requested not in {"cuda", "mps", "cpu"}:
            raise SystemExit(f"unsupported DEVICE={requested!r}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stage_map(out_dir: Path) -> dict[str, Path]:
    return {stage.name: out_dir / stage.filename for stage in STAGES}


def _stage_config(stage: str) -> dict:
    """Banked, non-experimental recipe for each clean-lineage objective."""
    common = dict(
        seq=64, max_jitter=16, val_frac=0.1, holdout_start=HOLDOUT_START,
        steps_per_epoch=int(os.environ.get("STEPS_PER_EPOCH", "200")),
        patience=int(os.environ.get("PATIENCE", "8")), seed=int(os.environ.get("SEED", "0")),
        probe=True, controls=(), compile_model=False,
    )
    if stage == "contrastive":
        return {**common, "pretext": "contrastive", "pos_deltas": (2, 16, 64),
                "far_min": 512, "temperature": 0.1, "aug_noise": 0.10,
                "aug_scale": 0.20, "aug_tmask": 0.15, "crop_max": 0.2,
                "vol_weight": 1.0, "new_channels": 8, "proj_dim": 128,
                "metrics_n": 768, "lr": 1e-4, "weight_decay": 0.05,
                "clamp": 10.0, "grad_clip": 1.0, "freeze_encoder_layers": 0}
    if stage == "seq2seq":
        # Intended ctr_seq2seq base—not the later lc512/long-horizon experiment.
        return {**common, "pretext": "forecast", "horizons": (5, 10, 20, 25),
                "context_lengths": (64, 100, 150, 200), "objective": "candle_mse",
                "new_channels": 3, "dir_weight": 0.0, "lr": 0.0001188117389055629,
                "weight_decay": 0.0, "clamp": 10.0, "grad_clip": 1.0,
                "freeze_encoder_layers": 2}
    if stage == "nextleg":
        return {**common, "pretext": "nextleg", "horizons": (5, 10, 20, 25),
                "context_lengths": (64, 100, 150, 200), "leg_k": 2,
                "leg_cap": 256, "leg_w": 1.0, "mse_weight": 1.0,
                "new_channels": 3, "lr": 0.0001188117389055629,
                "weight_decay": 0.0, "clamp": 10.0, "grad_clip": 1.0,
                "freeze_encoder_layers": 2}
    raise KeyError(stage)


def _seal(data_dir: Path) -> dict:
    streams = ((ticker, timeframe) for ticker in TICKERS for timeframe in TIMEFRAMES)
    result = seal_continuous_streams(data_dir, streams, repo_root=ROOT)
    if set(result["streams"]) != {
            f"{ticker}@{timeframe}" for ticker in TICKERS for timeframe in TIMEFRAMES}:
        raise RuntimeError("data provenance did not seal the exact 9x4 universe")
    return result


def _write_lineage(path: Path, stage: str, parent: Path | None, provenance: dict,
                   config: dict) -> None:
    report = {
        **provenance,
        "stage": stage,
        "holdout_start": HOLDOUT_START,
        "parent": ({"kind": "public_model", "model_id": "paris-noah/Mantis-8M"}
                   if parent is None else
                   {"kind": "checkpoint", "path": str(parent), "sha256": sha256(parent)}),
        "config": config,
    }
    Path(str(path) + ".data_provenance.json").write_text(
        json.dumps(report, indent=2, default=list) + "\n")


def _run_child(args: argparse.Namespace) -> None:
    """Execute one non-mask stage. Parent mode runs us in a fresh process."""
    from futures_foundation.finetune import ssl

    stage_by_name = {stage.name: stage for stage in STAGES}
    stage = stage_by_name[args.run_stage]
    out_dir = Path(args.out_dir).resolve()
    paths = _stage_map(out_dir)
    out_path = paths[stage.name]
    parent = paths[stage.parent] if stage.parent else None
    if parent is None or not parent.is_file():
        raise FileNotFoundError(f"valid parent checkpoint missing for {stage.name}: {parent}")
    data_dir = Path(args.data_dir).resolve()
    provenance = _seal(data_dir)
    config = _stage_config(stage.name)
    epochs = int(os.environ.get(f"{stage.name.upper()}_EPOCHS", str(stage.epochs)))
    batch = int(os.environ.get(f"{stage.name.upper()}_BATCH", str(stage.batch[args.device])))
    resume = out_path.exists() and not Path(str(out_path) + ".report.json").exists()
    _write_lineage(out_path, stage.name, parent, provenance,
                   {**config, "epochs": epochs, "batch": batch, "device": args.device})
    print(f"\n[{stage.name}] parent={parent}\n[{stage.name}] output={out_path}", flush=True)
    verdict = ssl.loop_ssl(
        data_dir=str(data_dir), out_path=str(out_path), tickers=TICKERS, tfs=TIMEFRAMES,
        backbone_ckpt=str(parent), device=args.device, epochs=epochs, batch=batch,
        resume=resume, **config)
    if not out_path.is_file() or not Path(str(out_path) + ".report.json").is_file():
        raise RuntimeError(f"{stage.name} returned without complete checkpoint/report artifacts")
    print(f"[{stage.name}] COMPLETE verdict={verdict}", flush=True)


def _notify(message: str) -> None:
    if sys.platform != "darwin":
        return
    subprocess.run(["osascript", "-e",
                    f'display notification {json.dumps(message)} with title "FFM SSL"'],
                   check=False)


def _run_logged(command: list[str], *, env: dict[str, str], log_path: Path) -> tuple[int, bool]:
    """Stream a child to terminal+log and identify accelerator-memory failures."""
    oom = False
    needles = ("out of memory", "mps backend out of memory", "mpsallocator")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            if any(needle in line.lower() for needle in needles):
                oom = True
        return process.wait(), oom


def _run_parent(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = _seal(data_dir)
    paths = _stage_map(out_dir)
    python = Path(sys.executable).resolve()
    device = args.device or _device()

    print("\nCLEAN SSL PIPELINE PREFLIGHT PASSED")
    print(f"  data    : {data_dir}")
    print(f"  holdout : >= {HOLDOUT_START} excluded from every stage")
    print(f"  device  : {device}")
    print(f"  output  : {out_dir}")
    for stage in STAGES:
        print(f"  {stage.name:11s}: batch={stage.batch[device]:4d} max_epochs={stage.epochs:3d} "
              f"-> {stage.filename}")
    if args.preflight_only:
        return

    for stage in STAGES:
        out_path = paths[stage.name]
        report_path = Path(str(out_path) + ".report.json")
        if out_path.is_file() and report_path.is_file():
            print(f"\n[{stage.name}] already complete; skipping {out_path}", flush=True)
            continue
        env = os.environ.copy()
        batch = int(env.get(f"{stage.name.upper()}_BATCH", stage.batch[device]))
        min_batch = 8 if stage.name != "contrastive" else 4
        while True:
            if stage.name == "mask":
                env.update({"DATA_DIR": str(data_dir), "OUT_PATH": str(out_path),
                            "DEVICE": device, "BATCH": str(batch),
                            "EPOCHS": str(stage.epochs)})
                if out_path.exists():
                    env["RESUME"] = "1"
                command = [str(python), str(ROOT / "scripts" / "mantis_ssl_pretrain.py")]
            else:
                env[f"{stage.name.upper()}_BATCH"] = str(batch)
                command = [str(python), str(Path(__file__).resolve()),
                           "--run-stage", stage.name, "--data-dir", str(data_dir),
                           "--out-dir", str(out_dir), "--device", device]
            print(f"\n[{stage.name}] START batch={batch}", flush=True)
            code, oom = _run_logged(command, env=env,
                                    log_path=out_dir / "logs" / f"{stage.name}.log")
            if code == 0:
                break
            if device == "mps" and oom and batch > min_batch:
                new_batch = max(min_batch, batch // 2)
                print(f"[{stage.name}] MPS memory limit at batch={batch}; retrying batch={new_batch}",
                      flush=True)
                batch = new_batch
                continue
            _notify(f"Clean SSL pipeline failed at {stage.name}")
            raise SystemExit(code)
        # Validate the full child artifact contract before allowing the next parent link.
        if not out_path.is_file() or not report_path.is_file():
            _notify(f"Clean SSL pipeline missing artifacts at {stage.name}")
            raise RuntimeError(f"incomplete stage artifacts: {stage.name}")
        print(f"[{stage.name}] PASS; advancing lineage", flush=True)

    final_path = paths["nextleg"]
    summary = {
        "schema": "ffm_clean_ssl_pipeline_v1",
        "holdout_start": HOLDOUT_START,
        "data_provenance": provenance,
        "stages": {stage.name: {"path": str(paths[stage.name]),
                                  "sha256": sha256(paths[stage.name])} for stage in STAGES},
        "final_checkpoint": str(final_path),
    }
    (out_dir / "pipeline_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    _notify("Clean SSL pipeline completed through NextLeg")
    print(f"\nPIPELINE COMPLETE\n  NextLeg: {final_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local/MPS clean Mantis SSL pipeline")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--out-dir", default=os.environ.get(
        "SSL_OUT_DIR", str(ROOT / "temp" / "clean_ssl_pre2026")))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-stage", choices=STAGE_ORDER, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run_stage:
        if not args.device:
            raise SystemExit("internal --run-stage requires --device")
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
