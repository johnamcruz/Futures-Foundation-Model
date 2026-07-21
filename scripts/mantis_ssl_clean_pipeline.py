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
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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
    lora = dict(lora_r=int(os.environ.get("LORA_R", "8")),
                lora_alpha=float(os.environ.get("LORA_ALPHA", "16")),
                lora_dropout=float(os.environ.get("LORA_DROPOUT", "0")))
    common = dict(
        seq=64, max_jitter=16, val_frac=0.1, holdout_start=HOLDOUT_START,
        steps_per_epoch=int(os.environ.get("STEPS_PER_EPOCH", "200")),
        patience=int(os.environ.get("PATIENCE", "8")), seed=int(os.environ.get("SEED", "0")),
        probe=True, controls=(), compile_model=False,
        log_every_steps=int(os.environ.get("SSL_LOG_EVERY_STEPS", "25")), **lora,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(out_dir: Path, event: str, **fields) -> None:
    """Append durable stage-level progress and update the current status snapshot."""
    record = {"time": _utc_now(), "event": event, **fields}
    events = out_dir / "pipeline_events.jsonl"
    with events.open("a", buffering=1) as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    status = out_dir / "pipeline_status.json"
    temporary = status.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(temporary, status)


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
            stamped = f"[{_utc_now()}] {line}"
            print(stamped, end="", flush=True)
            log.write(stamped)
            if any(needle in line.lower() for needle in needles):
                oom = True
        return process.wait(), oom


def _refresh_atlas_progress(out_dir: Path) -> dict:
    """Build a compact stage-over-stage capability table from completed atlas JSON files."""
    rows, previous = [], None
    for stage in STAGES:
        result_path = out_dir / "probe_atlas" / f"{stage.name}.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text())
        probes = payload.get("probes", {})
        family_means = {}
        for family in ("retention", "prediction"):
            aucs = [float(item["auc"]) for item in probes.values()
                    if item.get("family") == family]
            family_means[family + "_mean_auc"] = (sum(aucs) / len(aucs) if aucs else None)
        deltas = {}
        if previous is not None:
            for name in sorted(set(previous["probes"]) & set(probes)):
                deltas[name] = round(float(probes[name]["auc"])
                                     - float(previous["probes"][name]["auc"]), 4)
        row = {"stage": stage.name, "checkpoint_sha256": payload.get("checkpoint_sha256"),
               **family_means, "deltas_vs_previous": deltas, "probes": probes}
        rows.append(row)
        previous = row
    progress = {"schema": "ffm_probe_atlas_progress_v1", "updated_at": _utc_now(),
                "stages": rows}
    path = out_dir / "probe_atlas_progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(temporary, path)
    return progress


def _ensure_atlas_labels(*, data_dir: Path, out_dir: Path, provenance: dict,
                         python: Path) -> Path:
    """Generate one checkpoint-independent, pre-2026 lifecycle truth corpus for every atlas."""
    labels = out_dir / "probe_atlas" / "trend_lifecycle_labels_pre2026.npz"
    manifest = labels.with_suffix(".npz.provenance.json")
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    reusable = False
    if labels.is_file() and manifest.is_file():
        saved = json.loads(manifest.read_text())
        reusable = (saved.get("data_provenance_sha256") == provenance_hash
                    and saved.get("holdout_start") == HOLDOUT_START)
    if not reusable:
        generator = ROOT / "colabs" / "mantis" / "generate_trend_labels.py"
        if not generator.is_file():
            raise FileNotFoundError(f"lifecycle-label generator missing: {generator}")
        labels.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"FFM_ROOT": str(ROOT), "DATA_DIR": str(data_dir),
                    "TREND_LABELS": str(labels), "TREND_LABEL_END": HOLDOUT_START})
        _event(out_dir, "probe_atlas_labels_started", output=str(labels),
               holdout_start=HOLDOUT_START)
        code, _ = _run_logged(
            [str(python), str(generator)], env=env,
            log_path=out_dir / "logs" / "probe_atlas_labels.log")
        if code != 0 or not labels.is_file():
            _event(out_dir, "probe_atlas_labels_failed", exit_code=code,
                   log=str(out_dir / "logs" / "probe_atlas_labels.log"))
            raise RuntimeError("failed to generate clean Probe Atlas lifecycle labels")
        manifest.write_text(json.dumps({
            "schema": "ffm_probe_atlas_labels_v1",
            "holdout_start": HOLDOUT_START,
            "data_provenance_sha256": provenance_hash,
            "data_provenance": provenance,
        }, indent=2) + "\n")

    import numpy as np
    with np.load(labels, allow_pickle=False) as payload:
        if set(payload["ticker"].astype(str)) != set(TICKERS):
            raise RuntimeError("Probe Atlas labels do not contain the exact 9-ticker universe")
        if set(payload["tf"].astype(str)) != set(TIMEFRAMES):
            raise RuntimeError("Probe Atlas labels do not contain the exact 4-timeframe universe")
        if payload["ts"].max() >= np.datetime64(HOLDOUT_START):
            raise RuntimeError("Probe Atlas labels cross the immutable 2026 holdout")
        rows = int(len(payload["ticker"]))
        max_ts = str(payload["ts"].max())
    _event(out_dir, "probe_atlas_labels_ready", output=str(labels), rows=rows,
           max_timestamp=max_ts, data_provenance_sha256=provenance_hash)
    return labels


def _run_probe_atlas(stage: Stage, checkpoint: Path, *, out_dir: Path, device: str,
                     python: Path, labels: Path) -> dict:
    """Run the private capability atlas for one exact checkpoint; fail closed."""
    atlas_script = ROOT / "scripts" / "probe_atlas.py"
    if not atlas_script.is_file():
        raise FileNotFoundError(f"Probe Atlas missing: {atlas_script}")
    atlas_dir = out_dir / "probe_atlas"
    result_path = atlas_dir / f"{stage.name}.json"
    checkpoint_hash = sha256(checkpoint)
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if existing.get("checkpoint_sha256") == checkpoint_hash:
            _event(out_dir, "probe_atlas_skipped", stage=stage.name,
                   reason="matching_checkpoint", result=str(result_path))
            return _refresh_atlas_progress(out_dir)

    batch = int(os.environ.get("ATLAS_BATCH", "128" if device == "mps" else "512"))
    minimum = 16
    while True:
        env = os.environ.copy()
        env.update({"FFM_ROOT": str(ROOT), "CKPT_NAME": checkpoint.name,
                    "CKPT_PATH": str(checkpoint), "CKPT_SHA256": checkpoint_hash,
                    "EMB_CACHE": str(atlas_dir / f"{stage.name}_emb.npy"),
                    "ATLAS_OUT": str(result_path), "ATLAS_BATCH": str(batch),
                    "DEVICE": device, "TREND_LABELS": str(labels)})
        _event(out_dir, "probe_atlas_started", stage=stage.name, batch=batch,
               checkpoint=str(checkpoint), result=str(result_path))
        code, oom = _run_logged(
            [str(python), str(atlas_script)], env=env,
            log_path=out_dir / "logs" / f"{stage.name}_probe_atlas.log")
        if code == 0 and result_path.is_file():
            progress = _refresh_atlas_progress(out_dir)
            current = progress["stages"][-1]
            _event(out_dir, "probe_atlas_completed", stage=stage.name,
                   retention_mean_auc=current["retention_mean_auc"],
                   prediction_mean_auc=current["prediction_mean_auc"],
                   deltas_vs_previous=current["deltas_vs_previous"], result=str(result_path))
            return progress
        if device == "mps" and oom and batch > minimum:
            next_batch = max(minimum, batch // 2)
            _event(out_dir, "probe_atlas_oom_retry", stage=stage.name,
                   previous_batch=batch, next_batch=next_batch)
            batch = next_batch
            continue
        _event(out_dir, "probe_atlas_failed", stage=stage.name, exit_code=code,
               log=str(out_dir / "logs" / f"{stage.name}_probe_atlas.log"))
        raise RuntimeError(f"Probe Atlas failed for {stage.name}; refusing to advance")


def _run_parent(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = _seal(data_dir)
    atlas_script = ROOT / "scripts" / "probe_atlas.py"
    if not atlas_script.is_file():
        raise FileNotFoundError(f"Probe Atlas entrypoint missing: {atlas_script}")
    paths = _stage_map(out_dir)
    python = Path(sys.executable).resolve()
    device = args.device or _device()
    lora_r = int(args.lora_r)
    if lora_r < 0:
        raise SystemExit("--lora-r cannot be negative")
    os.environ["LORA_R"] = str(lora_r)
    os.environ["LORA_ALPHA"] = str(args.lora_alpha)
    os.environ["LORA_DROPOUT"] = str(args.lora_dropout)
    os.environ["SSL_LOG_EVERY_STEPS"] = str(args.log_every_steps)

    print("\nCLEAN SSL PIPELINE PREFLIGHT PASSED")
    print(f"  data    : {data_dir}")
    print(f"  holdout : >= {HOLDOUT_START} excluded from every stage")
    print(f"  device  : {device}")
    print(f"  tuning  : {'LoRA' if lora_r else 'full'}"
          + (f" r={lora_r} alpha={args.lora_alpha:g}" if lora_r else ""))
    print(f"  output  : {out_dir}")
    for stage in STAGES:
        print(f"  {stage.name:11s}: batch={stage.batch[device]:4d} max_epochs={stage.epochs:3d} "
              f"-> {stage.filename}")
    if args.preflight_only:
        return

    _event(out_dir, "pipeline_started", device=device, lora_r=lora_r,
           holdout_start=HOLDOUT_START, stage_order=STAGE_ORDER)
    atlas_labels = _ensure_atlas_labels(data_dir=data_dir, out_dir=out_dir,
                                        provenance=provenance, python=python)

    for stage in STAGES:
        out_path = paths[stage.name]
        report_path = Path(str(out_path) + ".report.json")
        if out_path.is_file() and report_path.is_file():
            print(f"\n[{stage.name}] already complete; skipping {out_path}", flush=True)
            _run_probe_atlas(stage, out_path, out_dir=out_dir, device=device, python=python,
                             labels=atlas_labels)
            _event(out_dir, "stage_skipped", stage=stage.name, reason="complete",
                   checkpoint=str(out_path))
            continue
        env = os.environ.copy()
        batch = int(env.get(f"{stage.name.upper()}_BATCH", stage.batch[device]))
        min_batch = 8 if stage.name != "contrastive" else 4
        while True:
            if stage.name == "mask":
                env.update({"DATA_DIR": str(data_dir), "OUT_PATH": str(out_path),
                            "DEVICE": device, "BATCH": str(batch),
                            "EPOCHS": str(stage.epochs), "LORA_R": str(lora_r),
                            "LORA_ALPHA": str(args.lora_alpha),
                            "LORA_DROPOUT": str(args.lora_dropout)})
                if out_path.exists():
                    env["RESUME"] = "1"
                command = [str(python), str(ROOT / "scripts" / "mantis_ssl_pretrain.py")]
            else:
                env[f"{stage.name.upper()}_BATCH"] = str(batch)
                command = [str(python), str(Path(__file__).resolve()),
                           "--run-stage", stage.name, "--data-dir", str(data_dir),
                           "--out-dir", str(out_dir), "--device", device]
            print(f"\n[{stage.name}] START batch={batch}", flush=True)
            _event(out_dir, "stage_started", stage=stage.name, batch=batch,
                   checkpoint=str(out_path), parent=stage.parent)
            code, oom = _run_logged(command, env=env,
                                    log_path=out_dir / "logs" / f"{stage.name}.log")
            if code == 0:
                break
            if device == "mps" and oom and batch > min_batch:
                new_batch = max(min_batch, batch // 2)
                print(f"[{stage.name}] MPS memory limit at batch={batch}; retrying batch={new_batch}",
                      flush=True)
                _event(out_dir, "stage_oom_retry", stage=stage.name,
                       previous_batch=batch, next_batch=new_batch)
                batch = new_batch
                continue
            _notify(f"Clean SSL pipeline failed at {stage.name}")
            _event(out_dir, "stage_failed", stage=stage.name, exit_code=code,
                   log=str(out_dir / "logs" / f"{stage.name}.log"))
            raise SystemExit(code)
        # Validate the full child artifact contract before allowing the next parent link.
        if not out_path.is_file() or not report_path.is_file():
            _notify(f"Clean SSL pipeline missing artifacts at {stage.name}")
            raise RuntimeError(f"incomplete stage artifacts: {stage.name}")
        _run_probe_atlas(stage, out_path, out_dir=out_dir, device=device, python=python,
                         labels=atlas_labels)
        print(f"[{stage.name}] PASS; advancing lineage", flush=True)
        _event(out_dir, "stage_completed", stage=stage.name, batch=batch,
               checkpoint=str(out_path), checkpoint_sha256=sha256(out_path),
               report=str(report_path))

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
    _event(out_dir, "pipeline_completed", final_checkpoint=str(final_path),
           manifest=str(out_dir / "pipeline_manifest.json"))
    print(f"\nPIPELINE COMPLETE\n  NextLeg: {final_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local/MPS clean Mantis SSL pipeline")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--out-dir", default=os.environ.get(
        "SSL_OUT_DIR", str(ROOT / "temp" / "clean_ssl_pre2026_lora")))
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "8")),
                        help="LoRA rank for every stage; 0 restores full fine-tuning")
    parser.add_argument("--lora-alpha", type=float,
                        default=float(os.environ.get("LORA_ALPHA", "16")))
    parser.add_argument("--lora-dropout", type=float,
                        default=float(os.environ.get("LORA_DROPOUT", "0")))
    parser.add_argument("--log-every-steps", type=int,
                        default=int(os.environ.get("SSL_LOG_EVERY_STEPS", "25")),
                        help="Print optimizer progress every N steps; 0 disables step logs")
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
