#!/usr/bin/env python3
"""Run the complete clean Mantis SSL lineage locally, in production order.

The historical stage scripts were Colab launchers. This driver is the local/MPS
replacement and uses the repository's sealed continuous-contract ``data/`` corpus:

    public Mantis-8M
      -> masked OHLCV
      -> causal Kaufman-regime contrastive
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
import math
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
PROBE_SPLIT_SCHEMA = "balanced_stream_purged_temporal_v1"
ATLAS_MEAN_RETENTION_TOLERANCE = 0.01
ATLAS_SINGLE_PROBE_TOLERANCE = 0.03


@dataclass(frozen=True)
class Stage:
    name: str
    filename: str
    parent: str | None
    epochs: int
    batch: dict[str, int]
    samples_per_epoch: int


# Canonical production lineage. Keep this tuple authoritative: parent execution,
# Probe Atlas progression, manifests, and completion all derive from it.
STAGES = (
    Stage("mask", "mantis_ssl_ohlcv.pt", None, 60,
          {"cuda": 1024, "mps": 256, "cpu": 64}, 51_200),
    Stage("contrastive", "mantis_ssl_regime_kaufman_from_mask.pt", "mask", 60,
          {"cuda": 256, "mps": 64, "cpu": 16}, 12_800),
    Stage("seq2seq", "mantis_ssl_kaufman_seq2seq.pt", "contrastive", 120,
          {"cuda": 512, "mps": 128, "cpu": 32}, 25_600),
    Stage("nextleg", "mantis_ssl_nextleg.pt", "seq2seq", 120,
          {"cuda": 512, "mps": 128, "cpu": 32}, 25_600),
)
STAGE_ORDER = tuple(stage.name for stage in STAGES)


def _steps_for(stage: Stage, batch: int, env: dict[str, str] | None = None) -> int:
    """Keep each epoch's sampled-data budget invariant as physical batch changes.

    Explicit per-stage/global step overrides retain their historical meaning.
    Otherwise an OOM fallback halves the batch and automatically doubles steps,
    preventing a memory retry from silently weakening training.
    """
    env = os.environ if env is None else env
    override = env.get(f"{stage.name.upper()}_STEPS", env.get("STEPS_PER_EPOCH"))
    return max(1, int(override)) if override is not None else math.ceil(stage.samples_per_epoch / batch)


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


def _assert_completed_stage_recipe(report_path: Path, *, sampling_mode: str) -> None:
    """Refuse to reuse a complete checkpoint from a different source mixture."""
    report = json.loads(report_path.read_text())
    # Reports written before source mixtures were configurable used the historical
    # bar-proportional behavior and contain no sampling_mode field.
    saved = report.get("config", {}).get("sampling_mode", "bar_proportional")
    if saved != sampling_mode:
        raise RuntimeError(
            f"completed checkpoint uses sampling_mode={saved!r}, requested {sampling_mode!r}; "
            "choose a new --out-dir so experimental lineages cannot overwrite or reuse each other")


def _assert_stage_parent(out_path: Path, stage: Stage,
                         paths: dict[str, Path]) -> None:
    """Refuse to resume or reuse a child produced from a different parent.

    This matters when an output directory survives a curriculum change. A child
    filename alone cannot prove which encoder initialized it.
    """
    if stage.parent is None or not out_path.exists():
        return
    provenance_path = Path(str(out_path) + ".data_provenance.json")
    if not provenance_path.is_file():
        raise RuntimeError(
            f"cannot establish parent identity for existing {stage.name} checkpoint: "
            f"{provenance_path} is missing; move it aside and restart this stage")
    saved = json.loads(provenance_path.read_text())
    expected_parent = paths[stage.parent]
    expected_hash = sha256(expected_parent)
    parent = saved.get("parent", {})
    if saved.get("stage") != stage.name or parent.get("sha256") != expected_hash:
        raise RuntimeError(
            f"existing {stage.name} checkpoint has the wrong parent; expected "
            f"{stage.parent} sha256={expected_hash}, found {parent.get('sha256')!r}; "
            "move the stale child artifacts aside and restart this stage")


def _stage_config(stage: str) -> dict:
    """Banked, non-experimental recipe for each clean-lineage objective."""
    lora = dict(lora_r=int(os.environ.get("LORA_R", "8")),
                lora_alpha=float(os.environ.get("LORA_ALPHA", "16")),
                lora_dropout=float(os.environ.get("LORA_DROPOUT", "0")))
    common = dict(
        seq=64, max_jitter=16, val_frac=0.1, holdout_start=HOLDOUT_START,
        sampling_mode=os.environ.get("SAMPLING_MODE", "bar_proportional"),
        steps_per_epoch=int(os.environ.get("STEPS_PER_EPOCH", "200")),
        patience=int(os.environ.get("PATIENCE", "8")), seed=int(os.environ.get("SEED", "0")),
        probe=True, controls=(), compile_model=False,
        log_every_steps=int(os.environ.get("SSL_LOG_EVERY_STEPS", "25")), **lora,
        )
    if stage == "contrastive":
        return {**common, "pretext": "contrastive", "regime_key": "kaufman",
                "kaufman_chop": 0.25, "kaufman_trend": 0.50,
                "pos_deltas": (2, 16, 64),
                "far_min": 512, "temperature": 0.1, "aug_noise": 0.02,
                "aug_scale": 0.10, "aug_tmask": 0.0, "crop_max": 0.0,
                "vol_weight": 0.0, "new_channels": 8, "proj_dim": 128,
                "metrics_n": 768,
                "lr": float(os.environ.get("CONTRASTIVE_LR", "5e-5")),
                "weight_decay": 0.05,
                "clamp": 10.0, "grad_clip": 1.0, "freeze_encoder_layers": 2}
    if stage == "seq2seq":
        # Banked multi-horizon recipe; the parent-hash contract guarantees this run
        # starts from the retained Kaufman-regime representation.
        return {**common, "pretext": "forecast", "horizons": (5, 10, 20, 25),
                "context_lengths": (64, 100, 150, 200), "objective": "candle_mse",
                "new_channels": 3, "dir_weight": 0.0,
                "lr": float(os.environ.get("SEQ2SEQ_LR", "4e-5")),
                "weight_decay": 0.0, "clamp": 10.0, "grad_clip": 1.0,
                "freeze_encoder_layers": 2}
    if stage == "nextleg":
        return {**common, "pretext": "nextleg", "horizons": (5, 10, 20, 25),
                "context_lengths": (64, 100, 150, 200), "leg_k": 2,
                "leg_cap": 256, "leg_w": 1.0, "mse_weight": 1.0,
                "new_channels": 3,
                "lr": float(os.environ.get("NEXTLEG_LR", "3e-5")),
                "weight_decay": 0.0, "clamp": 10.0, "grad_clip": 1.0,
                "freeze_encoder_layers": 2}
    raise KeyError(stage)


def _assert_stage_verdict(report_path: Path) -> None:
    """Fail closed on stale probe methodology or a failed representation/control gate."""
    report = json.loads(report_path.read_text())
    schema = report.get("probe", {}).get("split_schema")
    if schema != PROBE_SPLIT_SCHEMA:
        raise RuntimeError(
            f"stale probe split in {report_path}: {schema!r}; expected {PROBE_SPLIT_SCHEMA!r}")
    verdict = report.get("verdict", {})
    if not verdict.get("all_pass", False):
        raise RuntimeError(
            f"stage validation failed in {report_path}: "
            f"representation_pass={verdict.get('representation_pass')} "
            f"beats_controls={verdict.get('beats_controls')}")


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
    _assert_stage_parent(out_path, stage, paths)
    data_dir = Path(args.data_dir).resolve()
    provenance = _seal(data_dir)
    config = _stage_config(stage.name)
    epochs = int(os.environ.get(f"{stage.name.upper()}_EPOCHS", str(stage.epochs)))
    batch = int(os.environ.get(f"{stage.name.upper()}_BATCH", str(stage.batch[args.device])))
    steps = _steps_for(stage, batch)
    config["steps_per_epoch"] = steps
    resume = out_path.exists() and not Path(str(out_path) + ".report.json").exists()
    _write_lineage(out_path, stage.name, parent, provenance,
                   {**config, "epochs": epochs, "batch": batch, "steps_per_epoch": steps,
                    "samples_per_epoch": stage.samples_per_epoch, "device": args.device})
    print(f"\n[{stage.name}] parent={parent}\n[{stage.name}] output={out_path}"
          f"\n[{stage.name}] batch={batch} steps={steps} "
          f"samples/epoch={batch * steps:,}", flush=True)
    verdict = ssl.loop_ssl(
        data_dir=str(data_dir), out_path=str(out_path), tickers=TICKERS, tfs=TIMEFRAMES,
        backbone_ckpt=str(parent), device=args.device, epochs=epochs, batch=batch,
        resume=resume, **config)
    if not out_path.is_file() or not Path(str(out_path) + ".report.json").is_file():
        raise RuntimeError(f"{stage.name} returned without complete checkpoint/report artifacts")
    _assert_stage_verdict(Path(str(out_path) + ".report.json"))
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
    """Stream a child without letting transient Drive logging kill the workload.

    Colab's FUSE-mounted Drive can briefly raise ``EIO`` during otherwise healthy GPU
    work. Console output remains authoritative; when the durable log becomes unavailable,
    subsequent lines are also spooled to the runtime disk for diagnosis.
    """
    oom = False
    needles = ("out of memory", "mps backend out of memory", "mpsallocator")
    durable = fallback = None
    fallback_path = Path(env.get(
        "FFM_LOCAL_LOG_DIR",
        "/content/ffm_runtime_logs" if Path("/content").is_dir()
        else str(ROOT / "temp" / "runtime_logs"))) / log_path.name
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        durable = log_path.open("a", buffering=1)
    except OSError as exc:
        print(f"[log-warning] durable log unavailable ({exc}); using {fallback_path}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            stamped = f"[{_utc_now()}] {line}"
            print(stamped, end="", flush=True)
            if durable is not None:
                try:
                    durable.write(stamped)
                except OSError as exc:
                    print(f"[log-warning] durable log write failed ({exc}); "
                          f"continuing via {fallback_path}", flush=True)
                    try:
                        durable.close()
                    except OSError:
                        pass
                    durable = None
            if durable is None and fallback is not False:
                try:
                    if fallback is None:
                        fallback_path.parent.mkdir(parents=True, exist_ok=True)
                        fallback = fallback_path.open("a", buffering=1)
                    fallback.write(stamped)
                except OSError as exc:
                    if fallback is not False:
                        print(f"[log-warning] runtime log also unavailable ({exc}); "
                              "continuing with console output", flush=True)
                    fallback = False
            if any(needle in line.lower() for needle in needles):
                oom = True
        return process.wait(), oom
    finally:
        for handle in (durable, fallback if fallback is not False else None):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass


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


def _assert_atlas_retention(progress: dict, stage_name: str) -> None:
    """Prevent a child objective from advancing after material capability loss.

    Family means catch broad forgetting while the per-probe bound prevents one
    important market-context capability from being hidden by gains elsewhere.
    """
    rows = progress.get("stages", [])
    index = next((i for i, row in enumerate(rows) if row.get("stage") == stage_name), None)
    if index is None or index == 0:
        return
    current, previous = rows[index], rows[index - 1]
    failures = []
    for metric in ("retention_mean_auc", "prediction_mean_auc"):
        before, after = previous.get(metric), current.get(metric)
        if before is not None and after is not None:
            delta = float(after) - float(before)
            if delta < -ATLAS_MEAN_RETENTION_TOLERANCE:
                failures.append(f"{metric} delta={delta:+.4f}")
    for name, delta in current.get("deltas_vs_previous", {}).items():
        if float(delta) < -ATLAS_SINGLE_PROBE_TOLERANCE:
            failures.append(f"{name} delta={float(delta):+.4f}")
    if failures:
        raise RuntimeError(
            f"Probe Atlas retention failed after {stage_name}: " + ", ".join(failures))


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
        generator = ROOT / "scripts" / "generate_trend_labels.py"
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
                     python: Path, labels: Path, data_dir: Path) -> dict:
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
                    "DATA_DIR": str(data_dir),
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
    # Do NOT resolve this path: a virtualenv's python is normally a symlink to the base
    # interpreter. Resolving it silently discards the venv for child stages (and therefore
    # installed packages such as torch). Keep the exact executable used to launch the master.
    python = Path(sys.executable).absolute()
    device = args.device or _device()
    lora_r = int(args.lora_r)
    if lora_r < 0:
        raise SystemExit("--lora-r cannot be negative")
    os.environ["LORA_R"] = str(lora_r)
    os.environ["LORA_ALPHA"] = str(args.lora_alpha)
    os.environ["LORA_DROPOUT"] = str(args.lora_dropout)
    os.environ["SAMPLING_MODE"] = args.sampling_mode
    os.environ["SSL_LOG_EVERY_STEPS"] = str(args.log_every_steps)

    print("\nCLEAN SSL PIPELINE PREFLIGHT PASSED")
    print(f"  data    : {data_dir}")
    print(f"  holdout : >= {HOLDOUT_START} excluded from every stage")
    print(f"  device  : {device}")
    print(f"  tuning  : {'LoRA' if lora_r else 'full'}"
          + (f" r={lora_r} alpha={args.lora_alpha:g}" if lora_r else ""))
    print(f"  sampling: {args.sampling_mode}")
    print(f"  output  : {out_dir}")
    for stage in STAGES:
        steps = _steps_for(stage, stage.batch[device])
        adaptation = ""
        if stage.name != "mask":
            cfg = _stage_config(stage.name)
            adaptation = (f" lr={cfg['lr']:.1e}"
                          f" freeze={cfg['freeze_encoder_layers']}")
        print(f"  {stage.name:11s}: batch={stage.batch[device]:4d} steps={steps:3d} "
              f"samples/epoch={stage.batch[device] * steps:6,d} max_epochs={stage.epochs:3d} "
              f"{adaptation} -> {stage.filename}")
    if args.preflight_only:
        return

    _event(out_dir, "pipeline_started", device=device, lora_r=lora_r,
           sampling_mode=args.sampling_mode,
           holdout_start=HOLDOUT_START, stage_order=STAGE_ORDER)
    atlas_labels = _ensure_atlas_labels(data_dir=data_dir, out_dir=out_dir,
                                        provenance=provenance, python=python)

    for stage in STAGES:
        out_path = paths[stage.name]
        report_path = Path(str(out_path) + ".report.json")
        # The Mask checkpoint is expensive and independently crash-safe. If its report predates
        # the balanced per-stream probe, discard only the stale report and re-finalize the exact
        # checkpoint; never retrain REAL merely to update evaluation methodology.
        if stage.name == "mask" and out_path.is_file() and report_path.is_file():
            saved = json.loads(report_path.read_text())
            if saved.get("probe", {}).get("split_schema") != PROBE_SPLIT_SCHEMA:
                report_path.unlink()
                _event(out_dir, "stage_probe_invalidated", stage=stage.name,
                       reason="stale_split_schema", checkpoint=str(out_path))
        if out_path.is_file() and report_path.is_file():
            _assert_completed_stage_recipe(report_path, sampling_mode=args.sampling_mode)
            _assert_stage_parent(out_path, stage, paths)
            _assert_stage_verdict(report_path)
            print(f"\n[{stage.name}] already complete; skipping {out_path}", flush=True)
            progress = _run_probe_atlas(
                stage, out_path, out_dir=out_dir, device=device, python=python,
                labels=atlas_labels, data_dir=data_dir)
            _assert_atlas_retention(progress, stage.name)
            _event(out_dir, "stage_skipped", stage=stage.name, reason="complete",
                   checkpoint=str(out_path))
            continue
        env = os.environ.copy()
        # Keep explicit user overrides separate from the per-attempt value we
        # inject below, so an OOM retry can recompute steps for its smaller batch.
        budget_env = env.copy()
        batch = int(env.get(f"{stage.name.upper()}_BATCH", stage.batch[device]))
        min_batch = 8 if stage.name != "contrastive" else 4
        while True:
            steps = _steps_for(stage, batch, budget_env)
            if stage.name == "mask":
                env.update({"DATA_DIR": str(data_dir), "OUT_PATH": str(out_path),
                            "DEVICE": device, "BATCH": str(batch),
                            "EPOCHS": str(stage.epochs), "STEPS": str(steps),
                            "CONTROL_EPOCHS": str(args.control_epochs),
                            "LORA_R": str(lora_r),
                            "LORA_ALPHA": str(args.lora_alpha),
                            "LORA_DROPOUT": str(args.lora_dropout),
                            "SAMPLING_MODE": args.sampling_mode})
                real_marker = Path(str(out_path) + ".real_complete.json")
                if out_path.exists() and (args.reuse_mask_real or real_marker.exists()):
                    env["REUSE_REAL_CHECKPOINT"] = "1"
                elif out_path.exists():
                    env["RESUME"] = "1"
                command = [str(python), str(ROOT / "scripts" / "mantis_ssl_pretrain.py")]
            else:
                env[f"{stage.name.upper()}_BATCH"] = str(batch)
                env["STEPS_PER_EPOCH"] = str(steps)
                command = [str(python), str(Path(__file__).resolve()),
                           "--run-stage", stage.name, "--data-dir", str(data_dir),
                           "--out-dir", str(out_dir), "--device", device]
            print(f"\n[{stage.name}] START batch={batch} steps={steps} "
                  f"samples/epoch={batch * steps:,}", flush=True)
            _event(out_dir, "stage_started", stage=stage.name, batch=batch, steps=steps,
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
        _assert_stage_verdict(report_path)
        progress = _run_probe_atlas(
            stage, out_path, out_dir=out_dir, device=device, python=python,
            labels=atlas_labels, data_dir=data_dir)
        _assert_atlas_retention(progress, stage.name)
        print(f"[{stage.name}] PASS; advancing lineage", flush=True)
        _event(out_dir, "stage_completed", stage=stage.name, batch=batch,
               checkpoint=str(out_path), checkpoint_sha256=sha256(out_path),
               report=str(report_path))

    final_path = paths["nextleg"]
    summary = {
        "schema": "ffm_clean_ssl_pipeline_v1",
        "holdout_start": HOLDOUT_START,
        "sampling_mode": args.sampling_mode,
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
    parser.add_argument("--sampling-mode", choices=("bar_proportional", "uniform_stream"),
                        default=os.environ.get("SAMPLING_MODE", "bar_proportional"),
                        help="Training source mixture; default preserves historical behavior")
    parser.add_argument("--log-every-steps", type=int,
                        default=int(os.environ.get("SSL_LOG_EVERY_STEPS", "25")),
                        help="Print optimizer progress every N steps; 0 disables step logs")
    parser.add_argument("--control-epochs", type=int,
                        default=int(os.environ.get("CONTROL_EPOCHS", "8")),
                        help="maximum epochs for each diagnostic shuffle/random control")
    parser.add_argument("--reuse-mask-real", action="store_true",
                        default=os.environ.get("REUSE_MASK_REAL") == "1",
                        help="finalize an existing REAL Mask checkpoint without retraining it")
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
