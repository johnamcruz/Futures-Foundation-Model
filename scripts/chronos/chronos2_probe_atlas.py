#!/usr/bin/env python3
"""Run the public 9x4 Probe Atlas with a frozen Chronos-2 checkpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
ATLAS_SCRIPT = ROOT / "scripts" / "probe_atlas.py"
DEFAULT_RUN = ROOT / "temp" / "chronos2_small_36stream"
DEFAULT_CHECKPOINT = DEFAULT_RUN / "contrastive_kaufman_40" / "checkpoint"
DEFAULT_ATLAS_DIR = DEFAULT_RUN / "probe_atlas"
DEFAULT_LABELS = DEFAULT_ATLAS_DIR / "trend_lifecycle_labels_pre2026.npz"
MPS_SAFE_MAX_BATCH_SERIES = 320
MPS_SAFE_MAX_CHUNK_WINDOWS = 1024
CHRONOS2_STAGE_SCHEMAS = {
    "ffm_chronos2_mask_v1": "mask",
    "ffm_chronos2_contrastive_v2": "contrastive",
    "ffm_chronos2_volume_structure_ssl_v3": "volume_structure_ssl",
}
CHRONOS2_MODEL_ID = "autogluon/chronos-2-small"
VOLUME_STRUCTURE_TRAINER_SCHEMA = (
    "ffm_chronos2_volume_structure_trainer_v4"
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


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


def _volume_structure_run_identity(
    *,
    parent_sha256: str,
    data_identity_sha256: str,
    config: dict,
) -> str:
    """Recompute the trainer identity from the published immutable inputs."""
    payload = {
        "schema": VOLUME_STRUCTURE_TRAINER_SCHEMA,
        "parent_sha256": parent_sha256,
        "data_identity_sha256": data_identity_sha256,
        "config": config,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _volume_structure_stage_contract(
        report: dict,
        report_path: Path,
        checkpoint_path: Path,
) -> bool:
    """Authenticate the v3 native-checkpoint and discarded-head contract."""
    run_identity = report.get("run_identity_sha256")
    native = report.get("checkpoint_only_validation")
    artifact = report.get("final_artifact_contract")
    config = report.get("config")
    source_parent = report.get("parent")
    data_identity = report.get("data_identity_sha256")
    if (
        not isinstance(run_identity, str)
        or _HEX64.fullmatch(run_identity) is None
        or not isinstance(native, dict)
        or native.get("status") != "pass"
        or native.get("contract")
        != "freshly_reloaded_lora_native_reg_without_temporary_heads"
        or not isinstance(artifact, dict)
        or not isinstance(config, dict)
        or not isinstance(source_parent, dict)
        or not isinstance(source_parent.get("sha256"), str)
        or _HEX64.fullmatch(source_parent["sha256"]) is None
        or not isinstance(data_identity, str)
        or _HEX64.fullmatch(data_identity) is None
        or run_identity != _volume_structure_run_identity(
            parent_sha256=source_parent["sha256"],
            data_identity_sha256=data_identity,
            config=config,
        )
        or (report_path.parent / "trainer.pt").exists()
        or (report_path.parent / ".trainer.pt.tmp").exists()
    ):
        return False
    parent = native.get("parent")
    checkpoint = native.get("checkpoint")
    lift = native.get("loss_lift_parent_minus_checkpoint")
    margin = native.get("required_margin")
    if (
        not isinstance(parent, dict)
        or not isinstance(parent.get("aggregate"), dict)
        or not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("aggregate"), dict)
        or not isinstance(lift, dict)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        return False
    for objective in ("participation", "concentration"):
        before = parent["aggregate"].get(objective)
        after = checkpoint["aggregate"].get(objective)
        saved_lift = lift.get(objective)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (before, after, saved_lift)
        ):
            return False
        measured = float(before) - float(after)
        if (
            not math.isclose(measured, float(saved_lift), abs_tol=1e-9)
            or measured <= float(margin)
        ):
            return False
    selection = config.get("checkpoint_selection")
    base = config.get("base_model")
    files = artifact.get("checkpoint_files")
    actual_files = sorted(
        str(path.relative_to(checkpoint_path))
        for path in checkpoint_path.rglob("*")
        if path.is_file()
    )
    return bool(
        isinstance(selection, dict)
        and selection.get("contract")
        == "gate_feasible_weighted_native_reg_participation_concentration_v1"
        and selection.get("temporary_head_metrics_used") is False
        and isinstance(base, dict)
        and base.get("model_id") == CHRONOS2_MODEL_ID
        and isinstance(base.get("revision"), str)
        and _HEX40.fullmatch(base["revision"]) is not None
        and isinstance(base.get("weights_sha256"), str)
        and _HEX64.fullmatch(base["weights_sha256"]) is not None
        and isinstance(base.get("config_sha256"), str)
        and _HEX64.fullmatch(base["config_sha256"]) is not None
        and artifact.get("temporary_heads_in_checkpoint") is False
        and artifact.get("ssl_heads_required_for_inference") is False
        and artifact.get("trainer_state")
        == "discarded_after_successful_checkpoint"
        and artifact.get("inference_requires")
        == ["chronos_base_model", "lora_checkpoint"]
        and isinstance(files, list)
        and files
        and sorted(files) == actual_files
        and not any(
            token in Path(name).name.lower()
            for name in files
            for token in ("head", "decoder", "projection", "trainer")
        )
    )


def _stage_identity(
    checkpoint: str,
    checkpoint_sha256: str,
    stage_report: Path | None,
    *,
    context_length: int,
) -> dict:
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        if stage_report is not None:
            raise SystemExit(
                "--stage-report cannot authenticate a remote checkpoint")
        return {
            "stage_report_path": None,
            "stage_report_sha256": None,
            "parent_checkpoint_sha256": None,
            "data_identity_sha256": None,
            "base_revision": None,
            "base_weights_sha256": None,
            "base_config_sha256": None,
        }
    report_path = (
        Path(stage_report).expanduser()
        if stage_report is not None
        else checkpoint_path.parent / "report.json"
    ).resolve()
    if not report_path.is_file():
        raise SystemExit(
            f"Chronos-2 stage report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Chronos-2 stage report is unreadable: {report_path}") from exc
    if not isinstance(report, dict):
        raise SystemExit(
            f"Chronos-2 stage report must be an object: {report_path}")
    parent = report.get("parent")
    saved_checkpoint = report.get("checkpoint")
    config = report.get("config")
    expected_stage = CHRONOS2_STAGE_SCHEMAS.get(report.get("schema"))
    specialized_contract = (
        expected_stage != "volume_structure_ssl"
        or _volume_structure_stage_contract(
            report, report_path, checkpoint_path)
    )
    parent_sha256 = parent.get("sha256") if isinstance(parent, dict) else None
    data_sha256 = report.get("data_identity_sha256")
    if (
        expected_stage is None
        or not specialized_contract
        or report.get("status") != "complete"
        or report.get("stage") != expected_stage
        or not isinstance(saved_checkpoint, dict)
        or saved_checkpoint.get("sha256") != checkpoint_sha256
        or not isinstance(config, dict)
        or config.get("context_length") != context_length
        or set(config.get("timeframes", ()))
        != {"1min", "3min", "5min", "15min"}
        or not isinstance(parent_sha256, str)
        or len(parent_sha256) != 64
        or not isinstance(data_sha256, str)
        or len(data_sha256) != 64
    ):
        raise SystemExit(
            f"Chronos-2 stage report contract is invalid: {report_path}")
    return {
        "stage_report_path": str(report_path),
        "stage_report_sha256": _file_sha256(report_path),
        "parent_checkpoint_sha256": parent_sha256,
        "data_identity_sha256": data_sha256,
        "base_revision": (
            config["base_model"]["revision"]
            if expected_stage == "volume_structure_ssl" else None
        ),
        "base_weights_sha256": (
            config["base_model"]["weights_sha256"]
            if expected_stage == "volume_structure_ssl" else None
        ),
        "base_config_sha256": (
            config["base_model"]["config_sha256"]
            if expected_stage == "volume_structure_ssl" else None
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_identity(
    checkpoint: str,
    base_snapshot: Path,
) -> dict:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    base_snapshot = Path(base_snapshot).expanduser().resolve()
    adapter_path = checkpoint_path / "adapter_config.json"
    weights_path = base_snapshot / "model.safetensors"
    config_path = base_snapshot / "config.json"
    if (
        not checkpoint_path.is_dir()
        or not adapter_path.is_file()
        or not base_snapshot.is_dir()
        or _HEX40.fullmatch(base_snapshot.name) is None
        or not weights_path.is_file()
        or not config_path.is_file()
    ):
        raise SystemExit(
            "local Chronos-2 Atlas runs require a complete pinned base "
            "snapshot named by its 40-character revision"
        )
    try:
        adapter = json.loads(adapter_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("Chronos-2 adapter config is unreadable") from exc
    if (
        not isinstance(adapter, dict)
        or adapter.get("base_model_name_or_path") != CHRONOS2_MODEL_ID
        or adapter.get("peft_type") != "LORA"
        or adapter.get("inference_mode") is not True
        or adapter.get("revision") not in {None, base_snapshot.name}
    ):
        raise SystemExit(
            "Chronos-2 adapter does not match the pinned base snapshot")
    return {
        "base_snapshot_path": str(base_snapshot),
        "base_revision": base_snapshot.name,
        "base_weights_sha256": _file_sha256(weights_path),
        "base_config_sha256": _file_sha256(config_path),
    }


def _validate_stage_base_identity(
    stage_identity: dict,
    base_identity: dict,
) -> None:
    """Bind a Volume-SSL stage report to the supplied pinned base snapshot."""
    if stage_identity["base_revision"] is not None and (
        stage_identity["base_revision"] != base_identity["base_revision"]
        or stage_identity["base_weights_sha256"]
        != base_identity["base_weights_sha256"]
        or stage_identity["base_config_sha256"]
        != base_identity["base_config_sha256"]
    ):
        raise SystemExit(
            "Chronos-2 stage report base identity does not match "
            "--base-snapshot"
        )


@contextmanager
def _runtime_checkpoint(
    checkpoint: str,
    *,
    base_revision: str,
    base_snapshot: Path,
):
    """Load PEFT against the exact hashed base snapshot without mutation."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    base_snapshot = Path(base_snapshot).expanduser().resolve()
    if (
        base_snapshot.name != base_revision
        or not (base_snapshot / "config.json").is_file()
        or not (base_snapshot / "model.safetensors").is_file()
    ):
        raise SystemExit(
            "Chronos-2 runtime base snapshot is incomplete or revision-mismatched")
    adapter = json.loads(
        (checkpoint_path / "adapter_config.json").read_text())
    if (
        adapter.get("base_model_name_or_path") != CHRONOS2_MODEL_ID
        or adapter.get("revision") not in {None, base_revision}
    ):
        raise SystemExit(
            "Chronos-2 adapter does not match the authenticated base snapshot")
    adapter["base_model_name_or_path"] = str(base_snapshot)
    adapter["revision"] = None
    with tempfile.TemporaryDirectory(
        prefix="ffm-chronos2-atlas-adapter-",
    ) as temporary:
        runtime = Path(temporary)
        (runtime / "adapter_config.json").write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n")
        (runtime / "adapter_model.safetensors").symlink_to(
            checkpoint_path / "adapter_model.safetensors")
        yield runtime


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
    value.add_argument(
        "--base-snapshot",
        type=Path,
        required=True,
        help="pinned Chronos2-small Hugging Face snapshot directory",
    )
    value.add_argument(
        "--stage-report",
        type=Path,
        help=(
            "completed public stage report; inferred beside a local "
            "checkpoint when omitted"
        ),
    )
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
    base_identity = _base_identity(
        args.checkpoint, args.base_snapshot)
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
    if args.device == "mps" and args.batch_series > MPS_SAFE_MAX_BATCH_SERIES:
        raise SystemExit(
            f"--batch-series exceeds the verified M1-safe limit "
            f"({MPS_SAFE_MAX_BATCH_SERIES}); use 320 for 64 OHLCV windows"
        )
    if args.device == "mps" and args.chunk_windows > MPS_SAFE_MAX_CHUNK_WINDOWS:
        raise SystemExit(
            f"--chunk-windows exceeds the verified M1-safe limit "
            f"({MPS_SAFE_MAX_CHUNK_WINDOWS})"
        )
    if args.batch_series % 5:
        raise SystemExit(
            "--batch-series must be divisible by 5 because every window has "
            "five OHLCV variates"
        )
    stage_identity = _stage_identity(
        args.checkpoint,
        checkpoint_hash,
        args.stage_report,
        context_length=args.window,
    )
    _validate_stage_base_identity(stage_identity, base_identity)
    if not labels.is_file():
        _generate_labels(labels, data_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.name)
    stem = f"{safe_name}_{args.control}"
    result_path = out_dir / f"{stem}.json"
    cache_path = out_dir / f"{stem}_emb.npy"
    environment = {
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
    }
    for field in (
        "stage_report_sha256",
        "parent_checkpoint_sha256",
        "data_identity_sha256",
    ):
        value = stage_identity[field]
        if value is not None:
            environment[f"ATLAS_{field.upper()}"] = value
    for field in (
        "base_revision",
        "base_weights_sha256",
        "base_config_sha256",
    ):
        environment[f"ATLAS_{field.upper()}"] = base_identity[field]
    os.environ.update(environment)
    with _runtime_checkpoint(
        args.checkpoint,
        base_revision=base_identity["base_revision"],
        base_snapshot=args.base_snapshot,
    ) as runtime_checkpoint:
        os.environ["ATLAS_LOAD_CHECKPOINT_PATH"] = str(
            runtime_checkpoint)
        atlas = _load_atlas()
        if (
            "pred_persistent_trend_start"
            not in atlas.FORBIDDEN_NONCAUSAL_PROBES
        ):
            raise RuntimeError(
                "shared Probe Atlas is missing the noncausal-target guard"
            )
        if args.preflight_only:
            bars, keys, fields = atlas._load_pool()
            payload = {
                "schema": "ffm_chronos2_probe_atlas_preflight_v1",
                "checkpoint_sha256": checkpoint_hash,
                **stage_identity,
                **base_identity,
                "pool_rows": len(keys),
                "pool_sha256": atlas._pool_sha256(keys),
                "source_sha256": atlas._source_sha256(bars),
                "window": args.window,
                "horizons": list(horizons),
                "streams": sorted(set(fields["stream"])),
            }
            destination = out_dir / f"{stem}_preflight.json"
            destination.write_text(
                json.dumps(payload, indent=2) + "\n")
            print(
                f"[chronos2-atlas] PREFLIGHT PASS "
                f"rows={len(keys):,} -> {destination}",
                flush=True,
            )
            return payload
        return atlas.main()


if __name__ == "__main__":
    main()
