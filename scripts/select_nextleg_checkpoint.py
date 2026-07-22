#!/usr/bin/env python3
"""Select a uniform-stream NextLeg candidate against the prior clean checkpoint.

Both runs must use the same sealed 9x4 data and Probe Atlas examples. The candidate
must pass its own objective gate, beat shuffle and random controls, learn both leg
targets, and avoid material capability regressions. Promotion additionally requires
an observable improvement in prediction or worst-stream generalization; merely tying
the baseline does not justify replacing the frozen encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_PROBES = {
    "ret_vol_regime", "ret_squeeze", "ret_vol_surge", "ret_day_position",
    "ret_ny_session", "pred_fwd_direction", "pred_fwd_large_move",
    "pred_vol_expand", "pred_persistent_trend_start",
}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_artifact(directory: Path, filename: str) -> Path:
    """Resolve canonical Probe Atlas output, with the old downloaded-root layout supported."""
    canonical = directory / "probe_atlas" / filename
    legacy = directory / filename
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(canonical)


def _task_history(directory: Path, report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return complete NextLeg metrics, recovering older reports from their trainer sidecar.

    Early clean-pipeline reports persisted forecast skill but omitted leg correlations even though
    the exact selected values were present in ``.trainer.pt``. Treating missing values as zero can
    reverse a checkpoint-selection decision. Missing evidence now fails closed unless the trusted
    local trainer sidecar supplies it, and overlapping report/sidecar metrics must agree.
    """
    history = dict((report.get("history") or [{}])[0])
    required = ("forecast_skill", "leg_corr1", "leg_corr2")
    if all(history.get(key) is not None for key in required):
        return history, "report"

    trainer_path = directory / "mantis_ssl_nextleg.pt.trainer.pt"
    if not trainer_path.is_file():
        missing = [key for key in required if history.get(key) is None]
        raise RuntimeError(
            f"NextLeg comparison is missing {missing} and trainer sidecar: {trainer_path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - selector runs in the foundation environment
        raise RuntimeError("torch is required to recover legacy trainer-sidecar metrics") from exc
    payload = torch.load(trainer_path, map_location="cpu")
    sidecar = dict(payload.get("best_history_row") or {})
    if sidecar.get("forecast_skill") is None and sidecar.get("skill") is not None:
        sidecar["forecast_skill"] = sidecar["skill"]
    for key in required:
        report_value, sidecar_value = history.get(key), sidecar.get(key)
        if sidecar_value is None:
            raise RuntimeError(f"trainer sidecar is missing selected metric {key}: {trainer_path}")
        if (report_value is not None
                and not math.isclose(float(report_value), float(sidecar_value),
                                     rel_tol=1e-7, abs_tol=1e-9)):
            raise RuntimeError(
                f"report/trainer selected metric mismatch for {key} in {directory}")
        history.setdefault(key, sidecar_value)
    history.setdefault("best_epoch", sidecar.get("epoch"))
    return history, "report+trainer_sidecar"


def _metric(history: dict[str, Any], key: str, directory: Path) -> float:
    value = history.get(key)
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError(f"missing/non-finite {key} in selected NextLeg history: {directory}")
    return float(value)


def _artifacts(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    checkpoint = directory / "mantis_ssl_nextleg.pt"
    report = _read(Path(str(checkpoint) + ".report.json"))
    marker = _read(Path(str(checkpoint) + ".real_complete.json"))
    manifest = _read(directory / "pipeline_manifest.json")
    atlas = _read(_probe_artifact(directory, "nextleg.json"))
    pool = _read(_probe_artifact(directory, "nextleg_emb.npy.pool.json"))
    digest = _sha256(checkpoint)
    if marker.get("checkpoint_sha256") != digest:
        raise RuntimeError(f"REAL-complete marker hash mismatch in {directory}")
    if atlas.get("checkpoint_sha256") != digest:
        raise RuntimeError(f"Probe Atlas checkpoint hash mismatch in {directory}")
    stage = (manifest.get("stages") or {}).get("nextleg") or {}
    if stage.get("sha256") != digest:
        raise RuntimeError(f"pipeline manifest checkpoint hash mismatch in {directory}")
    probes = atlas.get("probes") or {}
    if set(probes) != EXPECTED_PROBES:
        raise RuntimeError(
            f"Probe Atlas contract mismatch in {directory}: {sorted(set(probes) ^ EXPECTED_PROBES)}")
    task_history, task_history_source = _task_history(directory, report)
    return {"directory": directory, "checkpoint": checkpoint, "sha256": digest,
            "report": report, "manifest": manifest, "atlas": atlas, "pool": pool,
            "task_history": task_history, "task_history_source": task_history_source}


def _family_mean(probes: dict[str, Any], family: str, field: str = "auc") -> float:
    values = [float(row[field]) for row in probes.values()
              if row.get("family") == family and row.get(field) is not None]
    if not values:
        raise RuntimeError(f"no {family} {field} values in Probe Atlas")
    return sum(values) / len(values)


def _worst_stream_mean(probes: dict[str, Any]) -> float:
    values = [float(row["worst_stream_auc"]) for row in probes.values()
              if row.get("worst_stream_auc") is not None]
    if len(values) != len(probes):
        raise RuntimeError("Probe Atlas is missing worst-stream AUC")
    return sum(values) / len(values)


def select(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline, candidate = _artifacts(baseline_dir), _artifacts(candidate_dir)
    failures: list[str] = []

    if baseline["pool"] != candidate["pool"]:
        failures.append("Probe Atlas pool identity differs")
    if baseline["manifest"].get("data_provenance") != candidate["manifest"].get("data_provenance"):
        failures.append("sealed data provenance differs")
    base_parent_row = ((baseline["manifest"].get("stages") or {}).get("seq2seq") or {})
    cand_parent_row = ((candidate["manifest"].get("stages") or {}).get("seq2seq") or {})
    base_parent = base_parent_row.get("sha256")
    cand_parent = cand_parent_row.get("sha256")
    parent_source_stage = cand_parent_row.get("source_stage", "seq2seq")
    refinement = parent_source_stage == "nextleg"
    expected_parent = baseline["sha256"] if refinement else base_parent
    if not expected_parent or cand_parent != expected_parent:
        failures.append(
            "candidate does not use the exact baseline "
            + ("NextLeg refinement parent" if refinement else "Seq2Seq parent"))

    report = candidate["report"]
    verdict = report.get("verdict") or {}
    config = report.get("config") or {}
    task_control = verdict.get("task_control") or report.get("task_control") or {}
    controls = task_control.get("controls") or {}
    if candidate["manifest"].get("sampling_mode") != "uniform_stream":
        failures.append("candidate pipeline is not uniform_stream")
    if config.get("sampling_mode") != "uniform_stream":
        failures.append("candidate report is not uniform_stream")
    if verdict.get("all_pass") is not True or verdict.get("representation_pass") is not True:
        failures.append("candidate objective/representation gate failed")
    if task_control.get("contract") != "nextleg_forecast_and_leg_skill_v1":
        failures.append("candidate lacks the NextLeg-specific control contract")
    if set(controls) != {"shuffle", "random"}:
        failures.append("candidate lacks exact shuffle and random task controls")
    if task_control.get("beats_controls") is not True or verdict.get("beats_controls") is not True:
        failures.append("REAL NextLeg task metrics did not beat every corrupted-input control")
    if verdict.get("temporal_signal") is None or float(verdict["temporal_signal"]) <= 0:
        failures.append("candidate has no positive temporal-order signal")

    history = candidate["task_history"]
    baseline_history = baseline["task_history"]
    if _metric(history, "forecast_skill", candidate["directory"]) <= 0:
        failures.append("candidate does not beat candle persistence")
    if (_metric(history, "leg_corr1", candidate["directory"]) <= 0
            or _metric(history, "leg_corr2", candidate["directory"]) <= 0):
        failures.append("candidate did not learn both future leg-duration targets")

    bp, cp = baseline["atlas"]["probes"], candidate["atlas"]["probes"]
    for name in EXPECTED_PROBES:
        for field in ("n_eval", "pos_rate"):
            if bp[name].get(field) != cp[name].get(field):
                failures.append(f"{name} uses different {field}")
        if set(bp[name].get("per_stream_auc") or {}) != set(cp[name].get("per_stream_auc") or {}):
            failures.append(f"{name} uses different stream coverage")

    metrics = {
        "retention_mean_auc": {
            "baseline": _family_mean(bp, "retention"),
            "candidate": _family_mean(cp, "retention"),
        },
        "prediction_mean_auc": {
            "baseline": _family_mean(bp, "prediction"),
            "candidate": _family_mean(cp, "prediction"),
        },
        "persistent_trend_start_auc": {
            "baseline": float(bp["pred_persistent_trend_start"]["auc"]),
            "candidate": float(cp["pred_persistent_trend_start"]["auc"]),
        },
        "worst_stream_mean_auc": {
            "baseline": _worst_stream_mean(bp),
            "candidate": _worst_stream_mean(cp),
        },
        "forecast_skill": {
            "baseline": _metric(baseline_history, "forecast_skill", baseline["directory"]),
            "candidate": _metric(history, "forecast_skill", candidate["directory"]),
        },
        "leg_corr1": {
            "baseline": _metric(baseline_history, "leg_corr1", baseline["directory"]),
            "candidate": _metric(history, "leg_corr1", candidate["directory"]),
        },
        "leg_corr2": {
            "baseline": _metric(baseline_history, "leg_corr2", baseline["directory"]),
            "candidate": _metric(history, "leg_corr2", candidate["directory"]),
        },
    }
    for row in metrics.values():
        row["delta"] = row["candidate"] - row["baseline"]

    if metrics["retention_mean_auc"]["delta"] < -0.005:
        failures.append("retention mean regressed by more than 0.005 AUC")
    if metrics["prediction_mean_auc"]["delta"] < -0.005:
        failures.append("prediction mean regressed by more than 0.005 AUC")
    if metrics["persistent_trend_start_auc"]["delta"] < -0.005:
        failures.append("persistent trend-start AUC regressed by more than 0.005")
    if metrics["worst_stream_mean_auc"]["delta"] < -0.010:
        failures.append("worst-stream mean regressed by more than 0.010 AUC")

    atlas_improves = bool(
        metrics["prediction_mean_auc"]["delta"] >= 0.001
        or metrics["persistent_trend_start_auc"]["delta"] >= 0.005
        or metrics["worst_stream_mean_auc"]["delta"] >= 0.005
    )
    refinement_improves = bool(
        refinement
        and metrics["leg_corr1"]["delta"] >= -0.005
        and metrics["leg_corr2"]["delta"] >= 0.005
        and metrics["forecast_skill"]["delta"] >= -0.002
    )
    if refinement and metrics["leg_corr1"]["delta"] < -0.005:
        failures.append("refinement materially regressed first-leg correlation")
    if refinement and metrics["leg_corr2"]["delta"] < 0.005:
        failures.append("refinement did not materially improve second-leg correlation")
    if refinement and metrics["forecast_skill"]["delta"] < -0.002:
        failures.append("refinement materially regressed forecast skill")
    improves = atlas_improves or refinement_improves
    if not improves:
        failures.append("uniform candidate has no material predictive/generalization improvement")

    return {
        "schema": "ffm_nextleg_checkpoint_selection_v1",
        "lineage_mode": "nextleg_uniform_refinement" if refinement else "seq2seq_uniform",
        "status": "PROMOTE_CANDIDATE" if not failures else "REJECT_CANDIDATE",
        "winner": str(candidate["checkpoint"]) if not failures else None,
        "baseline_sha256": baseline["sha256"],
        "candidate_sha256": candidate["sha256"],
        "seq2seq_parent_sha256": base_parent,
        "candidate_parent_sha256": cand_parent,
        "metric_sources": {
            "baseline": baseline["task_history_source"],
            "candidate": candidate["task_history_source"],
        },
        "metrics": metrics,
        "candidate_controls": {
            "task_control": task_control,
            "representation_real_delta": verdict.get("real_delta"),
            "representation_control_delta": verdict.get("control_delta"),
            "temporal_signal": verdict.get("temporal_signal"),
        },
        "candidate_task_metrics": {
            key: history.get(key) for key in
            ("best_epoch", "forecast_skill", "leg_corr1", "leg_corr2",
             "leg_bias1", "leg_bias2", "std")
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.baseline_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PROMOTE_CANDIDATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
