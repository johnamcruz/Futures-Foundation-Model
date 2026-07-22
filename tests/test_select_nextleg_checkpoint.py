import hashlib
import json
from pathlib import Path

from scripts import select_nextleg_checkpoint as selection


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_run(root: Path, name: str, *, uniform: bool, lift: float = 0.0,
              controls: bool = True, task=(0.03, 0.08, 0.03),
              refinement_parent: str | None = None):
    directory = root / name
    checkpoint = directory / "mantis_ssl_nextleg.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(name.encode())
    digest = hashlib.sha256(name.encode()).hexdigest()
    probes = {}
    for index, probe in enumerate(sorted(selection.EXPECTED_PROBES)):
        family = "retention" if probe.startswith("ret_") else "prediction"
        delta = lift if family == "prediction" else 0.0
        probes[probe] = {
            "family": family, "auc": 0.60 + index / 1000 + delta,
            "n_eval": 1000, "pos_rate": 0.4,
            "per_stream_auc": {"NQ@3min": 0.58 + delta},
            "worst_stream_auc": 0.58 + delta,
        }
    task_controls = ({
        "shuffle": {"forecast_skill": .01, "leg_corr1": .01, "leg_corr2": .005},
        "random": {"forecast_skill": .008, "leg_corr1": .0, "leg_corr2": -.01},
    } if controls else {})
    forecast_skill, leg_corr1, leg_corr2 = task
    task_control = {
        "contract": "nextleg_forecast_and_leg_skill_v1",
        "real": {"forecast_skill": forecast_skill, "leg_corr1": leg_corr1,
                 "leg_corr2": leg_corr2},
        "controls": task_controls,
        "beats_controls": bool(controls),
    }
    _write(Path(str(checkpoint) + ".real_complete.json"), {"checkpoint_sha256": digest})
    _write(Path(str(checkpoint) + ".report.json"), {
        "config": {"sampling_mode": "uniform_stream" if uniform else "bar_proportional"},
        "verdict": {"all_pass": True, "representation_pass": True,
                    "beats_controls": bool(controls), "real_delta": 0.05,
                    "control_delta": {"shuffle": .051, "random": .052},
                    "task_control": task_control,
                    "temporal_signal": 0.04 if controls else None},
        "task_control": task_control,
        "history": [{"best_epoch": 4, "forecast_skill": forecast_skill,
                     "leg_corr1": leg_corr1, "leg_corr2": leg_corr2, "std": 1.0}],
    })
    parent_row = {"sha256": refinement_parent or "parent"}
    if refinement_parent:
        parent_row["source_stage"] = "nextleg"
    _write(directory / "pipeline_manifest.json", {
        "sampling_mode": "uniform_stream" if uniform else "bar_proportional",
        "data_provenance": {"streams": "same"},
        "stages": {"seq2seq": parent_row,
                   "nextleg": {"sha256": digest}},
    })
    _write(directory / "probe_atlas" / "nextleg.json", {
        "checkpoint_sha256": digest, "probes": probes,
    })
    _write(directory / "probe_atlas" / "nextleg_emb.npy.pool.json", {
        "schema": "ffm_probe_atlas_pool_v1", "pool_sha256": "same", "rows": 9000,
    })
    return directory


def test_uniform_candidate_is_promoted_only_with_controls_and_predictive_lift(tmp_path):
    baseline = _make_run(tmp_path, "baseline", uniform=False)
    candidate = _make_run(tmp_path, "candidate", uniform=True, lift=0.006)
    result = selection.select(baseline, candidate)
    assert result["status"] == "PROMOTE_CANDIDATE"
    assert result["winner"].endswith("candidate/mantis_ssl_nextleg.pt")
    assert result["metrics"]["prediction_mean_auc"]["delta"] > 0.005


def test_candidate_without_corrupted_input_controls_is_rejected(tmp_path):
    baseline = _make_run(tmp_path, "baseline", uniform=False)
    candidate = _make_run(tmp_path, "candidate", uniform=True, lift=0.006, controls=False)
    result = selection.select(baseline, candidate)
    assert result["status"] == "REJECT_CANDIDATE"
    assert any("controls" in reason for reason in result["failures"])


def test_candidate_with_different_probe_pool_is_rejected(tmp_path):
    baseline = _make_run(tmp_path, "baseline", uniform=False)
    candidate = _make_run(tmp_path, "candidate", uniform=True, lift=0.006)
    pool = candidate / "probe_atlas" / "nextleg_emb.npy.pool.json"
    _write(pool, {"schema": "ffm_probe_atlas_pool_v1", "pool_sha256": "different"})
    result = selection.select(baseline, candidate)
    assert result["status"] == "REJECT_CANDIDATE"
    assert "Probe Atlas pool identity differs" in result["failures"]


def test_uniform_refinement_can_promote_when_it_preserves_leg1_and_improves_leg2(tmp_path):
    baseline = _make_run(tmp_path, "baseline", uniform=False,
                         task=(0.030, 0.080, 0.030))
    digest = hashlib.sha256(b"baseline").hexdigest()
    candidate = _make_run(tmp_path, "candidate", uniform=True,
                          task=(0.031, 0.078, 0.041), refinement_parent=digest)
    result = selection.select(baseline, candidate)
    assert result["status"] == "PROMOTE_CANDIDATE"
    assert result["lineage_mode"] == "nextleg_uniform_refinement"
    assert result["metrics"]["leg_corr1"]["delta"] >= -0.005
    assert result["metrics"]["leg_corr2"]["delta"] >= 0.005


def test_uniform_refinement_rejects_second_leg_gain_that_sacrifices_leg1(tmp_path):
    baseline = _make_run(tmp_path, "baseline", uniform=False,
                         task=(0.030, 0.080, 0.030))
    digest = hashlib.sha256(b"baseline").hexdigest()
    candidate = _make_run(tmp_path, "candidate", uniform=True,
                          task=(0.031, 0.060, 0.045), refinement_parent=digest)
    result = selection.select(baseline, candidate)
    assert result["status"] == "REJECT_CANDIDATE"
    assert "refinement materially regressed first-leg correlation" in result["failures"]
