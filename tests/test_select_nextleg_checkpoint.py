import hashlib
import json
from pathlib import Path

from scripts import select_nextleg_checkpoint as selection


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_run(root: Path, name: str, *, uniform: bool, lift: float = 0.0,
              controls: bool = True):
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
    control_delta = {"shuffle": 0.01, "random": 0.0} if controls else {}
    _write(Path(str(checkpoint) + ".real_complete.json"), {"checkpoint_sha256": digest})
    _write(Path(str(checkpoint) + ".report.json"), {
        "config": {"sampling_mode": "uniform_stream" if uniform else "bar_proportional"},
        "verdict": {"all_pass": True, "representation_pass": True,
                    "real_delta": 0.05, "control_delta": control_delta,
                    "temporal_signal": 0.04 if controls else None},
        "history": [{"best_epoch": 4, "forecast_skill": 0.03,
                     "leg_corr1": 0.08, "leg_corr2": 0.03, "std": 1.0}],
    })
    _write(directory / "pipeline_manifest.json", {
        "sampling_mode": "uniform_stream" if uniform else "bar_proportional",
        "data_provenance": {"streams": "same"},
        "stages": {"seq2seq": {"sha256": "parent"},
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
