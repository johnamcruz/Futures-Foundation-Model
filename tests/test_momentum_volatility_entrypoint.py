import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mantis_ssl_momentum_volatility.py"
SPEC = importlib.util.spec_from_file_location("mantis_ssl_momentum_volatility", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _atlas_payload(predictive=.70, retention=.80, nq3=None):
    nq3 = predictive if nq3 is None else nq3
    return {
        "probes": {
            name: {
                "auc": predictive,
                "per_stream_auc": {"NQ@3min": nq3},
            }
            for name in (
                "pred_mv_trend_expansion",
                "pred_mv_trend_weakening",
                "pred_mv_noisy_expansion",
                "pred_mv_compression",
            )
        } | {
            name: {
                "auc": retention,
                "per_stream_auc": {"NQ@3min": retention},
            }
            for name in ("ret_vol_regime", "ret_squeeze", "ret_vol_surge")
        },
    }


def test_entrypoint_is_public_strategy_agnostic_and_calls_canonical_task():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert 'pretext="momentum_volatility"' in source
    assert 'holdout_start=HOLDOUT_START' in source
    assert 'backbone_ckpt=str(parent)' in source
    assert "shuffle,random" in source
    assert "--resume" in source
    assert "--lora-r" in source
    assert "uses_strategy_labels" in source
    assert "pivot" not in source.lower()
    assert "ffm-strategies" not in source.lower()
    assert all(
        not (isinstance(node, ast.Import) and
             any(alias.name == "torch" for alias in node.names))
        for node in tree.body)


def test_entrypoint_defaults_to_race_v2_parent_and_full_stream_matrix():
    source = SCRIPT.read_text()
    assert "mantis_ssl_nextleg_race_v2.pt" in source
    assert "mantis_ssl_mv_v3.pt" in source
    assert 'TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")' in source
    assert 'TIMEFRAMES = ("1min", "3min", "5min", "15min")' in source
    assert 'HOLDOUT_START = "2026-01-01"' in source
    assert 'default=os.environ.get("SAMPLING_MODE", "uniform_stream")' in source


def test_entrypoint_seals_data_and_runs_probe_atlas():
    source = SCRIPT.read_text()
    assert "seal_continuous_streams" in source
    assert "export_mv_readout" not in source
    assert "task heads are discarded" in source
    assert "probe_atlas.py" in source
    assert "checkpoint_sha256" in source
    assert ".data_provenance.json" in source
    assert "probe_baseline_ckpt=str(parent)" in source
    assert "encoder_transfer" in source


def test_encoder_transfer_gate_is_parent_relative_and_headless(tmp_path):
    parent, child = tmp_path / "parent.pt", tmp_path / "child.pt"
    parent_atlas = tmp_path / "parent_atlas.json"
    child_atlas = tmp_path / "child_atlas.json"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    parent_atlas.write_text(json.dumps(_atlas_payload(.70, .80, .70)))
    child_atlas.write_text(json.dumps(_atlas_payload(.71, .80, .71)))
    Path(str(child) + ".report.json").write_text(json.dumps({
        "probe": {
            "comparison_baseline": str(parent),
            "mean_core_delta": .01,
            "forward_score": .02,
        },
    }))

    result = launcher._assert_encoder_transfer(
        output=child, parent=parent, margin=.002,
        retention_tolerance=.005,
        child_atlas=child_atlas, parent_atlas=parent_atlas)
    payload = json.loads(result.read_text())
    assert payload["heads_available_to_gate"] is False
    assert payload["parent_relative_probe"]["core_delta"] == .01


def test_encoder_transfer_gate_rejects_material_parent_regression(tmp_path):
    parent, child = tmp_path / "parent.pt", tmp_path / "child.pt"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    Path(str(child) + ".report.json").write_text(json.dumps({
        "probe": {
            "comparison_baseline": str(parent),
            "mean_core_delta": -.006,
            "forward_score": .02,
        },
        "task_control": {"real": {"mv_transition_auc": .99}},
    }))

    with pytest.raises(RuntimeError, match="encoder transfer failed"):
        launcher._assert_encoder_transfer(
            output=child, parent=parent, margin=.002,
            retention_tolerance=.005)


def test_encoder_transfer_gate_rejects_missing_or_flat_mv_atlas(tmp_path):
    parent, child = tmp_path / "parent.pt", tmp_path / "child.pt"
    parent_atlas = tmp_path / "parent_atlas.json"
    child_atlas = tmp_path / "child_atlas.json"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    Path(str(child) + ".report.json").write_text(json.dumps({
        "probe": {
            "comparison_baseline": str(parent),
            "mean_core_delta": 0.0,
            "forward_score": 0.0,
        },
    }))

    with pytest.raises(RuntimeError, match="Probe Atlas comparison is required"):
        launcher._assert_encoder_transfer(
            output=child, parent=parent, margin=.002,
            retention_tolerance=.005)

    parent_atlas.write_text(json.dumps(_atlas_payload(.70, .80, .70)))
    child_atlas.write_text(json.dumps(_atlas_payload(.70, .80, .70)))
    with pytest.raises(RuntimeError, match="did not add transferable MV context"):
        launcher._assert_encoder_transfer(
            output=child, parent=parent, margin=.002,
            retention_tolerance=.005,
            child_atlas=child_atlas, parent_atlas=parent_atlas)
