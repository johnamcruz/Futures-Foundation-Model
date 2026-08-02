"""Public Probe Atlas and lifecycle infrastructure must stay strategy-independent."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_atlas_is_a_standalone_public_implementation():
    source = (ROOT / "scripts" / "probe_atlas.py").read_text()
    assert 'ROOT / "colabs"' not in source
    assert "import pivot_trend_mantis" not in source
    assert "def _load_pool" in source
    assert "def main" in source
    assert "FIXED_TARGETS" not in source
    assert "STOP_BUFFER" not in source
    assert "pred_reach_" not in source
    assert 'ATLAS_SCHEMA = "ffm_probe_atlas_v5"' in source
    assert source.count('"pred_persistent_trend_start"') == 1
    assert "FORBIDDEN_NONCAUSAL_PROBES" in source
    assert 'probes[f"pred_trend_h{horizon}"]' in source


def test_probe_atlas_rejects_removed_noncausal_probe_before_fit():
    atlas = _load(
        "public_probe_atlas_forbidden_probe",
        ROOT / "scripts" / "probe_atlas.py",
    )

    with pytest.raises(RuntimeError, match="not bounded"):
        atlas._assert_probe_is_causal("pred_persistent_trend_start")

    atlas._assert_probe_is_causal("pred_trend_h20")


def test_clean_pipeline_has_no_colabs_runtime_dependency():
    source = (ROOT / "scripts" / "mantis" / "mantis_ssl_clean_pipeline.py").read_text()
    assert 'ROOT / "colabs"' not in source
    assert 'ROOT / "scripts" / "generate_trend_labels.py"' in source


def test_atlas_sampling_is_deterministic_and_preserves_time_coverage():
    atlas = _load("public_probe_atlas", ROOT / "scripts" / "probe_atlas.py")
    rows = np.arange(100)
    selected = atlas._even_sample(rows, 10)
    assert np.array_equal(selected, atlas._even_sample(rows, 10))
    assert len(selected) == 10
    assert selected[0] == 0 and selected[-1] == 99


def test_atlas_mv_targets_use_the_exact_future_window_and_causal_scale():
    atlas = _load("public_probe_atlas_mv", ROOT / "scripts" / "probe_atlas.py")
    rows = 100
    close = np.arange(rows, dtype=float)
    width = np.full(rows, 2.0)
    high, low = close + width / 2, close - width / 2

    strength, expansion, state = atlas._momentum_volatility_fields(
        high, low, close)
    assert strength[63] == 1.0
    assert expansion[63] == 1.0
    assert state[63] == 1

    changed_high, changed_low = high.copy(), low.copy()
    changed_high[84:] += 1000
    changed_low[84:] -= 1000
    changed = atlas._momentum_volatility_fields(
        changed_high, changed_low, close)
    assert changed[1][63] == expansion[63]
    assert changed[2][63] == state[63]


def test_atlas_multihorizon_targets_use_requested_completed_future_bars():
    atlas = _load("public_probe_atlas_horizons", ROOT / "scripts" / "probe_atlas.py")
    rows = 100
    close = np.arange(rows, dtype=float)
    width = np.full(rows, 2.0)
    high, low = close + width / 2, close - width / 2

    strength, expansion, state = atlas._momentum_volatility_fields(
        high, low, close, horizon=5)

    assert strength[63] == 1.0
    assert expansion[63] == 1.0
    assert state[63] == 1
    changed_high, changed_low = high.copy(), low.copy()
    changed_high[69:] += 1000
    changed_low[69:] -= 1000
    changed = atlas._momentum_volatility_fields(
        changed_high, changed_low, close, horizon=5)
    assert changed[1][63] == expansion[63]


def test_atlas_direction_target_is_exact_horizon_and_target_side_only():
    atlas = _load("public_probe_atlas_direction", ROOT / "scripts" / "probe_atlas.py")
    close = np.asarray([10, 11, 9, 12, 8, 13, 7, 14], dtype=float)
    direction = atlas._future_direction(close, horizon=3)

    np.testing.assert_array_equal(
        direction[:5],
        np.asarray([1, 0, 1, 0, 1], dtype=np.float32),
    )
    assert np.isnan(direction[-3:]).all()

    changed = close.copy()
    changed[4:] = -1_000
    changed_direction = atlas._future_direction(changed, horizon=3)
    # Row 0 uses exactly close[3], so bars after its target cannot change it.
    assert changed_direction[0] == direction[0]


def test_atlas_input_controls_are_deterministic_and_input_only(monkeypatch):
    monkeypatch.setenv("ATLAS_CONTROL", "shuffle")
    atlas = _load("public_probe_atlas_control", ROOT / "scripts" / "probe_atlas.py")
    window = np.arange(40, dtype=np.float32).reshape(5, 8)
    key = ("NQ", "3min", 42)
    first = atlas._controlled_window(window, key)
    second = atlas._controlled_window(window, key)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, window)
    for channel in range(5):
        np.testing.assert_array_equal(
            np.sort(first[channel]), np.sort(window[channel]))
    # One shared permutation preserves OHLCV alignment across each shuffled bar.
    np.testing.assert_array_equal(first[1] - first[0], np.full(8, 8))


def test_chronos2_probe_atlas_entrypoint_ports_public_atlas_contract(tmp_path):
    launcher = _load(
        "chronos2_probe_atlas",
        ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py",
    )
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "autogluon/chronos-2-small",
        "peft_type": "LORA",
        "inference_mode": True,
        "revision": None,
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    base = tmp_path / ("a" * 40)
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"base")

    assert launcher._checkpoint_identity(str(adapter)) == launcher._tree_sha256(adapter)
    defaults = launcher.parser().parse_args([
        "--base-snapshot", str(base),
    ])
    assert defaults.window == 256
    assert defaults.horizons == "5,10,20,50"
    assert defaults.batch_series == 320
    assert defaults.control == "real"
    assert (
        defaults.checkpoint
        == str(
            ROOT
            / "temp"
            / "chronos2_small_36stream"
            / "contrastive_kaufman_40"
            / "checkpoint"
        )
    )
    assert launcher.MPS_SAFE_MAX_BATCH_SERIES == 320
    assert launcher.MPS_SAFE_MAX_CHUNK_WINDOWS == 1024
    base_identity = launcher._base_identity(str(adapter), base)
    assert base_identity["base_revision"] == "a" * 40
    assert len(base_identity["base_weights_sha256"]) == 64
    with launcher._runtime_checkpoint(
        str(adapter),
        base_revision=base_identity["base_revision"],
    ) as runtime:
        runtime_adapter = json.loads(
            (runtime / "adapter_config.json").read_text())
        assert runtime_adapter["revision"] == "a" * 40
        assert launcher._tree_sha256(adapter) == (
            launcher._checkpoint_identity(str(adapter)))
    source = Path(launcher.__file__).read_text()
    assert 'ATLAS_BACKBONE": "chronos2"' in source
    assert 'ROOT / "scripts" / "probe_atlas.py"' in source
    atlas_source = (ROOT / "scripts" / "probe_atlas.py").read_text()
    assert 'pred_direction_h{horizon}' in atlas_source
    assert 'pred_trend_direction_h{horizon}' in atlas_source
    assert '"ret_structural_direction"' in atlas_source


@pytest.mark.parametrize(("schema", "stage"), [
    ("ffm_chronos2_mask_v1", "mask"),
    ("ffm_chronos2_volume_structure_ssl_v2", "volume_structure_ssl"),
])
def test_chronos2_atlas_authenticates_completed_stage_lineage(
        tmp_path, schema, stage):
    launcher = _load(
        "chronos2_probe_atlas_stage",
        ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py",
    )
    run = tmp_path / stage
    adapter = run / "checkpoint"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    checkpoint_sha256 = launcher._tree_sha256(adapter)
    report = run / "report.json"
    report.write_text(json.dumps({
        "schema": schema,
        "stage": stage,
        "status": "complete",
        "checkpoint": {"sha256": checkpoint_sha256},
        "parent": {"sha256": "a" * 64},
        "data_identity_sha256": "b" * 64,
        "config": {
            "context_length": 256,
            "timeframes": ["1min", "3min", "5min", "15min"],
        },
    }))

    identity = launcher._stage_identity(
        str(adapter),
        checkpoint_sha256,
        None,
        context_length=256,
    )

    assert identity == {
        "stage_report_path": str(report.resolve()),
        "stage_report_sha256": launcher._file_sha256(report),
        "parent_checkpoint_sha256": "a" * 64,
        "data_identity_sha256": "b" * 64,
    }


def test_atlas_pool_identity_seals_model_data_targets_and_control(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_BACKBONE", "chronos2")
    monkeypatch.setenv("ATLAS_CONTROL", "random")
    monkeypatch.setenv("ATLAS_WINDOW", "256")
    atlas = _load("public_probe_atlas_identity", ROOT / "scripts" / "probe_atlas.py")
    corpus = tmp_path / "labels.npz"
    np.savez(corpus, ticker=np.asarray(["NQ"]))
    monkeypatch.setattr(atlas, "CORPUS", corpus)
    bars = {
        ("NQ", "3min"): {"source_sha256": "source-hash"},
    }

    assert atlas.BACKBONE == "chronos2"
    assert atlas.CONTROL == "random"
    assert atlas.WINDOW == 256
    assert atlas.POOL == "reg"
    assert len(atlas._source_sha256(bars)) == 64
    assert len(atlas._file_sha256(corpus)) == 64


def test_public_lifecycle_marks_breaks_without_strategy_imports():
    lifecycle = _load("public_trend_lifecycle", ROOT / "scripts" / "trend_lifecycle.py")
    high = np.array([10.0, 0.0, 12.0, 0.0, 11.0])
    low = np.array([0.0, 5.0, 0.0, 6.0, 0.0])
    pivots = [
        {"origin": 0, "confirm": 0, "direction": -1},
        {"origin": 1, "confirm": 1, "direction": 1},
        {"origin": 2, "confirm": 2, "direction": -1},
        {"origin": 3, "confirm": 3, "direction": 1},
        {"origin": 4, "confirm": 4, "direction": -1},
    ]
    labels = lifecycle.label_trend_lifecycle(high, low, pivots)
    assert labels[2]["role_kind"] == "start"
    assert labels[3]["kind"] == "end"
    assert labels[4]["role_kind"] == "start"
