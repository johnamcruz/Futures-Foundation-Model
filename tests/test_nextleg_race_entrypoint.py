"""Standalone NextLeg Race launcher contracts."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mantis_ssl_nextleg_race.py"
SPEC = importlib.util.spec_from_file_location("mantis_ssl_nextleg_race", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _preflight_environment(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    parent = tmp_path / "structural.pt"
    parent.write_bytes(b"encoder")
    output = tmp_path / "race.pt"
    monkeypatch.setattr(
        launcher, "_resolve",
        lambda args: (data, parent, output, tmp_path / "labels.npz"))
    streams = [
        {"ticker": ticker, "timeframe": timeframe}
        for ticker in launcher.TICKERS for timeframe in launcher.TIMEFRAMES
    ]
    monkeypatch.setattr(
        launcher, "seal_continuous_streams",
        lambda *args, **kwargs: {"streams": streams, "schema": "test"})
    monkeypatch.setattr(launcher, "sha256", lambda path: f"hash:{Path(path).name}")
    return output


def test_launcher_defaults_to_lora_bar_only_v2_contract():
    args = launcher._parser().parse_args([])
    assert args.lora_r == 8
    assert args.lora_alpha == 16
    assert args.freeze_encoder_layers == 2
    assert args.race_levels == "1,2,3,4"
    assert args.controls == "shuffle,random"
    source = SCRIPT.read_text()
    assert "compute_atr" not in source
    assert 'pretext="nextleg_race"' in source


def test_preflight_is_read_only_and_does_not_start_training(tmp_path, monkeypatch):
    output = _preflight_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher.sys, "argv",
        [str(SCRIPT), "--preflight-only", "--skip-atlas", "--device", "cpu"])
    monkeypatch.setattr(
        launcher.ssl, "loop_ssl",
        lambda **kwargs: pytest.fail("preflight must not start training"))
    launcher.main()
    assert not output.exists()
    assert not Path(str(output) + ".data_provenance.json").exists()


def test_launcher_refuses_full_training_without_lora(tmp_path, monkeypatch):
    _preflight_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher.sys, "argv",
        [str(SCRIPT), "--preflight-only", "--skip-atlas", "--lora-r", "0"])
    with pytest.raises(ValueError, match="requires LoRA"):
        launcher.main()


def test_colab_probe_label_discovery_prefers_clean_lineage(tmp_path, monkeypatch):
    monkeypatch.delenv("TREND_LABELS", raising=False)
    ordinary = tmp_path / "clean_ssl_pre2026_lora"
    selected = tmp_path / "other"
    ordinary.mkdir()
    selected.mkdir()
    labels = ordinary / "probe_atlas" / "trend_lifecycle_labels_pre2026.npz"
    labels.parent.mkdir()
    labels.write_bytes(b"labels")
    other = selected / "trend_lifecycle_labels_pre2026.npz"
    other.write_bytes(b"labels")
    assert launcher._discover_probe_labels(tmp_path) == labels


def test_colab_rejects_unmaterialized_lfs_checkpoint(tmp_path):
    pointer = tmp_path / "model.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\nsize 32000000\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        launcher._require_materialized_checkpoint(pointer)
