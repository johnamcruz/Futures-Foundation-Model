"""CLI guards for the flow runs and the atlas partial-timeframe opt-in."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_cli(monkeypatch, *args):
    module = _load("chronos2_ssl_pretrain_cli", ROOT / "scripts" / "chronos" / "chronos2_ssl_pretrain.py")
    parent = ROOT  # exists; the guards fire before any data is read
    monkeypatch.setattr(sys, "argv", ["x", "--parent", str(parent), *args])
    module.main()


@pytest.mark.parametrize("args,message", [
    (["--flow-control", "shuf"], "needs --flow-path"),
    (["--flow-control", "proxy"], "needs --flow-path"),
    (["--flow-path", "f.parquet", "--nq-only"], "mutually exclusive"),
])
def test_flow_cli_refuses_inconsistent_options(monkeypatch, args, message):
    with pytest.raises(SystemExit) as raised:
        _run_cli(monkeypatch, *args)
    assert message in str(raised.value)


def test_cli_exposes_flow_and_mask_options():
    module = _load("chronos2_ssl_pretrain_cli2", ROOT / "scripts" / "chronos" / "chronos2_ssl_pretrain.py")
    ns = module.parser().parse_args(["--flow-path", "f.parquet", "--flow-control", "shuf", "--mask-mode", "tail", "--holdout-start", "2025-11-29"])
    assert (ns.flow_control, ns.mask_mode, ns.holdout_start) == ("shuf", "tail", "2025-11-29")
    default = module.parser().parse_args([])
    assert (default.flow_path, default.flow_control, default.mask_mode, default.nq_only) == (None, "flow", "time", False)


def test_atlas_timeframe_contract_is_strict_unless_opted_in():
    atlas = _load("chronos2_probe_atlas_tf", ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py")
    full = ["1min", "3min", "5min", "15min"]
    assert atlas._timeframes_ok(full, False) and atlas._timeframes_ok(list(reversed(full)), False)
    assert not atlas._timeframes_ok(["3min"], False)                      # default refuses a partial stage
    assert atlas._timeframes_ok(["3min"], True) and atlas._timeframes_ok(["1min", "3min"], True)
    for junk in ("3min", ["7min"], [], None, [3], ["3min", "7min"]):
        assert not atlas._timeframes_ok(junk, True)                      # the opt-in does not accept junk


def test_cli_passes_holdout_start_and_flow_control_to_the_data_builder(monkeypatch, tmp_path):
    module = _load("chronos2_ssl_pretrain_cli3", ROOT / "scripts" / "chronos" / "chronos2_ssl_pretrain.py")
    seen = {}

    class Stub:
        import numpy as _np
        train = _np.zeros((11, 4)); report = {"finite_fraction": 1.0}

    def fake_prepare(*args, **kwargs):
        seen.update(kwargs); return Stub()

    monkeypatch.setattr(module, "prepare_multivariate", fake_prepare)
    monkeypatch.setattr(sys, "argv", ["x", "--parent", str(ROOT), "--flow-path", "f.parquet", "--flow-control", "shuf",
                                      "--holdout-start", "2025-11-29", "--preflight-only", "--out-dir", str(tmp_path)])
    module.main()
    assert seen["holdout_start"] == "2025-11-29" and seen["flow_control"] == "shuf" and seen["tickers"] == ("NQ",)


def test_atlas_timeframes_reject_duplicates():
    atlas = _load("chronos2_probe_atlas_tf2", ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py")
    assert not atlas._timeframes_ok(["3min", "3min"], True)
    assert not atlas._timeframes_ok(["1min", "3min", "5min", "15min", "15min"], False)


def test_resume_identity_is_checked_and_an_identityless_state_is_refused_for_flow_runs():
    pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import _check_resume_identity
    ident = {"data_identity_sha256": "a", "mask_mode": "cross", "mask_ratio": 0.4, "seed": 0}
    _check_resume_identity({"run_identity": dict(ident)}, ident, True)
    for changed in ({"mask_mode": "tail"}, {"seed": 1}, {"data_identity_sha256": "b"}):
        with pytest.raises(RuntimeError, match="different"):
            _check_resume_identity({"run_identity": {**ident, **changed}}, ident, True)
    with pytest.raises(RuntimeError, match="no run identity"):
        _check_resume_identity({}, ident, True)
    plain = {**ident, "mask_mode": "time"}
    _check_resume_identity({}, plain, False)              # an old plain time-mask state still resumes
    with pytest.raises(RuntimeError, match="no run identity"):
        _check_resume_identity({}, plain, True)


def test_cross_mode_without_flow_channels_is_refused_before_any_model_is_loaded(tmp_path):
    pytest.importorskip("torch")
    from futures_foundation.finetune.classifiers.chronos2.ssl_stages import train_mask

    class Item:
        channel_names = tuple(f"NQ.{c}" for c in ("open", "high", "low", "close", "volume"))
        report = {"timeframe": "3min", "identity_sha256": "x"}

    with pytest.raises(ValueError, match="needs flow channels"):
        train_mask(Item(), parent=tmp_path / "missing", out_dir=tmp_path / "o", mask_mode="cross")


def test_remote_checkpoint_identity_records_the_partial_opt_in():
    atlas = _load("chronos2_probe_atlas_remote", ROOT / "scripts" / "chronos" / "chronos2_probe_atlas.py")
    plain = atlas._stage_identity("org/model", "remote:org/model", None, context_length=256)
    opted = atlas._stage_identity("org/model", "remote:org/model", None, context_length=256, allow_partial_timeframes=True)
    assert "allow_partial_timeframes" not in plain and opted["allow_partial_timeframes"] is True
