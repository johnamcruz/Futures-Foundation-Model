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
