"""Rebuild FFM-owned continuous bars for the complete 9x4 model matrix.

Each raw archive is decoded once and resampled to every requested timeframe.
Outputs and provenance manifests are deliberately restricted to this repository;
downstream projects must consume them read-only instead of copying market data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "databento" / "build_continuous.py"
UPDATE_PATTERN = "NQESGCSIRTYYM-glbx-mdp3-*.ohlcv-1m.dbn.zst"
SOURCE_GROUPS = (
    (("NQ-glbx-mdp3-*.ohlcv-1m.csv.zst", UPDATE_PATTERN), "NQ"),
    (("ES-glbx-mdp3-*.ohlcv-1m.csv.zst", UPDATE_PATTERN), "ES"),
    (("GC-glbx-mdp3-*.ohlcv-1m.csv.zst", UPDATE_PATTERN), "GC"),
    (("SIL-glbx-mdp3-*.ohlcv-1m.csv.zst", UPDATE_PATTERN), "SI"),
    (("glbx-mdp3-*.ohlcv-1m.dbn.zst", UPDATE_PATTERN), "RTY,YM"),
    (("CL-glbx-mdp3-*.ohlcv-1m.dbn.zst",), "CL"),
    (("ZB-ZN-glbx-mdp3-*.ohlcv-1m.dbn.zst",), "ZB,ZN"),
)


def _notify(message: str) -> None:
    subprocess.run(
        ["osascript", "-e",
         f'display notification "{message}" with title '
         '"Futures Foundation Model"'],
        check=False,
    )


def _source(pattern: str) -> Path:
    matches = sorted((ROOT / "databento").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one source matching {pattern!r}, found {matches}")
    return matches[0]


def _inside_ffm(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"market-data output must remain inside FFM repository: {resolved}") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", default="1min,3min,5min,15min")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "replay")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "temp" / "roll_audit_9x4_v1")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    output_dir = _inside_ffm(args.output_dir)
    run_dir = _inside_ffm(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    (run_dir / ".started").touch()
    commands: list[list[str]] = []

    try:
        for patterns, tickers in SOURCE_GROUPS:
            sources = [_source(pattern) for pattern in patterns]
            command = [
                sys.executable, str(BUILDER), "--periods", args.periods,
                "--back-adjust", "--output-dir", str(output_dir),
            ]
            for source in sources:
                command.extend(["--source", str(source)])
            if len(sources) > 1:
                command.append("--merge-sources")
            command.extend(["--tickers", tickers])
            commands.append(command)
            subprocess.run(command, cwd=ROOT, check=True)
    except Exception as error:
        (run_dir / ".failed").touch()
        (run_dir / "run.json").write_text(json.dumps({
            "schema": "ffm_roll_safe_matrix_run_v1",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": str(error),
            "commands": commands,
        }, indent=2) + "\n")
        if not args.no_notify:
            _notify("FFM corrected 9x4 rebuild failed")
        raise

    (run_dir / ".completed").touch()
    (run_dir / "run.json").write_text(json.dumps({
        "schema": "ffm_roll_safe_matrix_run_v1",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "periods": args.periods.split(","),
        "output_dir": str(output_dir),
        "commands": commands,
    }, indent=2) + "\n")
    if not args.no_notify:
        _notify("FFM corrected 9x4 rebuild completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
