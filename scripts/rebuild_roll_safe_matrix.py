"""Rebuild FFM-owned continuous bars for the complete 9x4 model matrix.

Each raw archive is decoded once and resampled to every requested timeframe.
Outputs and provenance manifests are deliberately restricted to this repository;
downstream projects must consume them read-only instead of copying market data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from futures_foundation.data_provenance import seal_continuous_streams  # noqa: E402


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
PERMITTED_TICKERS = ('CL', 'ES', 'GC', 'NQ', 'RTY', 'SI', 'YM', 'ZB', 'ZN')


def _notify(message: str) -> None:
    subprocess.run(
        ["osascript", "-e",
         f'display notification "{message}" with title '
         '"Futures Foundation Model"'],
        check=False,
    )


def _sources(pattern: str) -> list[Path]:
    matches = sorted((ROOT / "databento").glob(pattern))
    if not matches:
        raise RuntimeError(f"no source matches {pattern!r}")
    return matches


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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "temp" / "roll_audit_9x4_v1")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    output_dir = _inside_ffm(args.output_dir)
    run_dir = _inside_ffm(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    periods = [item.strip() for item in args.periods.split(',') if item.strip()]
    if periods != ['1min', '3min', '5min', '15min']:
        parser.error('the production matrix must contain exactly 1min,3min,5min,15min')
    stage_dir = Path(tempfile.mkdtemp(
        prefix='.ffm-roll-safe-', dir=output_dir.parent))
    started = datetime.now(timezone.utc).isoformat()
    (run_dir / ".started").touch()
    commands: list[list[str]] = []

    try:
        for patterns, tickers in SOURCE_GROUPS:
            sources = []
            for pattern in patterns:
                sources.extend(_sources(pattern))
            sources = list(dict.fromkeys(path.resolve() for path in sources))
            command = [
                sys.executable, str(BUILDER), "--periods", args.periods,
                "--back-adjust", "--output-dir", str(stage_dir),
            ]
            for source in sources:
                command.extend(["--source", str(source)])
            if len(sources) > 1:
                command.append("--merge-sources")
            command.extend(["--tickers", tickers])
            commands.append(command)
            subprocess.run(command, cwd=ROOT, check=True)

        streams = [
            (ticker, period)
            for ticker in PERMITTED_TICKERS for period in periods
        ]
        sealed = seal_continuous_streams(stage_dir, streams, repo_root=ROOT)
        generation_id = str(uuid.uuid4())
        generation = {
            'schema': 'ffm_continuous_generation_v1',
            'generation_id': generation_id,
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'tickers': list(PERMITTED_TICKERS),
            'periods': periods,
            'streams': sealed['streams'],
        }
        (stage_dir / 'continuous_generation.json').write_text(
            json.dumps(generation, indent=2, sort_keys=True) + '\n')
        backup_dir = _promote_matrix(stage_dir, output_dir, streams, generation_id)
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
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    (run_dir / ".completed").touch()
    (run_dir / "run.json").write_text(json.dumps({
        "schema": "ffm_roll_safe_matrix_run_v1",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "periods": periods,
        "output_dir": str(output_dir),
        "generation_id": generation_id,
        "backup_dir": str(backup_dir),
        "commands": commands,
    }, indent=2) + "\n")
    if not args.no_notify:
        _notify("FFM corrected 9x4 rebuild completed")
    return 0


def _promote_matrix(stage_dir: Path, output_dir: Path,
                    streams: list[tuple[str, str]], generation_id: str) -> Path:
    """Promote the fully sealed 9x4 matrix while readers fail on the lock."""
    lock = output_dir / '.continuous_update.lock'
    if lock.exists():
        raise RuntimeError(f'another continuous-data update is active: {lock}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = output_dir / f'backup_{stamp}_{generation_id[:8]}'
    backup.mkdir(parents=True, exist_ok=False)
    lock.write_text(json.dumps({
        'generation_id': generation_id,
        'started_utc': datetime.now(timezone.utc).isoformat(),
    }) + '\n')
    names = []
    for ticker, period in streams:
        names.extend([
            f'{ticker}_{period}.csv',
            f'{ticker}_{period}.csv.manifest.json',
        ])
    names.append('continuous_generation.json')
    existing = set()
    try:
        for name in names:
            target = output_dir / name
            if target.exists():
                shutil.copy2(target, backup / name)
                existing.add(name)
        for name in names:
            os.replace(stage_dir / name, output_dir / name)
    except Exception:
        for name in names:
            saved = backup / name
            target = output_dir / name
            if saved.exists():
                shutil.copy2(saved, target)
            elif name not in existing and target.exists():
                target.unlink()
        raise
    finally:
        lock.unlink(missing_ok=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
    return backup


if __name__ == "__main__":
    raise SystemExit(main())
