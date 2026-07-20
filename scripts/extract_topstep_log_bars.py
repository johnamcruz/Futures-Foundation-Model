"""Extract hash-traceable OHLCV replay bars from an Algo Topstep backtest log.

Market data remains under FFM ownership.  The source Algo log is read-only and
the emitted manifest pins its SHA-256 so a diagnostic replay is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BAR = re.compile(
    r"Bar:\s+(?P<datetime>20\d{2}-\d{2}-\d{2} \d{2}:\d{2})\s+\|\s+"
    r"O:(?P<open>-?[\d.]+)\s+H:(?P<high>-?[\d.]+)\s+"
    r"L:(?P<low>-?[\d.]+)\s+C:(?P<close>-?[\d.]+)\s+"
    r"V:(?P<volume>[\d.]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bars(text: str) -> pd.DataFrame:
    rows = [match.groupdict() for match in BAR.finditer(text)]
    if not rows:
        raise ValueError("no Topstep bar records found")
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column])
    # Console and file handlers may print the same bar twice.  Exact duplicate
    # timestamps must agree; conflicting values indicate a corrupt source log.
    values = ["open", "high", "low", "close", "volume"]
    conflicts = frame.groupby("datetime")[values].nunique().max(axis=1) > 1
    if conflicts.any():
        raise ValueError(
            f"conflicting duplicate bars at {list(conflicts[conflicts].index[:5])}")
    return (frame.drop_duplicates("datetime", keep="first")
            .sort_values("datetime").reset_index(drop=True))


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
    parser.add_argument("log", type=Path)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timeframe", default="3min")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.log.expanduser().resolve()
    output = _inside_ffm(args.output)
    frame = parse_bars(source.read_text(errors="replace"))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = {
        "schema": "ffm_topstep_log_replay_v1",
        "ticker": args.ticker.upper(),
        "timeframe": args.timeframe,
        "timezone": "source_log_clock",
        "source_log": str(source),
        "source_log_sha256": _sha256(source),
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "rows": len(frame),
        "start": frame["datetime"].iloc[0].isoformat(),
        "end": frame["datetime"].iloc[-1].isoformat(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
