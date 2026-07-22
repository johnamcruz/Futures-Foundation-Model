"""Causal related-series alignment for Mantis pretraining and inference.

Chronos-2 lets time series with the same task/group identifier exchange information.  Mantis
keeps its compact single-series encoder; this module supplies the equivalent *group construction*
for futures.  Every primary window is paired with other timeframes of the same ticker and, when
configured, a same-timeframe sibling (for example NQ <-> ES).

CSV timestamps in this repository are bar-open timestamps.  Alignment therefore compares bar
*close* times (open + timeframe): a 15-minute bar stamped 10:00 is unavailable to a 3-minute
decision stamped 10:03, and becomes available only once it has closed.  This is the critical
closed-candle leakage boundary and is centralized here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from ..primitives.pairs import DEFAULT_SIBLINGS, parse_siblings


_TF_RE = re.compile(r"^\s*(\d+)\s*(?:m|min|mins|minute|minutes)\s*$", re.I)


def timeframe_minutes(value: str | int) -> int:
    """Return a positive minute count from repository timeframe names such as ``3min``."""
    if isinstance(value, (int, np.integer)):
        minutes = int(value)
    else:
        match = _TF_RE.match(str(value))
        if not match:
            raise ValueError(f"unsupported timeframe {value!r}; expected e.g. '3min'")
        minutes = int(match.group(1))
    if minutes <= 0:
        raise ValueError("timeframe must be positive")
    return minutes


def parse_timeframe_pairs(value) -> dict[str, tuple[str, ...]] | None:
    """Parse optional primary-to-context timeframe mappings.

    ``1min=5min,3min=15min`` is concise bidirectional pairing. Directional mappings use ``:``
    and may provide multiple contexts with ``+`` (for example ``1min:3min+5min+15min``).
    ``None`` preserves the historical all-configured-timeframes behavior; an empty string means
    primary-only context.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        parsed = {
            str(primary): tuple(dict.fromkeys(
                (contexts,) if isinstance(contexts, str) else tuple(contexts)))
            for primary, contexts in value.items()
        }
    else:
        text = str(value).strip()
        if not text or text.lower() in {"0", "none", "off"}:
            return {}
        parsed_lists: dict[str, list[str]] = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                left, right = (part.strip() for part in item.split("=", 1))
                if "+" in left or "+" in right:
                    raise ValueError("bidirectional timeframe pairs must contain one TF per side")
                parsed_lists.setdefault(left, []).append(right)
                parsed_lists.setdefault(right, []).append(left)
            elif ":" in item:
                left, right = (part.strip() for part in item.split(":", 1))
                parsed_lists.setdefault(left, []).extend(
                    part.strip() for part in right.split("+") if part.strip())
            else:
                raise ValueError(
                    f"invalid timeframe pair {item!r}; use 1min=5min or 1min:3min+5min")
        parsed = {primary: tuple(dict.fromkeys(contexts))
                  for primary, contexts in parsed_lists.items()}
    for primary, contexts in parsed.items():
        timeframe_minutes(primary)
        for context in contexts:
            timeframe_minutes(context)
            if context == primary:
                raise ValueError(f"timeframe {primary} cannot pair with itself")
    return parsed


def _ns(values) -> np.ndarray:
    """Datetime-like values -> int64 nanoseconds without changing their ordering."""
    return np.asarray(values).astype("datetime64[ns]").astype(np.int64)


@dataclass(frozen=True)
class RelatedStream:
    """One usable (pre-holdout) stream inside the concatenated SSL bar matrix."""

    sid: str
    ticker: str
    tf: str
    tf_minutes: int
    base: int
    size: int
    close_ns: np.ndarray

    @property
    def end(self) -> int:
        return self.base + self.size


@dataclass(frozen=True)
class RelatedWindowPlan:
    """Aligned global window starts and masks for primary starts.

    Columns are stable across primary streams: primary, one slot per configured exact timeframe,
    then sibling.  The slot matching the primary timeframe is intentionally masked to avoid
    feeding a duplicate of the primary series.  ``role_ids`` identify slot semantics for a learned
    role embedding; zero is always the primary role.
    """

    primary_starts: np.ndarray       # [N]
    starts: np.ndarray               # [N, R], global starts (dummy 0 where mask=False)
    mask: np.ndarray                 # [N, R]
    role_ids: np.ndarray             # [R]
    role_names: tuple[str, ...]

    def __post_init__(self):
        n = len(self.primary_starts)
        if self.starts.ndim != 2 or self.starts.shape != self.mask.shape:
            raise ValueError("related starts/mask must have the same [N,R] shape")
        if self.starts.shape[0] != n or self.starts.shape[1] != len(self.role_ids):
            raise ValueError("related plan dimensions do not match starts/roles")
        if len(self.role_names) != len(self.role_ids):
            raise ValueError("role_names and role_ids must have equal length")
        if n and (not self.mask[:, 0].all() or not np.array_equal(self.starts[:, 0], self.primary_starts)):
            raise ValueError("slot zero must be the valid primary window")

    @property
    def num_roles(self) -> int:
        return len(self.role_ids)

    def valid_fraction(self) -> dict[str, float]:
        return {name: float(self.mask[:, i].mean()) if len(self.mask) else 0.0
                for i, name in enumerate(self.role_names)}


@dataclass(frozen=True)
class RelatedSeriesLayout:
    """Metadata needed to align windows inside the concatenated SSL matrix."""

    streams: tuple[RelatedStream, ...]

    @classmethod
    def from_assembled(cls, records) -> "RelatedSeriesLayout":
        """Build from dict records containing sid/ticker/tf/base and truncated ``ts``."""
        out = []
        for record in records:
            ts_ns = _ns(record["ts"])
            minutes = timeframe_minutes(record["tf"])
            out.append(RelatedStream(
                sid=str(record["sid"]), ticker=str(record["ticker"]), tf=str(record["tf"]),
                tf_minutes=minutes, base=int(record["base"]), size=len(ts_ns),
                close_ns=ts_ns + np.int64(minutes) * 60_000_000_000,
            ))
        return cls(tuple(out))

    def _stream_for_starts(self, starts: np.ndarray) -> np.ndarray:
        ends = np.asarray([stream.end for stream in self.streams], np.int64)
        which = np.searchsorted(ends, starts, side="right")
        if len(starts) and ((which >= len(self.streams)).any()
                            or any(starts[i] < self.streams[int(j)].base
                                   for i, j in enumerate(which))):
            raise ValueError("a primary start falls outside the assembled stream layout")
        return which

    def align(self, primary_starts, context_length: int, *,
              related_tfs=("1min", "3min", "5min", "15min"),
              tf_pairs=None, siblings="default",
              max_gap_factor: float = 2.0) -> RelatedWindowPlan:
        """Align causal related windows to each primary window.

        A candidate bar is usable only when its close time is at-or-before the primary decision
        close.  Stale candidates (typically data gaps) are masked when the lag exceeds
        ``max_gap_factor * max(primary_tf, candidate_tf)``.  Missing/insufficient histories are
        represented by ``mask=False`` and never substituted with a future bar.
        """
        starts = np.asarray(primary_starts, np.int64)
        seq = int(context_length)
        if seq <= 0:
            raise ValueError("context_length must be positive")
        pair_map_tf = parse_timeframe_pairs(tf_pairs)
        if pair_map_tf is None:
            tfs = tuple(dict.fromkeys(str(tf) for tf in related_tfs))
        else:
            # Stable absolute-timeframe roles across every primary. A stream only enables the
            # subset declared for its own timeframe below.
            requested = {tf for contexts in pair_map_tf.values() for tf in contexts}
            tfs = tuple(sorted(requested, key=timeframe_minutes))
        for tf in tfs:
            timeframe_minutes(tf)
        pair_map = parse_siblings(siblings) if isinstance(siblings, str) else siblings
        pair_map = dict(pair_map or {})
        role_names = ("primary",) + tuple(f"same_ticker@{tf}" for tf in tfs) + ("sibling",)
        role_ids = np.arange(len(role_names), dtype=np.int64)
        aligned = np.zeros((len(starts), len(role_names)), dtype=np.int64)
        valid = np.zeros_like(aligned, dtype=bool)
        aligned[:, 0], valid[:, 0] = starts, True
        if not len(starts):
            return RelatedWindowPlan(starts, aligned, valid, role_ids, role_names)

        by_key = {(s.ticker, s.tf): s for s in self.streams}
        which = self._stream_for_starts(starts)
        for stream_i, primary in enumerate(self.streams):
            rows = np.flatnonzero(which == stream_i)
            if not len(rows):
                continue
            local_end = starts[rows] - primary.base + seq - 1
            if (local_end < 0).any() or (local_end >= primary.size).any():
                raise ValueError(f"primary context crosses the {primary.sid} stream boundary")
            decision_close = primary.close_ns[local_end]

            allowed = None if pair_map_tf is None else set(pair_map_tf.get(primary.tf, ()))
            candidates = [
                by_key.get((primary.ticker, tf))
                if tf != primary.tf and (allowed is None or tf in allowed) else None
                for tf in tfs
            ]
            sibling_ticker = pair_map.get(primary.ticker)
            candidates.append(by_key.get((sibling_ticker, primary.tf)) if sibling_ticker else None)
            for col, candidate in enumerate(candidates, start=1):
                if candidate is None:
                    continue
                end_i = np.searchsorted(candidate.close_ns, decision_close, side="right") - 1
                enough = end_i + 1 >= seq
                safe_i = np.maximum(end_i, 0)
                lag = decision_close - candidate.close_ns[safe_i]
                max_gap_ns = (float(max_gap_factor) * max(primary.tf_minutes, candidate.tf_minutes)
                              * 60_000_000_000)
                ok = enough & (lag >= 0) & (lag <= max_gap_ns)
                target_rows = rows[ok]
                aligned[target_rows, col] = candidate.base + end_i[ok] - seq + 1
                valid[target_rows, col] = True

        return RelatedWindowPlan(starts, aligned, valid, role_ids, role_names)


__all__ = ["RelatedSeriesLayout", "RelatedStream", "RelatedWindowPlan", "timeframe_minutes",
           "parse_timeframe_pairs", "DEFAULT_SIBLINGS"]
