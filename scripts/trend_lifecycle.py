"""Deterministic HH/HL versus LH/LL lifecycle labels for FFM evaluation.

This is ground-truth construction, not a served trading strategy.  A trend starts
when the structural direction changes, continues while successive same-role
swings agree, and ends retrospectively when a later swing breaks it.
"""
from __future__ import annotations


def label_trend_lifecycle(high, low, pivots):
    """Return structural lifecycle records parallel to alternating pivots."""
    output = []
    last_high_price = None
    last_low_price = None
    active_direction = 0
    last_index_by_direction = {1: None, -1: None}

    for pivot in pivots:
        origin = int(pivot["origin"])
        direction = int(pivot["direction"])
        confirm = int(pivot["confirm"])
        price = low[origin] if direction == 1 else high[origin]
        role = "low" if direction == 1 else "high"
        record = {
            "origin": origin, "confirm": confirm, "direction": direction,
            "px": float(price), "role": role, "swing_type": None,
            "trend_dir": None, "role_kind": None, "ended": False,
        }
        if role == "high":
            if last_high_price is not None:
                record["swing_type"] = "HH" if price > last_high_price else "LH"
                record["trend_dir"] = 1 if price > last_high_price else -1
            last_high_price = price
        else:
            if last_low_price is not None:
                record["swing_type"] = "LL" if price < last_low_price else "HL"
                record["trend_dir"] = -1 if price < last_low_price else 1
            last_low_price = price

        index = len(output)
        output.append(record)
        structural_direction = record["trend_dir"]
        if structural_direction is None:
            continue
        if structural_direction == active_direction:
            record["role_kind"] = "continue"
        else:
            record["role_kind"] = "start"
            previous = last_index_by_direction.get(active_direction)
            if previous is not None:
                output[previous]["ended"] = True
            active_direction = structural_direction
        last_index_by_direction[structural_direction] = index

    for record in output:
        if record["role_kind"] is None:
            record["kind"] = None
        elif record["role_kind"] == "start":
            record["kind"] = "start_end" if record["ended"] else "start"
        else:
            record["kind"] = "end" if record["ended"] else "continue"
    return output
