"""RR barrier evaluation for labeling trade outcomes."""

import numpy as np


def apply_rr_barriers(h, l, c, entry_idx, is_long, entry_price, sl_price,
                      rr_targets, lookahead=None, is_session_end=None):
    """
    Walk forward from entry_idx measuring R:R outcomes for each target.

    Stops early on stop hit, session end, lookahead exhaustion, or data end.

    Args:
        h, l, c        : float arrays aligned to the bar series
        entry_idx      : int — bar index of entry
        is_long        : bool
        entry_price    : float
        sl_price       : float
        rr_targets     : list[float] — e.g. [1.0, 1.5, 2.0, 3.0]
        lookahead      : int or None — max bars to look forward
        is_session_end : bool array or None — True on last bar of each session

    Returns:
        dict[float → dict] — keys: hit (bool), outcome (str), realized_rr (float)
        outcomes: 'target_hit', 'stopped', 'session_end', 'data_end', 'invalid'
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return {rr: {'hit': False, 'outcome': 'invalid', 'realized_rr': 0.0}
                for rr in rr_targets}

    targets = {
        rr: entry_price + stop_dist * rr * (1 if is_long else -1)
        for rr in rr_targets
    }
    results = {rr: {'hit': False, 'outcome': None, 'realized_rr': 0.0}
               for rr in rr_targets}

    end = n if lookahead is None else min(entry_idx + lookahead + 1, n)

    for j in range(entry_idx + 1, end):
        if is_session_end is not None and is_session_end[j]:
            fr = ((c[j] - entry_price) / stop_dist if is_long
                  else (entry_price - c[j]) / stop_dist)
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome']     = 'session_end'
                    results[rr]['realized_rr'] = fr
            break

        if (is_long and l[j] <= sl_price) or (not is_long and h[j] >= sl_price):
            for rr in rr_targets:
                if results[rr]['outcome'] is None:
                    results[rr]['outcome']     = 'stopped'
                    results[rr]['realized_rr'] = -1.0
            break

        for rr in sorted(rr_targets):
            if results[rr]['outcome'] is not None:
                continue
            if ((is_long and h[j] >= targets[rr]) or
                    (not is_long and l[j] <= targets[rr])):
                results[rr]['hit']         = True
                results[rr]['outcome']     = 'target_hit'
                results[rr]['realized_rr'] = rr

        if all(r['outcome'] is not None for r in results.values()):
            break

    for rr in rr_targets:
        if results[rr]['outcome'] is None:
            results[rr]['outcome']     = 'data_end'
            results[rr]['realized_rr'] = 0.0

    return results


def best_rr_hit(rr_results, min_rr=1.0):
    """Return the highest R:R hit at or above min_rr, or 0.0 if none."""
    return max(
        (rr for rr, r in rr_results.items() if r['hit'] and rr >= min_rr),
        default=0.0,
    )
