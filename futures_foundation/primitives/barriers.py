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


def realized_r_trailing(h, l, c, entry_idx, is_long, entry_price, sl_price,
                         atr, trail_atr_k, activate_r=1.0, max_hold=None,
                         is_session_end=None):
    """Realized R under an ATR-trailing-stop exit policy (causal).

    Unlike apply_rr_barriers (fixed targets) and the MFE/`compute_max_rr`
    high-water-mark, this returns the R **actually captured** by a concrete,
    walk-forward exit policy — the tradeable outcome a trend setup yields:

        - initial hard stop at sl_price (= -1R)
        - once price reaches +activate_r R favorably, a trailing stop turns on
          at favorable_extreme - trail_atr_k*atr, ratcheting only in profit
          (never looser than the initial stop)
        - exit on stop touch (filled at the stop), session end, horizon
          (max_hold) or data end (marked to that bar's close)

    Pessimistic intrabar convention (matches apply_rr_barriers): the stop is
    checked before the favorable extreme is updated, so a bar that both stops
    and extends is treated as a stop.

    Args:
        h, l, c        : float arrays aligned to the bar series
        entry_idx      : int — entry bar index (walk starts at entry_idx+1)
        is_long        : bool
        entry_price    : float
        sl_price       : float — initial hard stop
        atr            : float — ATR at entry (trailing distance unit)
        trail_atr_k    : float — trail = extreme ∓ trail_atr_k*atr
        activate_r     : float — R favorable before trailing arms (default 1.0)
        max_hold       : int or None — max bars held (horizon cap)
        is_session_end : bool array or None — True on last bar of each session

    Returns:
        dict: realized_r (float, in R units), outcome
              ('stopped' | 'trailed' | 'session_end' | 'horizon' | 'data_end'
               | 'invalid'), exit_idx (int)
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    risk = abs(entry_price - sl_price)
    if risk <= 0 or atr <= 0 or entry_idx + 1 >= n:
        return {'realized_r': 0.0, 'outcome': 'invalid', 'exit_idx': entry_idx}

    sign       = 1.0 if is_long else -1.0
    end        = n if max_hold is None else min(entry_idx + max_hold + 1, n)
    extreme    = entry_price            # best favorable price so far
    armed      = False
    trail_stop = sl_price               # active stop (initial until armed)

    def _r(price):
        return sign * (price - entry_price) / risk

    for j in range(entry_idx + 1, end):
        if is_session_end is not None and is_session_end[j]:
            return {'realized_r': _r(c[j]), 'outcome': 'session_end',
                    'exit_idx': j}

        # 1) stop check first (pessimistic) — fill at the stop level
        if is_long:
            if l[j] <= trail_stop:
                return {'realized_r': _r(trail_stop),
                        'outcome': 'trailed' if armed else 'stopped',
                        'exit_idx': j}
        else:
            if h[j] >= trail_stop:
                return {'realized_r': _r(trail_stop),
                        'outcome': 'trailed' if armed else 'stopped',
                        'exit_idx': j}

        # 2) update favorable extreme, then arm / ratchet the trail
        extreme = max(extreme, h[j]) if is_long else min(extreme, l[j])
        fav_r   = sign * (extreme - entry_price) / risk
        if not armed and fav_r >= activate_r:
            armed = True
        if armed:
            cand = (extreme - trail_atr_k * atr if is_long
                    else extreme + trail_atr_k * atr)
            trail_stop = (max(trail_stop, cand) if is_long
                          else min(trail_stop, cand))

    last = end - 1
    return {'realized_r': _r(c[last]),
            'outcome': 'horizon' if end < n else 'data_end',
            'exit_idx': last}


# R-tier bucket edges (left-closed): (-inf,0) [0,2) [2,5) [5,10) [10,inf)
R_BUCKET_EDGES = (0.0, 2.0, 5.0, 10.0)


def r_bucket(realized_r):
    """Ordinal R-tier index 0..4 for {<0, 0-2, 2-5, 5-10, 10+}."""
    b = 0
    for edge in R_BUCKET_EDGES:
        if realized_r >= edge:
            b += 1
    return b
