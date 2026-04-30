"""Session filters and VWAP."""

import numpy as np


def session_mask(index, start_hour=7, start_min=0, end_hour=16, end_min=0):
    """
    Boolean mask for bars inside the given session window.
    index : DatetimeIndex with tz-aware NY timestamps.
    Returns bool array (n,).
    """
    t = index.hour * 60 + index.minute
    s = start_hour * 60 + start_min
    e = end_hour   * 60 + end_min
    return (t >= s) & (t < e)


def session_end_mask(index, start_hour=7, start_min=0, end_hour=16, end_min=0):
    """
    Boolean mask marking the last bar inside each session window.
    Returns bool array (n,).
    """
    in_sess  = session_mask(index, start_hour, start_min, end_hour, end_min)
    out      = np.zeros(len(index), dtype=bool)
    out[:-1] = in_sess[:-1] & ~in_sess[1:]
    return out


def compute_vwap(h, l, c, v, session_start_mask):
    """
    Session-reset VWAP. Resets at each True entry in session_start_mask.
    Returns float64 array (n,). NaN before the first session start.
    """
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = len(c)

    tp  = (h + l + c) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)

    cum_tpv    = 0.0
    cum_v      = 0.0
    in_session = False

    for i in range(n):
        if session_start_mask[i]:
            cum_tpv    = 0.0
            cum_v      = 0.0
            in_session = True
        if in_session:
            cum_tpv += tp[i] * v[i]
            cum_v   += v[i]
            out[i]   = cum_tpv / cum_v if cum_v > 0 else tp[i]

    return out
