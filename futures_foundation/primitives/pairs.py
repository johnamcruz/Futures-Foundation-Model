"""Correlated-sibling pairing — the CROSS-INSTRUMENT window primitive (torch-free, causality-first).

THE IDEA (sibling participation): a true reversal shows BROAD participation — ES and NQ turn
together; a doomed bounce is usually NARROW (one instrument pops while its sibling keeps bleeding).
That divergence is evidence from OUTSIDE the instrument — invisible in its own OHLCV at any scale
(measured: the own-price discrimination ceiling on strong-counter fades is ~0.576 AUC for every
objective). Feeding the sibling's SAME-TF window alongside the instrument's own window puts the
participation signal INSIDE the model's input: [5,seq] + [5,seq] -> [10,seq] on the channel axis,
the channel-independent encoder makes that [emb_own | emb_sibling].

THE ONE LEAK SPOT — cross-stream alignment — lives here, once, tested: the sibling window must end
at the sibling's LAST CLOSED bar at-or-before the anchor bar's timestamp (asof; never a future bar).
Two same-TF streams stamp bars on the same clock grid, so the aligned bar normally IS the
simultaneous one; when the sibling has a gap (halt, missing bar) the asof falls back to an earlier
bar, and MAX_GAP guards against silently feeding STALE context (a halted sibling is no confirmation
signal at all -> the caller drops/flags the anchor instead).

DEFAULT_SIBLINGS is market-structure knowledge (like the ticker universe), not strategy IP:
tightest-cointegration partners; RTY/YM confirm against the complex leader (ES); CL has no clean
in-universe sibling -> ES as the risk proxy (the weakest link — measurable per-group, prune if noise).
"""
import numpy as np

# traded instrument -> the sibling whose same-TF window rides along (see module docstring)
DEFAULT_SIBLINGS = {
    'ES': 'NQ', 'NQ': 'ES',            # the tightest equity confirmation pair (both directions)
    'RTY': 'ES', 'YM': 'ES',           # satellites confirm against the complex leader
    'GC': 'SI', 'SI': 'GC',            # metals
    'ZB': 'ZN', 'ZN': 'ZB',            # rates
    'CL': 'ES',                        # risk proxy (no clean energy sibling in-universe)
}


def parse_siblings(spec):
    """Env-style override -> pair map. '' / '0' -> None (off); '1' / 'on' -> DEFAULT_SIBLINGS;
    'ES:NQ,CL:' -> custom map on top of the default (empty value = EXCLUDE that ticker)."""
    s = (spec or '').strip()
    if s in ('', '0', 'off'):
        return None
    if s in ('1', 'on', 'default'):
        return dict(DEFAULT_SIBLINGS)
    out = dict(DEFAULT_SIBLINGS)
    for part in s.split(','):
        k, _, v = part.strip().partition(':')
        if not k:
            continue
        if v:
            out[k] = v
        else:
            out.pop(k, None)                              # 'CL:' = exclude CL from pairing
    return out


def asof_sibling_index(own_ts_i, sib_ts, max_gap=None):
    """Index of the sibling's last bar stamped AT-OR-BEFORE own timestamp `own_ts_i` (asof — the
    causality primitive: NEVER a future bar). Returns -1 when no such bar exists or when the found
    bar is STALER than max_gap (same units as the timestamps' deltas) — a halted/gapped sibling is
    no confirmation signal, so the caller must drop/flag rather than silently use stale context.

    Timestamps are bar OPEN times on a shared clock grid: two same-TF bars with equal stamps CLOSE
    at the same moment, so at the anchor bar's close the equal-stamped sibling bar is closed too —
    and any consumer entering at anchor+1 sees only closed sibling data."""
    sib_ts = np.asarray(sib_ts)
    j = int(np.searchsorted(sib_ts, own_ts_i, side='right')) - 1
    if j < 0:
        return -1
    if max_gap is not None and (own_ts_i - sib_ts[j]) > max_gap:
        return -1
    return j


def sibling_ohlcv_window(own_ts_i, sib, seq, max_gap=None):
    """The sibling's [5, seq] raw OHLCV window ending at its last closed bar <= `own_ts_i`
    (asof-aligned, strictly causal). `sib` = dict with 'ts','o','h','l','c','v' arrays.
    Returns None when the sibling has no aligned bar, is staler than max_gap, or has fewer than
    `seq` bars of history at the aligned point — the caller drops/flags that anchor."""
    j = asof_sibling_index(own_ts_i, sib['ts'], max_gap=max_gap)
    if j < 0 or j + 1 < int(seq):
        return None
    sl = slice(j - int(seq) + 1, j + 1)
    w = np.stack([sib['o'][sl], sib['h'][sl], sib['l'][sl], sib['c'][sl], sib['v'][sl]])
    return np.nan_to_num(np.asarray(w, np.float32))
