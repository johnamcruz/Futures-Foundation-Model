"""Roll-splice detection — the DATA-HYGIENE primitive for unadjusted continuous contracts.

THE ISSUE: build_continuous.py stitches per-contract bars WITHOUT back-adjustment (a deliberate
convention — it preserves real traded prices instead of manufacturing synthetic price history).
That means quarterly contract rolls can leave a large bar-to-bar close jump in the series where
one contract's last bar meets the next contract's first bar. A candidate whose INPUT WINDOW
contains one of these jumps trains on a pattern live single-contract bars never show (train/serve
skew); one whose OUTCOME PATH crosses one books an untradeable R (the jump was never tradeable).

THE FIX IS EXCLUSION, NOT DATA REBUILD (2026-07-16 correction): the continuous contracts stay
as-is (correct by design). Splice-tainted CANDIDATES are dropped — bar-level, not year-level —
which keeps the ~95.5%+ of every year that's clean rather than discarding whole quarters around
roll dates. Verified on the pivot-trend strategy (2024+2025): ~4.5% of the pool excluded, the
clean edge SURVIVED AND SHARPENED (65.6%/57.2% WR@4R) — proof the excluded candidates were the
artifact, not the signal.

Usage (any Mantis strategy's `_bars()` + `build()`):
    B['splice_cum'] = splice_prefix_counts(c, atr, jump_atr=8.0)
    ...
    if splice_taints_window(B['splice_cum'], i, seq) or \\
       splice_taints_outcome(B['splice_cum'], i, vert, n):
        continue   # drop this candidate
"""
import numpy as np


def detect_splice_jumps(close, atr, jump_atr=8.0):
    """Boolean mask, True at bar j where |close[j] - close[j-1]| > jump_atr * atr[j] (bar 0 is
    always False — no prior bar to jump from). ATR-relative so the threshold is instrument- and
    regime-scale-free: a normal bar-to-bar move is O(1) ATR; a splice is one contract's price
    landing atop another's, routinely 5-50x ATR. jump_atr=8.0 is the house default (verified: the
    excluded set is exactly the roll-adjacent candidates, not ordinary high-volatility bars)."""
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    n = len(close)
    jump = np.zeros(n, dtype=bool)
    if n < 2:
        return jump
    denom = np.where(atr[1:] > 0, atr[1:], np.inf)         # atr<=0 (warmup) -> never flags
    ratio = np.abs(np.diff(close)) / denom
    jump[1:] = ratio > jump_atr
    return jump


def splice_prefix_counts(close, atr, jump_atr=8.0):
    """Prefix count of splice-jump bars: prefix[j] = number of jumps in bars [0, j]. O(1) range-
    jump-count queries via prefix[hi] - prefix[lo-1] (see splice_taints_window/_outcome). Length
    n+1 (prefix[0] = 0, prefix[n] = total jumps) so callers never need a `max(., 0)` guard on the
    lower bound of a range that starts at bar 0."""
    jump = detect_splice_jumps(close, atr, jump_atr=jump_atr)
    return np.concatenate([[0], np.cumsum(jump)])


def _range_has_jump(prefix_cum, lo, hi):
    """True if any splice-jump bar lies in [lo, hi] inclusive (both bar indices into the ORIGINAL
    close/atr arrays, not the prefix array). prefix_cum[k] = count of jump bars in [0, k-1], so
    the count over [lo, hi] is prefix_cum[hi+1] - prefix_cum[lo]. Clamps bar indices to the
    original series' valid range [0, n-1] so out-of-bounds window/outcome edges (start of series,
    end of series) never index-error."""
    n = len(prefix_cum) - 1                                # original series length
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    if hi < lo:
        return False
    return bool(prefix_cum[hi + 1] - prefix_cum[lo] > 0)


def splice_taints_window(prefix_cum, i, seq):
    """True if the causal [seq] input window ending at bar i (bars [i-seq+1, i]) contains a
    splice jump — the train/serve-skew channel (live single-contract bars never show this)."""
    return _range_has_jump(prefix_cum, i - int(seq) + 1, i)


def splice_taints_outcome(prefix_cum, i, vert, n):
    """True if the forward outcome window from entry (bars [i+1, min(i+1+vert, n-1)]) contains a
    splice jump — the untradeable-R channel (the barrier resolution rides on an unreal price move)."""
    j2 = min(i + 1 + int(vert), n - 1)
    return _range_has_jump(prefix_cum, i + 1, j2)
