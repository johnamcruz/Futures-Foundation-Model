"""Stage-2.8 pretext: NEXT-LEG + an ordered future adverse/favourable path race.

The graduated ``nextleg`` objective remains untouched. This additive experiment asks a generic,
candle-only question from a confirmed fractal pivot: how much adverse excursion occurs BEFORE the
new leg reaches 25/50/75/100% of its own eventual favourable extent?

Unlike stage-2.7's whole-leg roughness, the curve starts strictly after confirmation and is ordered:
an adverse move after a progress level was reached cannot contaminate that level. It contains no
ATR, entry, stop, target, R multiple, cost, or strategy label. The downstream strategy owns the
mapping from this generic path representation to its pivot-edge-versus-target race.

Design screen (ES+NQ 3min, untouched 2025, 21,551 resolved production pivots): the oracle
25/50/75/100% curve separates +3R-before-pivot-edge with AUC .742/.752/.759/.765. These are
target-validity diagnostics, not checkpoint results; the trained embedding must earn the edge.
"""
import numpy as np

from .nextleg import NextLegTask


RACE_LEVELS = (0.25, 0.50, 0.75, 1.00)


def ordered_adverse_curve(h, l, c, confirm, leg_end, direction,
                          levels=RACE_LEVELS, cap=2.0):
    """Adverse excursion before ordered favourable-progress levels of the future leg.

    The causal reference is ``close[confirm]`` and the first target bar is ``confirm + 1``.
    ``leg_end`` is the next opposite fractal's extreme. Each output is

        max adverse excursion through first reach(q * eventual favourable extent)
        -------------------------------------------------------------------------
                         eventual favourable extent

    Same-bar high/low ambiguity is conservative: adverse excursion on the first-reaching bar is
    included. Returns NaNs when the future leg has no positive extent or is already over at confirm.
    """
    levels = tuple(float(q) for q in levels)
    if not levels or any(not (0.0 < q <= 1.0) for q in levels):
        raise ValueError('levels must be non-empty and lie in (0, 1]')
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError('levels must be strictly increasing')
    confirm, leg_end, direction = int(confirm), int(leg_end), int(direction)
    if leg_end <= confirm:
        return np.full(len(levels), np.nan, np.float32)
    ref = float(c[confirm])
    seg_h = np.asarray(h[confirm + 1:leg_end + 1], np.float64)
    seg_l = np.asarray(l[confirm + 1:leg_end + 1], np.float64)
    if direction == 1:
        favourable, adverse = seg_h - ref, ref - seg_l
    elif direction == -1:
        favourable, adverse = ref - seg_l, seg_h - ref
    else:
        raise ValueError(f'direction must be +/-1, got {direction}')
    extent = float(np.max(favourable)) if len(favourable) else np.nan
    if not (np.isfinite(extent) and extent > 0.0):
        return np.full(len(levels), np.nan, np.float32)
    adverse = np.maximum(adverse, 0.0)
    out = []
    for q in levels:
        reached = np.flatnonzero(favourable >= q * extent)
        if not len(reached):                              # q=1 is reachable by construction
            return np.full(len(levels), np.nan, np.float32)
        mae = float(np.max(adverse[:int(reached[0]) + 1]))
        out.append(min(mae / extent, float(cap)))
    return np.asarray(out, np.float32)


class NextLegRaceTask(NextLegTask):
    name, trainer = 'nextleg_race', 'train_ssl_nextleg_race'

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict['pretext_note'] = ('next-leg + future ordered path race: race_corr must be positive, '
                                   'skill/leg correlations and retention must not regress, then the '
                                   'exact pivot-edge-before-target probe and downstream anchored WF '
                                   'must beat production nextleg')
        return verdict
