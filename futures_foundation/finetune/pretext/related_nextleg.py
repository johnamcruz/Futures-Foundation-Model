"""Opt-in related-series refinement of the leakage-safe NextLeg objective."""

from .nextleg import NextLegTask


class RelatedNextLegTask(NextLegTask):
    """NextLeg targets on the primary stream, with causal related context as evidence."""

    name, trainer = "related_nextleg", "train_ssl_related_nextleg"
    requires_related_series = True

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict["pretext_note"] = (
            "related-series NextLeg experiment: shared compact Mantis encoder plus gated "
            "multi-timeframe/sibling attention; require baseline, shuffled-context, missing-context, "
            "and walk-forward ablations before promotion")
        return verdict


__all__ = ["RelatedNextLegTask"]
