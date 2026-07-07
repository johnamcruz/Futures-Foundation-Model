"""SpanBERT-style SPAN masking — contiguous multi-bar corruption (torch-free, unit-testable).

Random single-bar masking lets the model interpolate a hole from its neighbors — a LOCAL skill.
Masking a CONTIGUOUS RUN of bars forces it to model how a MOVE DEVELOPS across bars — the span
phenomenon trend formation actually is (SpanBERT's win on span reasoning, mapped to candles).
Used by span-ELECTRA: the generator must fake a plausible multi-bar sequence (much harder) and
the encoder must detect the fake span (much richer signal than single-bar tells).

Span lengths ~ Geometric(1/mean_span) clipped to max_span (the SpanBERT recipe); spans are
sampled until ~ratio of the window is covered. Adjacent/overlapping spans may merge into longer
runs — that's fine (SpanBERT's do too); max_span clips the SAMPLED span, not the merged run.
"""
import numpy as np


def sample_span_mask(rng, batch, seq, ratio, mean_span=4.0, max_span=10):
    """Per-sample contiguous-span mask [batch, seq] bool covering ~ratio of each row (>=1 bar
    always). rng = np.random.Generator (deterministic). mean_span = geometric mean span length;
    max_span clips each sampled span."""
    target = max(1, int(round(ratio * seq)))
    p = 1.0 / max(mean_span, 1.0)
    m = np.zeros((batch, seq), bool)
    for b in range(batch):
        covered = 0
        for _ in range(64):                                # guard: never spin forever
            L = int(min(max_span, max(1, rng.geometric(p))))
            s = int(rng.integers(0, seq))
            e = min(seq, s + L)
            covered += int(np.count_nonzero(~m[b, s:e]))
            m[b, s:e] = True
            if covered >= target:
                break
        if not m[b].any():                                 # unreachable, but keep the invariant
            m[b, 0] = True
    return m
