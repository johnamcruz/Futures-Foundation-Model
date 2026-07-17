"""CAPABILITY PROBES — the representation report card emitted during SSL training."""
import numpy as np

from futures_foundation.finetune.pretext._torch.capability import probe_battery, format_probes


def test_probe_battery_detects_retained_vs_discarded_information():
    """An embedding that KEEPS a signal probes high; one that DISCARDS it probes ~chance.
    This is the whole instrument: a low retention AUC means the encoder threw the info away."""
    rng = np.random.default_rng(0)
    n = 600
    kept = (rng.random(n) < 0.5)                       # the signal the 'encoder' preserves
    discarded = (rng.random(n) < 0.5)                  # a signal it never encodes
    W = rng.standard_normal((n, 5, 16)).astype(np.float32)

    def embed_fn(w):                                   # dim0 carries `kept` (for the subset given)
        m = len(w)
        e = rng.standard_normal((m, 8)).astype(np.float32) * 0.1
        e[:, 0] = w[:, 0, 0]                           # deterministic passthrough of channel 0
        return e

    W[:, 0, 0] = np.where(kept, 3.0, -3.0)             # kept signal is READABLE from the window
    aucs = probe_battery(embed_fn, W, {'kept': kept, 'discarded': discarded}, max_n=600)
    assert aucs['kept'] > 0.9, aucs                    # retained -> probe finds it
    assert 0.35 < aucs['discarded'] < 0.65, aucs       # discarded -> chance


def test_probe_battery_skips_degenerate_and_formats():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((120, 5, 8)).astype(np.float32)
    out = probe_battery(lambda w: rng.standard_normal((len(w), 4)),
                        W, {'all_true': np.ones(120, bool), 'mixed': rng.random(120) < 0.5},
                        max_n=120)
    assert 'all_true' not in out                       # zero-variance truth -> skipped, no crash
    assert format_probes({'start': 0.7712, 'vol': 0.8841}) == '  probe start=0.771 vol=0.884'
    assert format_probes({}) == ''
