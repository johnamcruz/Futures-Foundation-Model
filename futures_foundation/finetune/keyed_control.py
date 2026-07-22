"""Helpers for honest shuffled controls in keyed multi-task classifiers."""
from __future__ import annotations


def shuffle_training_keys(labeler, keys, permutation):
    """Shuffle target-bearing keys while allowing feature identity to remain aligned.

    Most keyed classifiers store only target metadata in each key, so permuting complete keys is
    correct. Multi-stream classifiers may also use a key prefix as train-time sampling identity.
    Such labelers can implement ``shuffle_training_keys(keys, permutation)`` to preserve that
    non-target prefix while permuting every target field. The hook receives no validation or OOS
    rows and therefore cannot alter the control boundary.
    """
    keys = list(keys)
    permutation = list(permutation)
    if len(keys) != len(permutation):
        raise ValueError("shuffle permutation must align with training keys")
    hook = getattr(labeler, "shuffle_training_keys", None)
    shuffled = hook(keys, permutation) if callable(hook) else [keys[i] for i in permutation]
    if len(shuffled) != len(keys):
        raise ValueError("shuffle_training_keys hook returned the wrong row count")
    return list(shuffled)
