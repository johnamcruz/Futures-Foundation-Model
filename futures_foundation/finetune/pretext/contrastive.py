"""Contrastive regime pretext with causal Kaufman and historical temporal modes.

Production positives use completed-window Kaufman ER: chop windows group together, efficient
up/down trends form directional groups, and ambiguous transition windows keep only their second
augmented view as a positive. The historical temporal-neighborhood mode remains available for
ablation. Both are strategy-agnostic and use no future bars or trading outcomes.
"""
from .base import PretextTask


class ContrastiveTask(PretextTask):
    name, trainer = 'contrastive', 'train_ssl_contrastive'

    def reserve(self, cfg):
        if cfg.get('regime_key', 'temporal') == 'kaufman':
            return int(cfg.get('seq', 64))
        # A positive starts at anchor+delta and itself reads `seq` candles. Reserve the COMPLETE
        # positive window; reserving only delta admits anchors whose positive crosses a stream or
        # temporal-split boundary even though the positive start still looks legal.
        return int(cfg.get('seq', 64)) + max(int(d) for d in cfg.get('pos_deltas', (2, 16, 64)))

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        desc_ok = bool(probe_res.get('descriptive_delta', probe_res['mean_core_delta']) >= -1e-9)
        detail.update({'descriptive_ok': desc_ok})
        return bool(no_collapse and desc_ok), detail
