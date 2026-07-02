"""Stage-3 pretext v2: FORWARD trend-vs-chop contrastive — multi-positive InfoNCE grouped by the
FUTURE window's direction x path-efficiency key (self-supervised, anti-shortcut: the key is
target-side, not computable from the input — v1's trailing-slope key was, and washed). Reserves
context + the key's future horizon (same leak discipline as the forecast pretext). Gate is
report-only (descriptive content doesn't regress + no collapse); the REAL gate = trend-AUC +
decile spread (watch the BOTTOM decile = chop-filter quality) + WR@3R vs the stage-2 baseline,
judged offline on the one-shot 2026. Fallback = stage-2."""
from .base import PretextTask


class ContrastiveTask(PretextTask):
    name, trainer = 'contrastive', 'train_ssl_contrastive'

    def reserve(self, cfg):
        return (max(int(x) for x in cfg['context_lengths'])
                + int(cfg.get('contrast_horizon', 25)))             # ctx + FUTURE key horizon

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        desc_ok = bool(probe_res.get('descriptive_delta', probe_res['mean_core_delta']) >= -1e-9)
        detail.update({'descriptive_ok': desc_ok})
        return bool(no_collapse and desc_ok), detail
