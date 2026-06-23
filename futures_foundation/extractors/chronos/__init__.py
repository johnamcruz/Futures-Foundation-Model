"""Chronos-Bolt backbone extractor.

`backbone` holds the frozen Chronos-Bolt embedding seam (subprocess-isolated;
moved here from futures_foundation/foundation.py). `_worker` is its torch
subprocess. `ChronosExtractor` adapts it to the FeatureExtractor interface.
"""
import os

import numpy as np

from . import backbone
from ..base import _windows


class ChronosExtractor:
    name = 'chronos'

    def __init__(self, pool: str = 'mean'):
        self._fb = backbone
        self.pool = pool
        self.ctx = backbone.CTX                          # 128 (CHRONOS_CTX override)
        self.dim = backbone.pooled_dim(pool)             # 256 (mean); +2 if loc_scale

    @classmethod
    def from_pretrained(cls, checkpoint=None, pool: str = 'mean'):
        """HuggingFace-style loader. `checkpoint`:
          None / 'vanilla' / 'frozen' / 'base' -> frozen base (chronos-bolt-tiny)
          a NAME, e.g. 'chronos_bolt_ft'       -> <FFM>/checkpoints/<name>
          a path                               -> used as-is
        Resolves the spec, points the embed worker at it (sets CHRONOS_FT_CKPT)
        and auto-applies the checkpoint's PROVENANCE config (loc_scale), then
        returns a ready extractor. Remote (gh://, HF hub) resolution: future.
        """
        if not checkpoint or str(checkpoint).lower() in ('vanilla', 'frozen', 'base'):
            os.environ.pop('CHRONOS_FT_CKPT', None)
        else:
            os.environ['CHRONOS_FT_CKPT'] = backbone.resolve_ckpt(str(checkpoint))
            backbone._apply_ckpt_config()                # auto loc_scale, etc.
        return cls(pool=pool)

    def embed(self, contexts, batch: int = 64) -> np.ndarray:
        return self._fb.embed(contexts, batch=batch, pool=self.pool)

    def embed_bars(self, close, indices, batch: int = 64) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros((0, self.dim), np.float32)
        return self.embed(_windows(close, indices, self.ctx), batch=batch)


__all__ = ['ChronosExtractor', 'backbone']
