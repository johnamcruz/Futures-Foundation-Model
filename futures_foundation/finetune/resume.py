"""DISCONNECT-RESUME toolkit — ONE reusable implementation for every long run.

Colab (or any remote session) can die mid-run; anything on the runtime's local disk is lost.
Every FFM long-run resumes from durable storage using the SAME primitives here — no per-script
re-implementations:

  sweep (Optuna)   sqlite_snapshot/sqlite_restore — the study DB lives on fast LOCAL disk and is
                   snapshotted (consistent backup, not a raw copy of a live file) to durable
                   storage after every trial; restored on restart.
  walk-forward     KeyedCheckpoint — per-fold results saved (atomic) after EVERY fold, keyed on a
                   CONFIG SIGNATURE so a changed config/universe auto-invalidates; completed folds
                   reload on restart and only the remaining ones run (wf._run_folds).
  produce          durable EMBED CACHE (EMBED_CACHE_DIR on Drive; mantis_frozen bar-cache) makes a
                   re-run cheap (featurize = cache reads, one fast head fit); artifacts write to
                   durable OUTPUT_PATH. No mid-run state needed.
  SSL finetune     BaseTrainer's own progressive best-save + resume (torch checkpoints; the
                   specialized path — atomic via the same convention).

Convention everywhere: write to `<path>.tmp` then os.replace -> a disconnect can never leave a
half-written checkpoint at the final path; unreadable/mismatched state is treated as absent
(fresh start), never an error.
"""
import hashlib
import json
import os

import numpy as np


def config_signature(*parts):
    """Stable short hash of arbitrary config parts (dicts/lists/scalars). The KEY that decides
    whether saved state belongs to THIS run — include everything that changes the results
    (classifier, kwargs, fold layout, seed...); exclude derived noise (standardize stats)."""
    def norm(x):
        if isinstance(x, dict):
            return sorted(((str(k), norm(v)) for k, v in x.items()), key=lambda kv: kv[0])
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    sig = json.dumps(norm(list(parts)), sort_keys=True, default=str)
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def atomic_replace(tmp_path, final_path):
    """os.replace with parent-dir creation — the shared atomic-write tail."""
    os.makedirs(os.path.dirname(str(final_path)) or '.', exist_ok=True)
    os.replace(str(tmp_path), str(final_path))


class KeyedCheckpoint:
    """Config-keyed npz checkpoint: incremental state saved atomically, reloaded ONLY if the
    config signature matches (else treated as absent -> fresh start). State = a dict of lists
    (per-step results); numpy object arrays under the hood so ragged per-fold arrays are fine."""

    def __init__(self, path, key):
        self.path = str(path) if path else None
        self.key = str(key)

    def load(self, default):
        """-> dict of lists (a COPY of `default` filled from disk if key matches)."""
        res = {k: list(v) for k, v in default.items()}
        if not self.path or not os.path.exists(self.path):
            return res
        try:
            d = np.load(self.path, allow_pickle=True)
            if str(d['key']) != self.key:                 # config changed -> start fresh
                return res
            for k in res:
                res[k] = list(d[k])
        except Exception:
            return {k: list(v) for k, v in default.items()}   # unreadable -> fresh
        return res

    @staticmethod
    def _obj(v):
        """1-D object array with each element's native dtype preserved. (A bare
        np.array(v, dtype=object) on uniform-shape elements silently builds a 2-D
        OBJECT matrix -> elements round-trip as object arrays and poison downstream
        concatenates/ufuncs.)"""
        a = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            a[i] = x
        return a

    def save(self, res):
        """Atomic write of the full state (cheap at our sizes; never half-written)."""
        if not self.path:
            return
        tmp = self.path + '.tmp.npz'
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        np.savez(tmp, key=self.key, **{k: self._obj(v) for k, v in res.items()})
        atomic_replace(tmp, self.path)

    def done(self, res, field):
        """How many steps are already banked (len of the given field's list)."""
        return len(res[field])


def sqlite_snapshot(db_path, backup_path):
    """Consistent SQLite backup (sqlite3 backup API, NOT a raw file copy — copying a live DB
    mid-write corrupts it) -> atomic replace at backup_path. No-op if the DB doesn't exist."""
    import sqlite3
    if not backup_path or not os.path.exists(str(db_path)):
        return False
    snap = str(db_path) + '.snap'
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    dst = sqlite3.connect(snap)
    try:
        src.backup(dst)
    finally:
        src.close(); dst.close()
    atomic_replace(snap, backup_path)
    return True


def sqlite_restore(backup_path, db_path):
    """Restore the local working DB from the durable snapshot (only if the local DB is absent —
    an existing local DB is newer by construction). -> True if restored."""
    import shutil
    if not backup_path or os.path.exists(str(db_path)) or not os.path.exists(str(backup_path)):
        return False
    os.makedirs(os.path.dirname(str(db_path)) or '.', exist_ok=True)
    shutil.copy(str(backup_path), str(db_path))
    return True
