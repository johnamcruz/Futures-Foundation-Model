"""Back-compat shim — the embed worker moved to
`futures_foundation._embed_worker` with the foundation-seam promotion.

Kept so any caller still invoking `python -m pipelines.chronos._embed_worker`
(old run scripts, logs, docs) keeps working.
"""
import sys

from futures_foundation._embed_worker import main   # noqa: F401

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
