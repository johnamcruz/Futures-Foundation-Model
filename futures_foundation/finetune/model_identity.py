"""Immutable public-base identity and lightweight evaluation provenance."""
from importlib import metadata
import platform
import sys


MANTIS_MODEL_ID = "paris-noah/Mantis-8M"
MANTIS_MODEL_REVISION = "93a16a52a5e2e6d76c0b823533b5836dd83ca10a"


def evaluation_environment():
    """Versions that can change frozen embeddings or linear-probe scores."""
    packages = {}
    for name in ("torch", "mantis-tsfm", "numpy", "scikit-learn", "huggingface-hub"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "mantis_model_id": MANTIS_MODEL_ID,
        "mantis_model_revision": MANTIS_MODEL_REVISION,
        "packages": packages,
    }
