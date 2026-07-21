"""The public Probe Atlas launcher preserves the private strategy-IP boundary."""
from pathlib import Path


def test_probe_atlas_public_entrypoint_delegates_to_private_implementation():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts' / 'probe_atlas.py').read_text()
    assert 'colabs" / "mantis" / "probe_atlas.py' in source
    assert 'runpy.run_path' in source
