"""Contracts for the default, production-relevant FFM test suite."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_suite_excludes_retired_xgboost_tests():
    config = (ROOT / "pytest.ini").read_text()
    assert "legacy_xgboost" in config
    assert 'not legacy_xgboost' in config


def test_precommit_uses_repository_python_not_algotrader_environment():
    hook = (ROOT / ".githooks" / "pre-commit").read_text()
    assert "algoTraderAI" not in hook
    assert '"$REPO_ROOT/.venv/bin/python"' in hook

