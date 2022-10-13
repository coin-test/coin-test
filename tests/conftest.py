"""Pytest config."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest config."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
