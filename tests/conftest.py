"""Pytest config."""


from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest config."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
