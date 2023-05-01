"""Version test."""


import coin_test


def test_version() -> None:
    """It has the current version."""
    assert coin_test.__version__ == "0.2.0"
