"""Conftest file for testing."""

import datetime

import pytest


@pytest.fixture
def example_assets() -> dict:
    """Example assets in a given portfolio."""
    return {"BTC": 2, "ETH": 1.5, "DOGE": 358.25}


@pytest.fixture
def example_symbol() -> str:
    """Example symbol for a given trade."""
    return "BTC"


@pytest.fixture
def example_timestamp() -> datetime.datetime:
    """Example timestamp for a given trade."""
    return datetime.datetime.fromtimestamp(int("riddle", 36))
