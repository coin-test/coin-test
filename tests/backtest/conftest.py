"""Conftest file for testing."""

import datetime

import pytest

from coin_test.util import Side, TradeType


@pytest.fixture
def example_assets() -> dict:
    """Example assets in a given portfolio."""
    return {"BTC": 2, "ETH": 1.5, "DOGE": 358.25}


@pytest.fixture
def example_cash() -> float:
    """Example cash in a given portfolio."""
    return 10000.0


@pytest.fixture
def example_notional() -> float:
    """Example notional value (total monetary value) for a given trade."""
    return 100.0


@pytest.fixture
def example_price() -> float:
    """Example price for a given trade."""
    return 10.0


@pytest.fixture
def example_qty() -> float:
    """Example quantity for a given trade."""
    return 2


@pytest.fixture
def example_side() -> Side:
    """Example side for a given trade."""
    return Side.BUY


@pytest.fixture
def example_symbol() -> str:
    """Example symbol for a given trade."""
    return "BTC"


@pytest.fixture
def example_timestamp() -> datetime.datetime:
    """Example timestamp for a given trade."""
    return datetime.datetime.fromtimestamp(int("riddle", 36))


@pytest.fixture
def example_trade_type() -> TradeType:
    """Example trade type for a given trade."""
    return TradeType.MARKET
