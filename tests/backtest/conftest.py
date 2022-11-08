"""Conftest file for testing."""

import datetime

import pandas as pd
import pytest

from coin_test.util import AssetPair, Money, Ticker


@pytest.fixture
def assets() -> dict:
    """Example assets in a given portfolio."""
    return {
        Ticker("BTC"): Money(Ticker("BTC"), 1.51),
        Ticker("ETH"): Money(Ticker("ETH"), 2),
        Ticker("USDT"): Money(Ticker("USDT"), 10000),
    }


@pytest.fixture
def asset_pair() -> AssetPair:
    """Example symbol for a given trade."""
    return AssetPair(Ticker("BTC"), Ticker("USDT"))


@pytest.fixture
def timestamp() -> datetime.datetime:
    """Example timestamp for a given trade."""
    return datetime.datetime.fromtimestamp(int("riddle", 36))


@pytest.fixture
def timestamp_asset_price(asset_pair: AssetPair) -> dict[AssetPair, pd.DataFrame]:
    """Example asset price dictionary."""
    return {asset_pair: pd.DataFrame()}
