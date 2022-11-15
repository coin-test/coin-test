"""Conftest file for testing."""

import datetime as dt
from unittest.mock import Mock

from croniter import croniter
import pandas as pd
import pytest

from coin_test.backtest import Strategy
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
def asset_pair_eth_usdt() -> AssetPair:
    """Example symbol for a given trade."""
    return AssetPair(Ticker("ETH"), Ticker("USDT"))


@pytest.fixture
def asset_pair_btc_eth() -> AssetPair:
    """Example symbol for a given trade."""
    return AssetPair(Ticker("BTC"), Ticker("ETH"))


@pytest.fixture
def timestamp() -> dt.datetime:
    """Example timestamp for a given trade."""
    return dt.datetime.fromtimestamp(int("riddle", 36))


@pytest.fixture
def timestamp_asset_price(asset_pair: AssetPair) -> dict[AssetPair, pd.DataFrame]:
    """Example asset price dictionary."""
    column_names = ["High", "Low", "Open", "Close"]
    data = [[5.0, 1.0, 2.0, 3.0]]
    return {asset_pair: pd.DataFrame(data=data, columns=column_names)}


@pytest.fixture
def schedule(timestamp: dt.datetime) -> list[tuple[Strategy, croniter]]:
    """Example schedule for running strategies."""
    strat_1 = Mock()
    cron_1 = croniter("* * * * *", start_time=timestamp)  # every minute
    cron_1.get_next()
    strat_2 = Mock()
    cron_2 = croniter("0 * * * *", start_time=timestamp)  # hourly
    cron_2.get_next()
    strat_3 = Mock()
    cron_3 = croniter("0 0 * * *", start_time=timestamp)  # daily
    cron_3.get_next()
    return [(strat_1, cron_1), (strat_2, cron_2), (strat_3, cron_3)]
