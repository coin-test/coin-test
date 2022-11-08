"""Test the Portfolio class."""

from unittest.mock import Mock, PropertyMock

import pytest

from coin_test.backtest import Portfolio
from coin_test.util import Money, Ticker
from coin_test.util.ticker import AssetPair


def test_portfolio(assets: dict) -> None:
    """Initialize correctly."""
    base_currency = Ticker("USDT")

    p = Portfolio(base_currency, assets)

    assert p.base_currency == base_currency
    assert p.assets == assets


def test_no_base_currency(assets: dict) -> None:
    """Fail if there is no base currency in the portfolio."""
    base_currency = Ticker("DOGE")  # not in assets

    with pytest.raises(ValueError):
        Portfolio(base_currency, assets)


def test_bad_assets() -> None:
    """Fail if assets have the wrong tickers."""
    base_currency = Ticker("USDT")
    assets = {Ticker("USDT"): Money(Ticker("BTC"), 100)}

    with pytest.raises(ValueError):
        Portfolio(base_currency, assets)


def test_no_reserved_assets_on_init(assets: dict) -> None:
    """Return free cash property."""
    base_currency = Ticker("USDT")

    p = Portfolio(base_currency, assets)

    assert Money(Ticker("USDT"), 0) == p.reserved[Ticker("USDT")]


def test_available_assets(assets: dict) -> None:
    """Return correct amount of free cash."""
    base_currency = Ticker("USDT")

    p = Portfolio(base_currency, assets)

    assert Money(Ticker("USDT"), 10000) == p.available_assets(Ticker("USDT"))


def test_for_wrong_asset(assets: dict) -> None:
    """Error on checking for free cash with an invalid ticker."""
    base_currency = Ticker("USDT")

    p = Portfolio(base_currency, assets)

    with pytest.raises(ValueError):
        p.available_assets(Ticker("DOGE"))


# def _mock_dataset(
#     df: pd.DataFrame | None, metadata: MetaData | None
# ) -> tuple[Mock, PropertyMock, PropertyMock]:
#     dataset = Mock()
#     df_mock = PropertyMock(return_value=df)
#     metadata_mock = PropertyMock(return_value=metadata)
#     type(dataset).df = df_mock
#     type(dataset).metadata = metadata_mock
#     return dataset, df_mock, metadata_mock


def test_adjustment_success_buy(assets: dict, asset_pair: AssetPair) -> None:
    """Adjust a portfolio."""
    trade = Mock()
    type(trade).asset_pair = PropertyMock(return_value=asset_pair)

    # portfolio.adjust(trade)
    # type(trade).side =
    # TODO: Create portfolio based on the assets. Use a mocked trade
