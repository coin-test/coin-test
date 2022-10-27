"""Test the Portfolio class."""

import pytest

from coin_test.backtest import Portfolio
from coin_test.util import Money, Ticker


def test_portfolio(example_assets: dict) -> None:
    """Initialize correctly."""
    example_base_currency = Ticker("USDT")

    p = Portfolio(example_base_currency, example_assets)

    assert p.base_currency == example_base_currency
    assert p.assets == example_assets


def test_no_base_currency(example_assets: dict) -> None:
    """Fail if there is no base currency in the portfolio."""
    example_base_currency = Ticker("DOGE")  # not in assets

    with pytest.raises(ValueError):
        Portfolio(example_base_currency, example_assets)


def test_bad_assets() -> None:
    """Fail if assets have the wrong tickers."""
    example_base_currency = Ticker("USDT")
    example_assets = {Ticker("USDT"): Money(Ticker("BTC"), 100)}

    with pytest.raises(ValueError):
        Portfolio(example_base_currency, example_assets)


def test_no_reserved_assets_on_init(example_assets: dict) -> None:
    """Return free cash property."""
    example_base_currency = Ticker("USDT")

    p = Portfolio(example_base_currency, example_assets)

    assert Money(Ticker("USDT"), 0) == p.reserved[Ticker("USDT")]


def test_available_assets(example_assets: dict) -> None:
    """Return correct amount of free cash."""
    example_base_currency = Ticker("USDT")

    p = Portfolio(example_base_currency, example_assets)

    assert Money(Ticker("USDT"), 10000) == p.available_assets(Ticker("USDT"))


def test_for_wrong_asset(example_assets: dict) -> None:
    """Error on checking for free cash with an invalid ticker."""
    example_base_currency = Ticker("USDT")

    p = Portfolio(example_base_currency, example_assets)

    with pytest.raises(ValueError):
        p.available_assets(Ticker("DOGE"))
