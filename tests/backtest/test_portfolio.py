"""Test the Portfolio class."""

from copy import copy
from unittest.mock import Mock, PropertyMock

import pytest

from coin_test.backtest import Portfolio
from coin_test.util import AssetPair, Money, Side, Ticker


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


def _make_mock_trade(
    asset_pair: AssetPair, side: Side, qty: float, price: float, transaction_fee: float
) -> Mock:
    mock_trade = Mock()
    type(mock_trade).asset_pair = PropertyMock(return_value=asset_pair)
    type(mock_trade).side = PropertyMock(return_value=side)
    type(mock_trade).amount = PropertyMock(return_value=qty)
    type(mock_trade).price = PropertyMock(return_value=price)
    type(mock_trade).transaction_fee = PropertyMock(return_value=transaction_fee)
    return mock_trade


def test_adjustment_success_buy(asset_pair: AssetPair) -> None:
    """Adjust a portfolio."""
    transaction_fee = 0.50

    trade = _make_mock_trade(asset_pair, Side.BUY, 10, 10.1, transaction_fee)

    portfolio_assets = {
        Ticker("BTC"): Money(Ticker("BTC"), 1.51),
        Ticker("ETH"): Money(Ticker("ETH"), 2),
        Ticker("USDT"): Money(Ticker("USDT"), 1000),
    }
    portfolio = Portfolio(asset_pair.currency, portfolio_assets)
    copy_portfolio = copy(portfolio)

    adj_portfolio = portfolio.adjust(trade)

    exepcted_base_currency = portfolio.assets[asset_pair.currency]
    exepcted_base_currency.qty -= trade.amount * trade.price + transaction_fee
    exepcted_trade_currency = portfolio.assets[asset_pair.asset]
    exepcted_trade_currency.qty += trade.amount

    assert adj_portfolio is not None
    assert portfolio.assets == copy_portfolio.assets
    assert adj_portfolio is not portfolio
    assert adj_portfolio.assets[asset_pair.currency] == exepcted_base_currency
    assert adj_portfolio.assets[asset_pair.asset] == exepcted_trade_currency


def test_adjustment_failure_buy(assets: dict, asset_pair: AssetPair) -> None:
    """Adjust a portfolio."""
    transaction_fee = 0.5
    trade = _make_mock_trade(asset_pair, Side.BUY, 1000, 10.1, transaction_fee)

    portfolio = Portfolio(asset_pair.currency, assets)
    adj_portfolio = portfolio.adjust(trade)

    assert adj_portfolio is None


def test_adjustment_success_sell(assets: dict, asset_pair: AssetPair) -> None:
    """Adjust a portfolio."""
    transaction_fee = 0.50
    trade = _make_mock_trade(asset_pair, Side.SELL, 1.5, 10, transaction_fee)

    portfolio = Portfolio(asset_pair.currency, assets)
    copy_portfolio = copy(portfolio)
    adj_portfolio = portfolio.adjust(trade)

    exepcted_base_currency = portfolio.assets[asset_pair.currency]
    exepcted_base_currency.qty += trade.amount * trade.price - transaction_fee
    exepcted_trade_currency = portfolio.assets[asset_pair.asset]
    exepcted_trade_currency.qty -= trade.amount

    assert adj_portfolio is not None
    assert portfolio.assets == copy_portfolio.assets
    assert adj_portfolio is not portfolio
    assert adj_portfolio.assets[asset_pair.currency] == exepcted_base_currency
    assert adj_portfolio.assets[asset_pair.asset] == exepcted_trade_currency


def test_adjustment_failure_sell(assets: dict, asset_pair: AssetPair) -> None:
    """Adjust a portfolio."""
    transaction_fee = 0.5
    trade = _make_mock_trade(asset_pair, Side.SELL, 1.52, 10, transaction_fee)

    portfolio = Portfolio(asset_pair.currency, assets)
    adj_portfolio = portfolio.adjust(trade)

    assert adj_portfolio is None


def test_fail_on_transaction_fees(assets: dict, asset_pair: AssetPair) -> None:
    """Fail on an all-in buy trade because of transaction fees."""
    transaction_fee = 0.5
    # buy 1 BTC that costs all of our money
    trade = _make_mock_trade(
        asset_pair, Side.BUY, 1, assets[asset_pair.currency].qty, transaction_fee
    )
    portfolio = Portfolio(asset_pair.currency, assets)
    adj_portfolio = portfolio.adjust(trade)

    assert adj_portfolio is None


def test_fail_on_transaction_fees_sell(assets: dict, asset_pair: AssetPair) -> None:
    """Fail on an all-in buy trade because of transaction fees."""
    # what a rip off
    transaction_fee = 1e5
    # buy 1 BTC that costs all of our money
    trade = _make_mock_trade(asset_pair, Side.SELL, 1, 1, transaction_fee)
    portfolio = Portfolio(asset_pair.currency, assets)
    adj_portfolio = portfolio.adjust(trade)

    assert adj_portfolio is None


def test_portfolio_repr(assets: dict) -> None:
    """Builds string representation."""
    base_currency = Ticker("USDT")
    portfolio = Portfolio(base_currency, assets)
    assert repr(portfolio) == f"{base_currency} - {assets}"
