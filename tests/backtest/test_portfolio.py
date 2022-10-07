"""Test the Portfolio object."""

from coin_test.backtest import Portfolio

assets = {"BTC": 2, "ETH": 1.5, "DOGE": 358.25}
cash = 10000.0


def test_portfolio() -> None:
    """Test the initalization of a Portfolio object."""
    global assets, cash

    p = Portfolio(cash, assets)

    assert p.cash == cash
    assert p.assets == assets


def test_portfolio_no_constructor() -> None:
    """Test the default initialization of a Portfolio object."""
    p = Portfolio()

    assert p.cash == 0
    assert p.assets == {}


def test_free_cash() -> None:
    """Test the free cash functionality of a Portfolio object.

    When no money is reserved for shorting, determine
    the amount of cash available to buy assets.
    """
    global assets, cash

    p = Portfolio(cash, assets)

    assert cash == p.get_free_cash()
