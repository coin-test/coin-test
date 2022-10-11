"""Test the Portfolio object."""

from coin_test.backtest import Portfolio


def test_portfolio(example_cash: float, example_assets: dict) -> None:
    """Test the initalization of a Portfolio object."""
    p = Portfolio(example_cash, example_assets)

    assert p.cash == example_cash
    assert p.assets == example_assets


def test_portfolio_no_constructor() -> None:
    """Test the default initialization of a Portfolio object."""
    p = Portfolio()

    assert p.cash == 0
    assert p.assets == {}


def test_free_cash(example_cash: float, example_assets: dict) -> None:
    """Test the free cash functionality of a Portfolio object with no assets shorted."""
    p = Portfolio(example_cash, example_assets)

    assert example_cash == p.get_free_cash()
