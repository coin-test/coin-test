"""Test the Portfolio class."""

from coin_test.backtest import Portfolio


def test_portfolio(example_assets: dict) -> None:
    """Initialize correctly."""
    example_cash = 10000.0

    p = Portfolio(example_cash, example_assets)

    assert p.cash == example_cash
    assert p.assets == example_assets


def test_portfolio_no_constructor() -> None:
    """Initialize correctly on an empty constructor."""
    p = Portfolio()

    assert p.cash == 0
    assert p.assets == {}


def test_free_cash(example_assets: dict) -> None:
    """Return free cash property."""
    example_cash = 10000.0

    p = Portfolio(example_cash, example_assets)

    assert example_cash == p.free_cash
