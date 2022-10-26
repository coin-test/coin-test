"""Test the Money class."""
import pytest

from coin_test.util import Money, Ticker


def test_initialization() -> None:
    """Initialize correctly."""
    example_ticker = Ticker("BTC")
    example_qty = 12.5

    m = Money(example_ticker, example_qty)

    assert m.ticker == example_ticker
    assert m.qty == example_qty


def test_compute_equality() -> None:
    """Accurately tests when two Money objects are equal."""
    example_ticker = Ticker("BTC")
    example_qty = 12.5

    x = Money(example_ticker, example_qty)
    y = Money(example_ticker, example_qty)

    assert x == y
    assert not (x != y)


def test_compute_inequality() -> None:
    """Accurately tests when two Money objects are equal."""
    example_ticker = Ticker("BTC")
    example_qty_1 = 12.5
    example_qty_2 = 500

    x = Money(example_ticker, example_qty_1)
    y = Money(example_ticker, example_qty_2)

    assert x != y
    assert not (x == y)
    assert x < y
    assert y > x
    assert not (x > y)
    assert not (y < x)


def test_add_subtract() -> None:
    """Money can add and subtract properly."""
    example_ticker = Ticker("BTC")
    example_qty_1 = 12.5
    example_qty_2 = 500

    x = Money(example_ticker, example_qty_1)
    y = Money(example_ticker, example_qty_2)

    z = x + y
    a = x - y

    assert z.qty == example_qty_1 + example_qty_2
    assert a.qty == example_qty_1 - example_qty_2


@pytest.mark.exclude_typeguard
def test_invalid_comparison() -> None:
    """Errors on non-sensical type."""
    example_ticker_1 = Ticker("BTC")
    example_ticker_2 = Ticker("USDT")
    example_qty_1 = 12.5
    example_qty_2 = 500

    x = Money(example_ticker_1, example_qty_1)
    y = Money(example_ticker_2, example_qty_2)

    with pytest.raises(NotImplementedError):
        assert x == example_ticker_1

    with pytest.raises(ValueError):
        assert x > y
