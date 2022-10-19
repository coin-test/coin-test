"""Test the Ticker class."""
import pytest

from coin_test.util import Ticker


def test_ticker() -> None:
    """Initialize correctly."""
    example_symbol = " BtC"

    x = Ticker(symbol=example_symbol)

    assert x.symbol == example_symbol.strip().lower()


def test_invalid_ticker() -> None:
    """Error on bad initialization."""
    example_bad_symbol = " "
    with pytest.raises(ValueError):
        Ticker(symbol=example_bad_symbol)


def test_ticker_equality() -> None:
    """Equivalent tickers compare correctly."""
    ticker_symbol = "BTC"
    ticker2_symbol = " Btc"
    ticker = Ticker(symbol=ticker_symbol)
    ticker2 = Ticker(symbol=ticker2_symbol)

    assert ticker == ticker2
    assert ticker2 == ticker
    assert not (ticker != ticker2)


def test_ticker_inequality() -> None:
    """Inequivalent tickers compare correctly."""
    ticker_symbol = " .BTC"
    ticker2_symbol = " Btc"
    ticker = Ticker(symbol=ticker_symbol)
    ticker2 = Ticker(symbol=ticker2_symbol)

    assert ticker != ticker2
    assert ticker2 != ticker
    assert not (ticker == ticker2)


@pytest.mark.exclude_typeguard
def test_ticker_invalid_comparison() -> None:
    """Errors on non-sensical type."""
    ticker_symbol = " .BTC"
    ticker2_symbol = 5
    ticker = Ticker(symbol=ticker_symbol)

    with pytest.raises(NotImplementedError):
        assert ticker != ticker2_symbol
        assert ticker2_symbol != ticker
