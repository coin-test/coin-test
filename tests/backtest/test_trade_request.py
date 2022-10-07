"""Test the Trade Request object."""

import pytest

from coin_test.backtest import TradeRequest

symbol = "BTC"
side = "buy"
type_ = "market"
notional = 10  # buy $10 worth of Bitcoin
qty = 2  # buy 2 Bitcoin


def test_trade_request() -> None:
    """Test the default TradeRequest constructor."""
    global symbol, side, type_, notional

    x = TradeRequest(symbol, side, type_, notional)

    assert x.symbol == symbol
    assert x.side == side
    assert x.type_ == type_
    assert x.notional == notional


def test_bad_trade_request() -> None:
    """Test the default TradeRequest constructor with bad parameters.

    Only one of notional and qty must be specified.
    """
    global symbol, side, type_, notional, qty

    with pytest.raises(ValueError):
        TradeRequest(symbol, side, type_, notional, qty)

    with pytest.raises(ValueError):
        TradeRequest(symbol, side, type_)
