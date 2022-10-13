"""Test the TradeRequest class."""

import pytest

from coin_test.backtest import TradeRequest
from coin_test.util import Side, TradeType


def test_trade_request(example_symbol: str) -> None:
    """Initialize correctly."""
    example_side = Side.BUY
    example_trade_type = TradeType.MARKET
    example_notional = 1000.0

    x = TradeRequest(example_symbol, example_side, example_trade_type, example_notional)

    assert x.symbol == example_symbol
    assert x.side == example_side
    assert x.type_ == example_trade_type
    assert x.notional == example_notional


def test_bad_trade_request(example_symbol: str) -> None:
    """Error when supplying notional and buy argements or neither argument."""
    example_side = Side.BUY
    example_trade_type = TradeType.MARKET
    example_notional = 1000.0
    example_qty = 2.0

    with pytest.raises(ValueError):
        TradeRequest(
            example_symbol,
            example_side,
            example_trade_type,
            example_notional,
            example_qty,
        )

    with pytest.raises(ValueError):
        TradeRequest(example_symbol, example_side, example_trade_type)
