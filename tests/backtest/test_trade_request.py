"""Test the Trade Request object."""

import pytest

from coin_test.backtest import TradeRequest
from coin_test.util import Side, TradeType


def test_trade_request(
    example_symbol: str,
    example_side: Side,
    example_trade_type: TradeType,
    example_notional: float,
) -> None:
    """Test the default TradeRequest constructor."""
    x = TradeRequest(example_symbol, example_side, example_trade_type, example_notional)

    assert x.symbol == example_symbol
    assert x.side == example_side
    assert x.type_ == example_trade_type
    assert x.notional == example_notional


def test_bad_trade_request(
    example_symbol: str,
    example_side: Side,
    example_trade_type: TradeType,
    example_notional: float,
    example_qty: float,
) -> None:
    """Check that TradeRequest fails with bad parameters."""
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
