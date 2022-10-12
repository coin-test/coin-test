"""Test the Trade class."""

import datetime

from coin_test.backtest import Trade
from coin_test.util import Side


def test_trade(
    example_symbol: str,
    example_timestamp: datetime.datetime,
) -> None:
    """Initialize correctly."""
    example_price = 100.0
    example_side = Side.BUY

    x = Trade(
        symbol=example_symbol,
        side=example_side,
        price=example_price,
        timestamp=example_timestamp,
    )

    assert x.symbol == example_symbol
    assert x.side == example_side
    assert x.price == example_price
    assert x.timestamp == example_timestamp
