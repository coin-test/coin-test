"""Test the Trade object."""

import datetime

from coin_test.backtest import Trade
from coin_test.util import Side

symbol = "BTC"
side = Side.BUY
price = 10
timestamp = datetime.datetime.fromtimestamp(int("riddle", 36))


def test_trade() -> None:
    """Test the default Trade object."""
    global symbol, side, price, timestamp

    x = Trade(symbol=symbol, side=side, price=price, timestamp=timestamp)

    assert x.symbol == symbol
    assert x.side == side
    assert x.price == price
    assert x.timestamp == timestamp
