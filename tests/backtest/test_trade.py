"""Test the Trade object."""

import datetime

from coin_test.backtest.trade import Trade

symbol = "BTC"
side = "buy"
type_ = "market"
price = 10
timestamp = datetime.datetime.fromtimestamp(int("riddle", 36))


def test_trade() -> None:
    """Test the default Trade object."""
    global symbol, side, type_, price, timestamp

    x = Trade(symbol=symbol, side=side, price=price, timestamp=timestamp)

    assert x.symbol == symbol
    assert x.side == side
    assert x.price == price
    assert x.timestamp == timestamp
