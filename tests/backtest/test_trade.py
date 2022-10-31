"""Test the Trade class."""

import datetime as dt

from coin_test.backtest import Trade
from coin_test.util import AssetPair, Side


def test_trade(
    asset_pair: AssetPair,
    timestamp: dt.datetime,
) -> None:
    """Initialize correctly."""
    price = 100.0
    side = Side.BUY

    x = Trade(
        asset_pair=asset_pair,
        side=side,
        price=price,
        timestamp=timestamp,
    )

    assert x.asset_pair == asset_pair
    assert x.side == side
    assert x.price == price
    assert x.timestamp == timestamp
