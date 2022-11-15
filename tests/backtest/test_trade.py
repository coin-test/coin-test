"""Test the Trade class."""

from coin_test.backtest import Trade
from coin_test.util import AssetPair, Side


def test_trade(
    asset_pair: AssetPair,
) -> None:
    """Initialize correctly."""
    price = 100.0
    side = Side.BUY
    amount = 9.7

    x = Trade(asset_pair=asset_pair, side=side, price=price, amount=amount)

    assert x.asset_pair == asset_pair
    assert x.side == side
    assert x.price == price
    assert x.amount == amount
