"""Test the Trade class."""

from coin_test.backtest import Trade
from coin_test.util import AssetPair, Side


def test_trade(asset_pair: AssetPair) -> None:
    """Initialize correctly."""
    price = 100.0
    side = Side.BUY
    amount = 9.7
    transaction_fee = 0.25

    x = Trade(
        asset_pair=asset_pair,
        side=side,
        price=price,
        amount=amount,
        transaction_fee=transaction_fee,
    )

    assert x.asset_pair == asset_pair
    assert x.side == side
    assert x.price == price
    assert x.amount == amount
    assert x.transaction_fee == transaction_fee


def test_trade_repr_buy(asset_pair: AssetPair) -> None:
    """Build string representation for buy."""
    price = 100.0
    side = Side.BUY
    amount = 9.7
    currency_amount = price * amount
    trade = Trade(asset_pair=asset_pair, side=side, price=price, amount=amount)
    rep = repr(trade)
    expected = f"{currency_amount} {asset_pair.currency} -> {amount} {asset_pair.asset}"
    assert rep == expected


def test_trade_repr_sell(asset_pair: AssetPair) -> None:
    """Build string representation for sell."""
    price = 100.0
    side = Side.SELL
    amount = 9.7
    currency_amount = price * amount
    trade = Trade(asset_pair=asset_pair, side=side, price=price, amount=amount)
    rep = repr(trade)
    expected = f"{amount} {asset_pair.asset} -> {currency_amount} {asset_pair.currency}"
    assert rep == expected
