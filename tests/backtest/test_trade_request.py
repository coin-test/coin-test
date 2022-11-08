"""Test the TradeRequest class."""

import pandas as pd
import pytest

from coin_test.backtest import (
    LimitTradeRequest,
    MarketTradeRequest,
    StopLimitTradeRequest,
)
from coin_test.util import AssetPair, Side


def test_market_trade_request(asset_pair: AssetPair) -> None:
    """Initialize correctly."""
    side = Side.BUY
    notional = 1000.0

    x = MarketTradeRequest(asset_pair, side, notional)

    assert x.asset_pair == asset_pair
    assert x.side == side
    assert x.notional == notional

    assert x.should_execute(999.99) is True


def test_market_trade_request_build_trade(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Build Trade correctly."""
    side = Side.BUY
    notional = 1000.0

    MarketTradeRequest(asset_pair, side, notional)

    # trade = x.build_trade(time)
    # TODO: Validate trade attributes


def test_limit_trade_request(asset_pair: AssetPair) -> None:
    """Check for a limit order correctly."""
    side_1 = Side.BUY
    side_2 = Side.SELL
    limit_price = 1100
    notional = 1000.0

    above_limit_price = 1101
    below_limit_price = 1099

    x = LimitTradeRequest(asset_pair, side_1, limit_price, notional)
    y = LimitTradeRequest(asset_pair, side_2, limit_price, notional)

    assert x.should_execute(above_limit_price) is False
    assert x.should_execute(below_limit_price) is True

    assert y.should_execute(above_limit_price) is True
    assert y.should_execute(below_limit_price) is False


def test_stop_limit_trade_request(asset_pair: AssetPair) -> None:
    """Check for a stop limit order correctly."""
    side_1 = Side.BUY
    side_2 = Side.SELL
    limit_price = 1100
    notional = 1000.0

    above_limit_price = 1101
    below_limit_price = 1099

    x = StopLimitTradeRequest(asset_pair, side_1, limit_price, notional)
    y = StopLimitTradeRequest(asset_pair, side_2, limit_price, notional)

    assert x.should_execute(above_limit_price) is True
    assert x.should_execute(below_limit_price) is False

    assert y.should_execute(above_limit_price) is False
    assert y.should_execute(below_limit_price) is True


def test_bad_trade_request(asset_pair: AssetPair) -> None:
    """Error when supplying notional and buy argements or neither argument."""
    side = Side.BUY
    notional = 1000.0
    qty = 2.0

    with pytest.raises(ValueError):
        MarketTradeRequest(
            asset_pair,
            side,
            notional,
            qty,
        )

    with pytest.raises(ValueError):
        MarketTradeRequest(asset_pair, side)
