"""Test the TradeRequest class."""

from statistics import mean
from unittest.mock import Mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.backtest import (
    LimitTradeRequest,
    MarketTradeRequest,
    StopLimitTradeRequest,
)
from coin_test.backtest.trade_request import TradeRequest
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


def test_slippage_calculator_applied(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
) -> None:
    """Price increases correctly on buy."""
    slippage = 10
    slippage_calculator = Mock()
    slippage_calculator.return_value = slippage

    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )

    expected_price = average_price + slippage

    assert expected_price == MarketTradeRequest._determine_price(
        asset_pair,
        Side.BUY,
        timestamp_asset_price,
        slippage_calculator,  # pyright: ignore
    )
    slippage_calculator.assert_called_with(asset_pair, Side.BUY, timestamp_asset_price)

    assert expected_price == MarketTradeRequest._determine_price(
        asset_pair,
        Side.SELL,
        timestamp_asset_price,
        slippage_calculator,  # pyright: ignore
    )
    slippage_calculator.assert_called_with(asset_pair, Side.SELL, timestamp_asset_price)


def test_market_trade_request_build_trade_notional(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Build Buy Trade with notional correctly."""
    side = Side.BUY
    notional = 1000.0

    trade_request = MarketTradeRequest(asset_pair, side, notional)

    trade_price = 46

    slippage_calculator = Mock()
    mocker.patch("coin_test.backtest.TradeRequest._determine_price")
    TradeRequest._determine_price.return_value = trade_price

    tx_fees = Mock()
    tx_fees.return_value = 0

    trade = trade_request.build_trade(
        timestamp_asset_price, slippage_calculator, tx_fees
    )

    assert trade.side == side
    assert trade.asset_pair == asset_pair
    assert trade.price == trade_price
    assert trade.amount == notional / trade_price
    assert trade.transaction_fee == 0


def test_market_trade_request_build_trade_buy(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Build Sell Trade with quantity correctly."""
    side = Side.SELL
    qty = 1000.0

    trade_request = MarketTradeRequest(asset_pair, side, qty=qty)

    trade_price = 46

    slippage_calculator = Mock()
    mocker.patch("coin_test.backtest.TradeRequest._determine_price")
    TradeRequest._determine_price.return_value = trade_price

    tx_fees = Mock()
    tx_fees.return_value = 0

    trade = trade_request.build_trade(
        timestamp_asset_price, slippage_calculator, tx_fees
    )

    assert trade.side == side
    assert trade.asset_pair == asset_pair
    assert trade.price == trade_price
    assert trade.amount == qty


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
