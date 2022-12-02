"""Test the TradeRequest class."""

from statistics import mean

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


def test_correct_slippage_buy(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
) -> None:
    """Price increases correctly on buy."""
    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )

    BASIS_POINT_ADJ = 10

    expected_price = average_price * (1 + BASIS_POINT_ADJ / 10000)

    assert expected_price == MarketTradeRequest._calculate_slippage(
        asset_pair, Side.BUY, timestamp_asset_price
    )


def test_correct_slippage_sell(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
) -> None:
    """Price decreases correctly on sell."""
    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )

    BASIS_POINT_ADJ = 10

    expected_price = average_price * (1 - BASIS_POINT_ADJ / 10000)

    assert expected_price == TradeRequest._calculate_slippage(
        asset_pair, Side.SELL, timestamp_asset_price
    )


def test_correct_transaction_fees() -> None:
    """Properly calculate transaction fees for a Trade."""
    amount = 1000
    adjusted_price = 1.07

    TRANSACTION_FEE_BP = 50
    assert amount * adjusted_price * (
        TRANSACTION_FEE_BP / 10000
    ) == TradeRequest._generate_transaction_fee(amount, adjusted_price)


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

    mocker.patch("coin_test.backtest.TradeRequest._calculate_slippage")
    TradeRequest._calculate_slippage.return_value = trade_price

    mocker.patch("coin_test.backtest.TradeRequest._generate_transaction_fee")
    TradeRequest._generate_transaction_fee.return_value = 0

    trade = trade_request.build_trade(timestamp_asset_price)

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

    mocker.patch("coin_test.backtest.TradeRequest._calculate_slippage")
    TradeRequest._calculate_slippage.return_value = trade_price

    mocker.patch("coin_test.backtest.TradeRequest._generate_transaction_fee")
    TradeRequest._generate_transaction_fee.return_value = 0

    trade = trade_request.build_trade(timestamp_asset_price)

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
