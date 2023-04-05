"""Test the Strategy class."""
import datetime as dt

import pandas as pd
import pytest

from coin_test.backtest import (
    MarketTradeRequest,
    Portfolio,
    Strategy,
    TradeRequest,
)
from coin_test.util import AssetPair, Money, Side, Ticker


def test_strategy_valid(assets: dict, asset_pair: AssetPair) -> None:
    """Implement correctly."""

    class TestStrategy(Strategy):
        def __init__(self) -> None:
            """Initialize a TestStrategy object."""
            super().__init__(
                name="Pro Strat",
                asset_pairs=[asset_pair],
                schedule="* * * * *",
                lookback=pd.Timedelta(days=5),
            )

        def __call__(
            self, time: dt.datetime, portfolio: Portfolio, lookback_data: pd.DataFrame
        ) -> list[TradeRequest]:
            """Execute test strategy."""
            asset_pair = self.asset_pairs[0]

            if portfolio.available_assets(Ticker("BTC")) == Money(Ticker("BTC"), 0):
                # if no holdings in bitcoin, go all in
                x = MarketTradeRequest(
                    asset_pair,
                    Side.BUY,
                    notional=portfolio.available_assets(Ticker("USDT")).qty,
                )
            else:
                # otherwise sell all bitcoin holdings
                x = MarketTradeRequest(
                    asset_pair,
                    Side.SELL,
                    qty=portfolio.available_assets(Ticker("BTC")).qty,
                )
            return [x]

    test_strategy = TestStrategy()

    assert test_strategy.name == "Pro Strat"
    assert test_strategy.asset_pairs[0] == asset_pair
    assert test_strategy.schedule == "* * * * *"
    assert test_strategy.lookback == pd.Timedelta(days=5)

    portfolio = Portfolio(asset_pair.currency, assets)

    result = test_strategy(
        dt.datetime.fromtimestamp(int("runrun", 36)), portfolio, pd.DataFrame()
    )

    assert result != []


def test_strategy_invalid(asset_pair: AssetPair) -> None:
    """Error on missing methods."""

    class TestStrategyPartial(Strategy):
        def __init__(self) -> None:
            """Initialize a TestStrategy object."""
            super().__init__(
                name="Pro Strat",
                asset_pairs=[asset_pair],
                schedule="* * * * *",
                lookback=pd.Timedelta(days=5),
            )

    with pytest.raises(TypeError):
        TestStrategyPartial()  # type: ignore
