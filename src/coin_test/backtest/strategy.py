"""Define the Strategy class."""

from abc import ABC, abstractmethod
import datetime as dt
import uuid

from pandas import DataFrame

from .portfolio import Portfolio
from .trade_request import MarketTradeRequest, TradeRequest
from ..util import AssetPair, Money, Side, Ticker


class Strategy(ABC):
    """Strategy generates TradeRequests."""

    def __init__(
        self,
        name: str,
        asset_pairs: list[AssetPair],
        schedule: str,
        lookback: dt.timedelta,
    ) -> None:
        """Initialize a Strategy.

        Args:
            name: The name of the strategy
            asset_pairs: The list of asset pairs the strategy depends on
            schedule: The cron string specifying when the strategy should be run
            lookback: The amount of data to send to the strategy function
        """
        self.name = name
        self.asset_pairs = asset_pairs
        self.schedule = schedule
        self.lookback = lookback
        self.id = uuid.uuid1()

    @abstractmethod
    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
    ) -> list[TradeRequest]:
        """Execute a strategy."""
        pass


class TestStrategy(Strategy):
    """Test Strategy abstract implementation."""

    def __init__(self) -> None:
        """Initialize a TestStrategy object."""
        super().__init__(
            name="Pro Strat",
            asset_pairs=[AssetPair(Ticker("BTC"), Ticker("USDT"))],
            schedule="* * * * *",
            lookback=dt.timedelta(days=5),
        )

    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
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
                asset_pair, Side.SELL, qty=portfolio.available_assets(Ticker("BTC")).qty
            )

        return [x]
