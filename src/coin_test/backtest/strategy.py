"""Define the Strategy class."""

from abc import ABC, abstractmethod
import datetime as dt

from pandas import DataFrame

from .portfolio import Portfolio
from .trade_request import MarketTradeRequest, TradeRequest
from ..util import AssetPair, Side, Ticker


class UserDefinedStrategy(ABC):
    """Strategy generates TradeRequests."""

    def __init__(
        self,
        name: str,
        used_asset_pairs: list[AssetPair],
        schedule: str,
        lookback: dt.timedelta,
    ) -> None:
        """Initialize a Strategy.

        Args:
            name: The name of the strategy
            used_asset_pairs: The list of asset pairs the strategy depends on
            schedule: The cron string specifying when the strategy should be run
            lookback: The amount of data to send to the strategy function
        """
        self.name = name
        self.used_asset_pairs = used_asset_pairs
        self.schedule = schedule
        self.lookback = lookback

    @abstractmethod
    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
    ) -> list[TradeRequest]:
        """Execute a strategy."""
        pass


class TestStrategy(UserDefinedStrategy):
    """Test Strategy abstract implementation."""

    def __init__(self) -> None:
        """Initialize a TestStrategy object."""
        super().__init__(
            "Pro Strat",
            [AssetPair(Ticker("BTC"), Ticker("USDT"))],
            "* * * * *",
            dt.timedelta(days=5),
        )

    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
    ) -> list[TradeRequest]:
        """Execute test strategy."""
        example_asset_pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
        example_side = Side.BUY
        example_notional = 1000.0

        x = MarketTradeRequest(example_asset_pair, example_side, example_notional)
        return [x]
