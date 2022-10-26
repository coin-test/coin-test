"""Define the Strategy class."""

from abc import ABC, abstractmethod
import datetime as dt

from pandas import DataFrame

from .portfolio import Portfolio
from .trade_request import TradeRequest
from ..util import AssetPair, Side, Ticker, TradeType


class UserDefinedStrategy(ABC):
    """Strategy generates TradeRequests."""

    @abstractmethod
    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
    ) -> list[TradeRequest]:
        """Execute a strategy."""
        pass

    @property
    @abstractmethod
    def used_asset_pairs(self) -> list[AssetPair]:
        """Assetpairs used within the strategy."""
        pass

    @property
    @abstractmethod
    def schedule(self) -> str:
        """Cron string specifying when the strategy should be run."""
        pass

    @property
    @abstractmethod
    def lookback(self) -> dt.datetime:
        """The amount of data to send to the strategy function."""
        pass


class TestStrategy(UserDefinedStrategy):
    """Test Strategy abstract implementation."""

    @property
    def used_asset_pairs(self) -> list[AssetPair]:
        """Asset Pairs used in test strategy."""
        return [AssetPair(Ticker("BTC"), Ticker("USDT"))]

    @property
    # @classmethod
    def schedule(self) -> str:
        """Test schedule."""
        return "test"

    @property
    # @classmethod
    def lookback(self) -> dt.timedelta:
        """Lookback amount."""
        return dt.timedelta(days=1)

    # @classmethod
    def __call__(
        self, time: dt.datetime, portfolio: Portfolio, lookback_data: DataFrame
    ) -> list[TradeRequest]:
        """Execute test strategy."""
        example_asset_pair = AssetPair(Ticker("btc"), Ticker("USDT"))
        example_side = Side.BUY
        example_trade_type = TradeType.MARKET
        example_notional = 1000.0

        x = TradeRequest(
            example_asset_pair, example_side, example_trade_type, example_notional
        )
        return [x]
