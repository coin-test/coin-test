"""Define the Strategy class."""

from abc import ABC, abstractmethod
import datetime as dt
import uuid

from pandas import DataFrame

from .portfolio import Portfolio
from .trade_request import TradeRequest
from ..util import AssetPair


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
        self,
        time: dt.datetime,
        portfolio: Portfolio,
        lookback_data: dict[AssetPair, DataFrame],
    ) -> list[TradeRequest]:
        """Execute a strategy."""
        pass
