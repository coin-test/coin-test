"""Define the Strategy class."""

import datetime as dt
from typing import Callable

from .portfolio import Portfolio
from .trade_request import TradeRequest
from ..data import Dataset


class Strategy:
    """A class to store a trading strategy run as a scheduled job."""

    def __init__(
        self,
        strategy: Callable[[dt.datetime, Portfolio, Dataset], list[TradeRequest]],
        schedule: str,
        lookback: None = None,
    ) -> None:
        """Initialize a Strategy object.

        Args:
            strategy: The function called to run the strategy
            schedule: A cron string specifying when the strategy should be run
            lookback: The amount of data to send to the strategy function
        """
        self.strategy = strategy
        self.schedule = schedule
        self.lookback = lookback
