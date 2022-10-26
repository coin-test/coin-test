"""Define the Simulator class."""

import datetime as dt
from typing import Callable

from .portfolio import Portfolio
from .strategy import Strategy
from .trade_request import TradeRequest
from ..data import Dataset


class Simulator:
    """Manage the simulation of a backtest."""

    def __init__(
        self,
        portfolio: Portfolio,
        symbols: list[str],
    ) -> None:
        """Initialize a Simulator object.

        Args:
            portfolio: The starting portfolio for the backtest,
                ideally only holding cash
            symbols: A list of symbols to trade on
        """
        self.symbols = symbols
        self._portfolio = portfolio
        self._data = []
        self._jobs = []

    def add_strategy(
        self,
        strategy: Callable[[dt.datetime, Portfolio, Dataset], list[TradeRequest]],
        schedule: str,
        lookback: None = None,
    ) -> Strategy:
        """Add a strategy to the simulator.

        Args:
            strategy: The function called to run the strategy
            schedule: A cron string specifying when the strategy should be run
            lookback: The amount of data to send to the strategy function

        Returns:
            Strategy: The object representing this scheduled strategy
        """
        job = Strategy(strategy, schedule, lookback)
        self._jobs.append(job)
        return job

    # def run(self):
    #     pass
