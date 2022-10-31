"""Define the Simulator class."""

import datetime as dt

from croniter import croniter

from .portfolio import Portfolio
from .strategy import UserDefinedStrategy
from .trade import Trade
from .trade_request import TradeRequest
from ..data import Dataset
from ..util import AssetPair


class Simulator:
    """Manage the simulation of a backtest."""

    @staticmethod
    def collect_asset_pairs() -> list[AssetPair]:
        """Create asset pair list from strategies."""
        return []

    @staticmethod
    def validate() -> bool:
        """Validate a simulation can be run."""
        return True

    def __init__(
        self,
        simulation_data: Dataset,
        starting_portfolio: Portfolio,
        strategies: list[UserDefinedStrategy],
    ) -> None:
        """Initialize a Simulator object.

        Args:
            simulation_data: Data for the strategies to use during simulation
            starting_portfolio: The starting portfolio for the backtest,
                ideally only holding cash
            strategies: User Defined strategies to run in the simulation
        """
        self._portfolio = starting_portfolio
        self._datastore = simulation_data
        self._strategies = strategies

        self._start_time = dt.datetime.now()  # self._datastore.start_time
        self._end_time = dt.datetime.now()  # self._datastore.end_time
        self._simulation_dt = dt.timedelta(days=1)  # self.datacomposer.interval
        # self._timesteps = simulation_data.data[simulation_data.base_asset].
        # ["time"].tolist()

        self._schedule = {
            s: croniter(s.schedule, self._start_time) for s in self._strategies
        }
        self._asset_pairs = Simulator.collect_asset_pairs()

    def strategies_to_run(self, time: dt.datetime) -> list[UserDefinedStrategy]:
        """Determine which strategies are triggered at a current timestep."""
        # TODO: Is this doing a big copy? should we be
        # passing id's of strategies?
        strategies_to_run = []
        for strat, cron in self._schedule.items():
            if time - self._simulation_dt < cron.get_current(dt.datetime) <= time:
                cron.get_next(dt.datetime)
                strategies_to_run.append(strat)
        return strategies_to_run

    def run(self) -> None:
        """Run a simulation."""

        def Execute(TradeRequest: TradeRequest) -> None | Trade:
            """Execute a TradeRequest and update the portfolio."""
            return None

        historical_portfolios = [self._portfolio]
        historical_trades: list[Trade] = []
        historical_pending_orders: list[list[TradeRequest]] = [[]]

        # State
        time = self._start_time
        portfolio = self._portfolio
        pending_orders: list[TradeRequest] = []

        while time <= self._end_time:  # Refactor to use self._timesteps
            # Per-Timestep State
            trade_requests: list[TradeRequest] = []
            pending_requests_to_remove: set[int] = set()

            # Try to execute existing orders
            for index, pending_order in enumerate(pending_orders):
                # Fulfill existing orders if conditions are met
                trade = Execute(pending_order)
                if trade is not None:
                    historical_trades.append(trade)
                    pending_requests_to_remove.add(index)

            # Now remove fulfilled pending requests oding it by index
            # instead of making another list lookup
            pending_orders = [
                elem
                for index, elem in enumerate(pending_orders)
                if index not in pending_requests_to_remove
            ]

            for strategy in self.strategies_to_run(time):
                print(strategy.schedule)
                # trade_requests.extend(self._strategies[_strategy]
                # (time, portfolio, self._datastore.lookback(
                # _strategy.lookpack,))
                pass

            for request in trade_requests:
                # Execute the strategies output and update the portfolio
                trade = Execute(request)
                if trade is not None:
                    historical_trades.append(trade)

            time += self._simulation_dt
            historical_portfolios.append(portfolio)
            historical_pending_orders.append(pending_orders)

        # End of Simulation
        # TODO: Prepare dataframes and build output object
        # return BacktestResults
