"""Define the Simulator class."""

import datetime as dt

from .portfolio import Portfolio
from .strategy import UserDefinedStrategy
from .trade import Trade
from .trade_request import TradeRequest
from ..data import Dataset
from ..util import AssetPair

# from .trade_request import TradeRequest


class Simulator:
    """Manage the simulation of a backtest."""

    @staticmethod
    def build_schedule_object() -> str:
        """Create schedule from strategies."""
        return "Not implemented"

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
        strategies: dict[str, UserDefinedStrategy],  # TODO: This should be an name type
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

        self._schedule = Simulator.build_schedule_object()
        self._asset_pairs = Simulator.collect_asset_pairs()

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

        while time <= self._end_time:
            # Per-Timestep State
            trade_requests: list[TradeRequest] = []
            pending_requests_to_remove: set[int] = set()

            if (
                (time - self._start_time) % dt.timedelta(days=2)
            ).days == 0:  # TODO: Change to a scheduled aspect
                for (
                    _strategy
                ) in self._strategies:  # TODO: onle run strategy indicated by chron
                    trade_requests = []  # self._strategies[_strategy](time, portfolio,
                    # self._datastore.lookback(_strategy.lookpack,)

            for request in trade_requests:
                # Execute the strategies output and update the portfolio
                trade = Execute(request)
                if trade is not None:
                    historical_trades.append(trade)

            for index, pending_order in enumerate(pending_orders):
                # Fulfill existing orders if conditions are met
                trade = Execute(pending_order)
                if trade is not None:
                    historical_trades.append(trade)
                    pending_requests_to_remove.add(index)
                pass

            # Now remove fulfilled pending requests oding it by index
            # instead of making another list lookup
            pending_orders = [
                elem
                for index, elem in enumerate(pending_orders)
                if index not in pending_requests_to_remove
            ]

            time += self._simulation_dt
            historical_portfolios.append(portfolio)
            historical_pending_orders.append(pending_orders)
