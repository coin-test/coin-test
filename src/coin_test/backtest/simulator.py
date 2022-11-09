"""Define the Simulator class."""

from collections.abc import Iterable
import datetime as dt

from croniter import croniter
import pandas as pd

from .portfolio import Portfolio
from .strategy import Strategy
from .trade import Trade
from .trade_request import TradeRequest
from ..data import Composer
from ..util import AssetPair


class Simulator:
    """Manage the simulation of a backtest."""

    def __init__(
        self,
        composer: Composer,
        starting_portfolio: Portfolio,
        strategies: list[Strategy],
    ) -> None:
        """Initialize a Simulator object.

        Args:
            composer: Data composer for the strategies to use during simulation
            starting_portfolio: The starting portfolio for the backtest,
                ideally only holding cash
            strategies: User Defined strategies to run in the simulation
        """
        self._portfolio = starting_portfolio
        self._composer = composer
        self._strategies = strategies

        self._start_time = composer.start_time
        self._end_time = composer.end_time
        self._simulation_dt = composer.freq

        if not self._validate(composer, strategies):
            ValueError("Strategy uses AssetPair that composer does not")

    @staticmethod
    def _collect_asset_pairs(strategies: list[Strategy]) -> set[AssetPair]:
        """Create asset pair list from strategies."""
        asset_list = []
        for strat in strategies:
            asset_list.extend(strat.asset_pairs)
        return set(asset_list)

    @staticmethod
    def _validate(composer: Composer, strategies: list[Strategy]) -> bool:
        """Validate a simulation can be run."""
        strat_assets = Simulator._collect_asset_pairs(strategies)
        data_assets = set(composer.datasets.keys())
        return len(strat_assets - data_assets) == 0

    @staticmethod
    def _split_pending_orders(
        pending_orders: Iterable[TradeRequest],
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> tuple[list[TradeRequest], list[TradeRequest]]:
        """Split pending orders into executable and remaining orders.

        Args:
            pending_orders: Uncompleted TradeRequests
            current_asset_price: Current timestamp's price by AssetPair

        Returns:
            (executable orders, pending orders)
        """
        remaining_orders = []
        executable_orders = []
        for order in pending_orders:
            # Fulfill existing orders if conditions are met
            if not order.should_execute(
                current_asset_price[order.asset_pair]["Open"].iloc[0]
            ):
                remaining_orders.append(order)
            else:
                executable_orders.append(order)
        return executable_orders, remaining_orders

    @staticmethod
    def _execute_orders(
        portfolio: Portfolio,
        orders: Iterable[TradeRequest],
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> tuple[Portfolio, list[Trade]]:
        """Execute orders by adjusting portfolio.

        Args:
            portfolio: Current Portfolio at given timestamp
            orders: TradeRequests to execute
            current_asset_price: Current timestamp's price by AssetPair

        Returns:
            (updated portfolio, completed Trade objects)
        """
        completed_trades = []
        for order in orders:
            trade = order.build_trade(current_asset_price)
            adjusted_portfolio = portfolio.adjust(trade)

            # Check if adjusting the portfolio failed
            if adjusted_portfolio is None:
                continue
            completed_trades.append(trade)
            portfolio = adjusted_portfolio

        return portfolio, completed_trades

    @staticmethod
    def handle_pending_orders(
        pending_orders: list[TradeRequest],
        current_asset_price: dict[AssetPair, pd.DataFrame],
        portfolio: Portfolio,
    ) -> tuple[list[TradeRequest], Portfolio, list[Trade]]:
        """Process pending orders by adjusting the Portfolio appropriately.

        Args:
            pending_orders: Uncompleted TradeRequests
            current_asset_price: Current timestamp's price by AssetPair
            portfolio: Current Portfolio at given timestamp

        Returns:
            (remaining pending orders, updated portfolio, executed trades)
        """
        executable_orders, pending_orders = Simulator._split_pending_orders(
            pending_orders, current_asset_price
        )
        portfolio, executed_trades = Simulator._execute_orders(
            portfolio, executable_orders, current_asset_price
        )
        return pending_orders, portfolio, executed_trades

    @staticmethod
    def _strategies_to_run(
        schedule: list[tuple[Strategy, croniter]],
        time: dt.datetime,
        simulation_dt: pd.DateOffset,
    ) -> list[Strategy]:
        """Determine which strategies are triggered at a current timestep.

        Args:
            schedule: The list of strategies to run and their croniters
            time: The current timestamp
            simulation_dt: The time interval of the simulation

        Returns:
            list[Strategy]: The strategies to run this timestep
        """
        strategies_to_run = []
        for strat, cron in schedule:
            if cron.get_current(dt.datetime) < time + simulation_dt:
                strategies_to_run.append(strat)
                while cron.get_current(dt.datetime) < time + simulation_dt:
                    cron.get_next(dt.datetime)

        return strategies_to_run

    def run_strategies(
        self,
        schedule: list[tuple[Strategy, croniter]],
        time: pd.Timestamp,
        portfolio: Portfolio,
    ) -> list[TradeRequest]:
        """Create TradeRequests for a given timestamp.

        Args:
            schedule: List of strategies indicating their next run time
            time: Current timestamp used to determine which strategies should run
            portfolio: Current Portfolio at given timestamp

        Returns:
            _type_: _description_
        """
        trade_requests = []
        for strat in self._strategies_to_run(schedule, time, self._simulation_dt):
            lookback_data = self._composer.get_range(
                time - strat.lookback, time, strat.asset_pairs
            )
            trade_requests.extend(strat(time, portfolio, lookback_data))
        return trade_requests

    def run(self) -> tuple[list[Portfolio], list[Trade], list[list[TradeRequest]]]:
        """Run a simulation."""
        schedule = [
            (s, croniter(s.schedule, self._start_time)) for s in self._strategies
        ]

        # hack fix to remove start time from croniter data
        for _, cron in schedule:
            cron.get_next()

        historical_portfolios = [self._portfolio]
        historical_trades: list[Trade] = []
        historical_pending_orders: list[list[TradeRequest]] = [[]]

        # State
        time = self._start_time
        portfolio = self._portfolio
        pending_orders: list[TradeRequest] = []

        while time < self._end_time:
            # Get Timestep data
            current_asset_price = self._composer.get_timestep(time, mask=False)

            pending_orders, portfolio, executed_trades = self.handle_pending_orders(
                pending_orders, current_asset_price, portfolio
            )
            historical_trades.extend(executed_trades)

            trade_requests = self.run_strategies(schedule, time, portfolio)

            remaining_tr, portfolio, executed_trades = self.handle_pending_orders(
                trade_requests, current_asset_price, portfolio
            )

            # TODO: DO these objects need to be copied
            pending_orders.extend(remaining_tr)
            historical_trades.extend(executed_trades)
            historical_portfolios.append(portfolio)
            historical_pending_orders.append(pending_orders)
            time += self._simulation_dt

        return historical_portfolios, historical_trades, historical_pending_orders
