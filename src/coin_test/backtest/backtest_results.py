"""Define the BacktestResults class."""

import logging
import os
import pickle
from typing import Iterable

import pandas as pd

from .portfolio import Portfolio
from .strategy import Strategy
from .trade import Trade
from .trade_request import TradeRequest
from ..data import Composer
from ..util import AssetPair


logger = logging.getLogger(__name__)


class BacktestResults:
    """Record the results of a backtest."""

    def __init__(
        self,
        composer: Composer,
        starting_portfolio: Portfolio,
        strategies: Iterable[Strategy],
        sim_data: tuple[
            list[pd.Timestamp],
            list[Portfolio],
            list[list[Trade]],
            list[list[TradeRequest]],
        ],
        slippage_calculator_type: type,
        transaction_fee_calculator_type: type,
    ) -> None:
        """Initialize a BacktestResults object.

        Args:
            composer (Composer): Composer used in BackTest
            starting_portfolio (Portfolio): Initial Portfolio
            strategies (Iterable[Strategy]): Strategies used in the backtest
            sim_data (list[tuple[int]]): Results from a backtest
            slippage_calculator_type (type): Slippage Calculator used
            transaction_fee_calculator_type (type): Tx Fees used
        """
        logger.debug("Generating Backtest Results")

        self.seed = None
        self.starting_portfolio = starting_portfolio
        self.slippage_type = slippage_calculator_type
        self.tx_fee_type = transaction_fee_calculator_type

        self.data_dict = {ds.metadata: ds.df for ds in composer.datasets.values()}
        self.strategy_names = [s.name for s in strategies]
        self.strategy_lookbacks = [s.lookback for s in strategies]
        self.sim_data = pd.DataFrame(
            list(zip(sim_data[0], sim_data[1], sim_data[2], sim_data[3], strict=True)),
            columns=["Timestamp", "Portfolios", "Trades", "Pending Trades"],
        )
        self.sim_data.set_index("Timestamp", inplace=True, drop=True)
        self.sim_data["Price"] = self.create_date_price_df(self.sim_data, composer)

    def save(self, path: str) -> None:
        """Save to disk.

        Args:
            path: Path to save to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def create_date_price_df(sim_data: pd.DataFrame, composer: Composer) -> pd.Series:
        """Create a TimeSeries dataframe for portfolio value over time."""

        def value_func(x: pd.Series) -> float:
            return BacktestResults.value_from_portfolio(
                x.name, x["Portfolios"], composer  # type: ignore
            )

        sim_price_data = sim_data.apply(value_func, axis=1)
        sim_price_data.index.name = None  # Otherwise will retain "Timestamp" name
        return sim_price_data

    @staticmethod
    def value_from_portfolio(t: pd.Timestamp, p: Portfolio, c: Composer) -> float:
        """Get the monetary value of a portfolio."""
        base_currency = p.base_currency
        total = p.available_assets(base_currency).qty
        all_assets = c.get_timestep(t)

        for ticker, money in p.assets.items():
            if ticker == base_currency:
                continue

            asset_pair = AssetPair(ticker, base_currency)
            if len(all_assets[asset_pair]) == 0:
                continue

            conversion = all_assets[asset_pair]["Open"].iloc[0]
            total += conversion * money.qty

        return total

    @staticmethod
    def load(fp: str) -> "BacktestResults":
        """Load BacktestResults from disk.

        Args:
            fp: filepath to pickle file to load from

        Returns:
            BacktestResults: BacktestResults stored at the location

        Raises:
            ValueError: raises ValueError if the specified file path is not a file
        """
        if not os.path.isfile(fp):
            raise ValueError(f"'{fp}' is not a file.")

        with open(fp, "rb") as f:
            return pickle.load(f)
