"""Define the BacktestResults class."""

from typing import Iterable

import pandas as pd

from .portfolio import Portfolio
from .strategy import Strategy
from .trade import Trade
from .trade_request import TradeRequest
from ..data import Composer


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
        self._seed = None
        self._starting_portfolio = starting_portfolio
        self._slippage_type = slippage_calculator_type
        self._tx_fee_type = transaction_fee_calculator_type

        self._data_dict = {ds.metadata: ds.df for (_, ds) in composer.datasets.items()}
        self._strategy_names = [s.name for s in strategies]
        self._sim_data = pd.DataFrame(
            list(zip(sim_data[0], sim_data[1], sim_data[2], sim_data[3], strict=True)),
            columns=["Timestamp", "Portfolios", "Trades", "Pending Trades"],
        )

        self._sim_data.set_index("Timestamp")
