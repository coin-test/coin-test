"""Conftest file for analysis."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from coin_test.backtest import BacktestResults
from coin_test.data import MetaData
from coin_test.util import AssetPair, Side, Ticker


@pytest.fixture
def backtest_results_dataset() -> str:
    """The csv file with the backtest results."""
    return "tests/analysis/assets/backtest_results_data.csv"


def _build_backtest_results(
    asset_data: pd.DataFrame, portfolio_data: str
) -> BacktestResults:
    backtest_results = MagicMock()

    backtest_results.seed = None
    backtest_results.starting_portfolio = Mock()
    backtest_results.slippage_type = Mock()
    backtest_results.strategy_names = ["Mockery"]
    backtest_results.strategy_lookbacks = [pd.Timedelta(days=14)]

    backtest_results.data_dict = {
        MetaData(AssetPair(Ticker("a"), Ticker("b")), "H"): asset_data
    }

    # Create fake sim_data
    dtypes = {
        "Open Time": int,
        "Price": float,
    }

    sim_data_df = pd.read_csv(portfolio_data, dtype=dtypes)  # type: ignore
    index = pd.DatetimeIndex(
        pd.to_datetime(sim_data_df["Open Time"], unit="s", utc=True),
    )
    sim_data_df.set_index(index, inplace=True)
    sim_data_df.drop(columns=["Open Time"], inplace=True)

    mock_col = pd.Series(
        ([MagicMock()] for _ in range(len(sim_data_df))),
        index=sim_data_df.index,
    )
    sim_data_df["Portfolios"] = mock_col
    sim_data_df["Pending Trades"] = mock_col

    mock_buy = Mock()
    mock_buy.side = Side.BUY
    mock_sell = Mock()
    mock_sell.side = Side.SELL
    mock_trades = [mock_buy, mock_buy, mock_sell, mock_sell]
    mock_trades_col = pd.Series(
        (mock_trades for _ in range(len(sim_data_df))),
        index=sim_data_df.index,
    )
    sim_data_df["Trades"] = mock_trades_col

    backtest_results.sim_data = sim_data_df

    return backtest_results


@pytest.fixture
def backtest_results(
    hour_data_indexed_df: pd.DataFrame, backtest_results_dataset: str
) -> BacktestResults:
    """Returns a BacktestResults mocked object."""
    return _build_backtest_results(hour_data_indexed_df, backtest_results_dataset)


@pytest.fixture
def backtest_results_factory(
    hour_data_indexed_df: pd.DataFrame,
    backtest_results_dataset: str,
) -> Callable[[str], BacktestResults]:
    """Returns a BackResults factory function."""

    def _factory(name: str) -> BacktestResults:
        result = _build_backtest_results(hour_data_indexed_df, backtest_results_dataset)
        result.strategy_names = [name]
        return result

    return _factory
