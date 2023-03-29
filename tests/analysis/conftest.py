"""Conftest file for analysis."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from coin_test.backtest import BacktestResults


@pytest.fixture
def backtest_results_dataset() -> str:
    """The csv file with the backtest results."""
    return "tests/analysis/assets/backtest_results_data.csv"


def _build_backtest_results(dataset: str) -> BacktestResults:
    backtest_results = MagicMock()

    backtest_results.seed = None
    backtest_results.starting_portfolio = Mock()
    backtest_results.slippage_type = Mock()
    backtest_results.strategy_names = ["Mockery"]

    # Create fake sim_data
    dtypes = {
        "Open Time": int,
        "Price": float,
    }
    sim_data_df = pd.read_csv(dataset, dtype=dtypes)  # type: ignore

    index = pd.PeriodIndex(
        data=pd.to_datetime(sim_data_df["Open Time"], unit="s", utc=True),
        freq="H",  # type: ignore
    )
    sim_data_df.set_index(index, inplace=True)
    sim_data_df.drop(columns=["Open Time"], inplace=True)

    def make_col(_: pd.Series) -> Mock:
        return Mock()

    sim_data_df["Portfolios"] = sim_data_df.apply(make_col)
    sim_data_df["Trades"] = sim_data_df.apply(make_col)
    sim_data_df["Pending Trades"] = sim_data_df.apply(make_col)

    backtest_results.sim_data = sim_data_df

    return backtest_results


@pytest.fixture
def backtest_results(backtest_results_dataset: str) -> BacktestResults:
    """Returns a BacktestResults mocked object."""
    return _build_backtest_results(backtest_results_dataset)


@pytest.fixture
def backtest_results_factory(
    backtest_results_dataset: str,
) -> Callable[[str], BacktestResults]:
    """Returns a BackResults factory function."""

    def _factory(name: str) -> BacktestResults:
        result = _build_backtest_results(backtest_results_dataset)
        result.strategy_names = [name]
        return result

    return _factory
