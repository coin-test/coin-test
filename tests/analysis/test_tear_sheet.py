"""Test the tear sheet class."""

from coin_test.analysis import TearSheet
from coin_test.backtest import BacktestResults


def test_single_backtest_tear_sheet(backtest_results: BacktestResults) -> None:
    """Successfully get metrics for a single backtest."""
    metrics = TearSheet.single_backtest_metrics(backtest_results)

    assert (metrics["Sharpe Ratio"] - 3.817) < 1e-3
