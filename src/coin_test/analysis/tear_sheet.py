"""Tear sheet class implementation."""

import pandas as pd

from ..backtest import BacktestResults


class TearSheet:
    """A tear sheet."""

    @staticmethod
    def single_backtest_metrics(backtest_results: BacktestResults) -> dict:
        """Calculate metrics for a single backtest.

        Args:
            backtest_results (BacktestResults): Results of the backtest

        Returns:
            dict: dictionary of metrics
        """
        price_series = backtest_results.sim_data.loc[:, "Price"]
        metrics: dict[str, float] = {}

        print(price_series)
        # General metrics
        pct_change = price_series.pct_change().dropna()
        print(pct_change)
        timedelta = price_series.index[1] - price_series.index[0]  # type: ignore
        per_year = pd.Timedelta(days=365) / timedelta

        # Sharpe Ratio
        mean_return = pct_change.mean()
        std_return = pct_change.std()

        metrics["Sharpe Ratio"] = (mean_return * per_year) / (
            std_return * per_year**0.5
        )

        return metrics
