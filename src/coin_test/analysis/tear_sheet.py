"""Tear sheet class implementation."""
from collections import defaultdict
from typing import Sequence

import pandas as pd

from .data_processing import DataframeGeneratorMultiple
from ..backtest import BacktestResults


class MetricsGenerator(DataframeGeneratorMultiple):
    """Raw metrics generator."""

    @staticmethod
    def _single_metrics(backtest_results: BacktestResults) -> dict:
        """Calculate metrics for a single backtest.

        Args:
            backtest_results (BacktestResults): Results of the backtest

        Returns:
            dict: dictionary of metrics
        """
        price_series = backtest_results.sim_data.loc[:, "Price"]
        metrics: dict[str, float] = {}

        # General metrics
        pct_change = price_series.pct_change().dropna()
        timedelta = price_series.index[1] - price_series.index[0]  # type: ignore
        per_year = pd.Timedelta(days=365) / timedelta

        # Sharpe Ratio
        mean_return = pct_change.mean()
        std_return = pct_change.std()

        metrics["Sharpe Ratio"] = (mean_return * per_year) / (
            std_return * per_year**0.5
        )

        return metrics

    @staticmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Generate raw backtest metrics.

        Args:
            backtest_results_list: List of backtest results.

        Returns:
            DataFrame: DataFrame of backtest metrics. Columns are metrics and
            rows are backtests.
        """
        all_metrics: defaultdict[str, list[float]] = defaultdict(list)
        for results in backtest_results_list:
            results_metrics = MetricsGenerator._single_metrics(results)
            for name, metric in results_metrics.items():
                all_metrics[name].append(metric)
        return pd.DataFrame.from_dict(all_metrics)


class TearSheet(DataframeGeneratorMultiple):
    """Summary metrics generator."""

    @staticmethod
    def _summary_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
        means = metrics_df.mean()
        stds = metrics_df.std()
        summary_df = pd.DataFrame(
            [means, stds],
            columns=["Mean", "Standard Deviation"],
            index=[col_name for col_name in metrics_df.columns],
        )
        return summary_df

    @staticmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Generate summar backtest metrics.

        Args:
            backtest_results_list: List of backtest results.

        Returns:
            DataFrame: DataFrame of summary metrics. Columns are "Mean" and
            "Standard Deviation" and rows are metrics.
        """
        metrics_df = MetricsGenerator.create(backtest_results_list)
        summary_df = TearSheet._summary_metrics(metrics_df)
        return summary_df
