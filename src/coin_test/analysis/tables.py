"""Tear sheet class implementation."""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

import pandas as pd

from .graphs import _get_strategy_results
from ..backtest import BacktestResults


class DataframeGenerator(ABC):
    """Generate a pandas DataFrame using a single BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results: BacktestResults) -> pd.DataFrame:
        """Create dataframe."""


class DataframeGeneratorMultiple(ABC):
    """Generate a pandas DataFrame using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Create dataframe."""


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
        index = backtest_results.sim_data.index
        price_series = backtest_results.sim_data.loc[:, "Price"]
        metrics: dict[str, float] = {}

        pct_change = price_series.pct_change().dropna()
        timedelta = index[1] - index[0]  # type: ignore
        per_year = pd.Timedelta(days=365) / timedelta

        mean_return = pct_change.mean()

        metrics["Average Annual Return (%)"] = (mean_return * per_year) * 100

        # Sharpe Ratio
        std_return = pct_change.std()
        metrics["Sharpe Ratio"] = (mean_return * per_year) / (
            std_return * per_year**0.5 + 1e-10
        )

        # Sortino Ratio
        neg_stddev = pct_change[pct_change < 0].std()
        metrics["Sortino Ratio"] = (mean_return * per_year) / (
            neg_stddev * per_year**0.5 + 1e-10
        )

        # Calmar Ratio
        cumilative_returns = (pct_change + 1).cumprod()
        peak = cumilative_returns.expanding(min_periods=1).max()
        draw_downs = (cumilative_returns / peak) - 1
        draw_down = abs(draw_downs.min())
        metrics["Max Drawdown (%)"] = draw_down * 100
        metrics["Calmar Ratio"] = mean_return * per_year / (draw_down + 1e-10)

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
    """Single strategy metrics."""

    name = "Tear Sheet"

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
        cols = {
            "Mean": metrics_df.mean().round(2).astype(str),
            "Standard Deviation": metrics_df.std().round(2).astype(str),
        }
        summary_df = pd.DataFrame.from_dict(cols)
        summary_df = summary_df.set_index(metrics_df.columns)
        return summary_df


class SummaryTearSheet(DataframeGeneratorMultiple):
    """Multiple strategy metrics."""

    name = "Total Tear Sheet"

    @staticmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Generate multiple strategy summary metrics.

        Args:
            backtest_results_list: List of backtest results.

        Returns:
            DataFrame: DataFrame of summary metrics. Columns are metrics and
                rows are strategies.
        """
        strategy_results = _get_strategy_results(backtest_results_list)
        summary_metrics = {}
        for strategy, results in strategy_results.items():
            tear_sheet = TearSheet.create(results)
            summary_metrics[strategy] = (
                tear_sheet["Mean"] + " ± " + tear_sheet["Standard Deviation"]
            )
        tear_sheet = pd.DataFrame.from_dict(summary_metrics, orient="index")
        return tear_sheet
