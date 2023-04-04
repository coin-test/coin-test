"""Functions to build Datapane Locally."""

from typing import Sequence

import datapane as dp

from .graphs import (
    _get_strategy_results,
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    PlotParameters,
    ReturnsHeatmapPlot,
    SignalPricePlot,
)
from .tables import SummaryTearSheet, TearSheet
from ..backtest import BacktestResults


def _build_strategy_page(
    strategy_name: str,
    results: Sequence[BacktestResults],
    plot_params: PlotParameters,
) -> dp.Page:
    """Builds a single strategy page.

    Args:
        strategy_name: Name of the strategy.
        results: Sequence of backtest results produced by the strategy.
        plot_params: Plot parameters.

    Returns:
        dp.Page: Strategy page.
    """
    tear_sheet = TearSheet.create(results)

    confidence_price = ConfidencePricePlot.create(results, plot_params)
    confidence_returns = ConfidenceReturnsPlot.create(results, plot_params)
    returns_heatmap = ReturnsHeatmapPlot.create(results, plot_params)
    signal_price = SignalPricePlot.create(results, plot_params)

    blocks = [
        "# " + strategy_name,
        "## Tables",
        "### Tear Sheet",
        tear_sheet,
        "## Graphs",
        "### Portfolio Value Over Time",
        confidence_price,
        "### Portfolio Return Over Time",
        confidence_returns,
        "### Portfolio Returns vs Dataset Returns",
        returns_heatmap,
        "### Signal Plot",
        signal_price,
    ]
    page = dp.Page(
        title=strategy_name,
        blocks=blocks,
    )
    return page


def _build_strategy_pages(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> list[dp.Page]:
    """Builds the strategy pages.

    Args:
        results: Sequence of backtest results produced by all strategies.
        plot_params: Plot parameters.

    Returns:
        list[dp.Page]: List of all strategy pages.
    """
    strategy_results = _get_strategy_results(results)
    return [
        _build_strategy_page(strategy, result, plot_params)
        for strategy, result in strategy_results.items()
    ]


def _build_home_page(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> dp.Page:
    """Builds the home page.

    Args:
        results: Sequence of backtest results produced by all strategies.
        plot_params: Plot parameters.

    Returns:
        dp.Page: Home page.
    """
    tear_sheet = SummaryTearSheet.create(results)
    blocks = [
        "# Home",
        "## Strategy Metrics",
        "### Tear Sheet",
        tear_sheet,
    ]
    page = dp.Page(title="Home", blocks=blocks)
    return page


def _build_data_page(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> dp.Page:
    """Builds the data page.

    Args:
        results: Sequence of backtest results produced by all strategies.
        plot_params: Plot parameters.

    Returns:
        dp.Page: Data page.
    """
    confidence_graph = ConfidenceDataPlot.create(results, plot_params)
    blocks = [
        "# Data",
        "### Asset Value Over Time",
        confidence_graph,
    ]
    page = dp.Page(title="Data", blocks=blocks)
    return page


def build_datapane(results: Sequence[BacktestResults]) -> None:
    """Build Datapane from large set of results.

    Args:
        results: List of BacktestResults.
    """
    plot_params = PlotParameters()

    page_list = []
    page_list.append(_build_home_page(results, plot_params))
    page_list.append(_build_data_page(results, plot_params))
    page_list.extend(_build_strategy_pages(results, plot_params))
    dashboard = dp.App(blocks=page_list)

    dashboard.save(path="report.html")
