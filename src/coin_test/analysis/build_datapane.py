"""Functions to build Datapane Locally."""

from typing import Sequence, Type

import datapane as dp

from .graphs import (
    _get_strategy_results,
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    DistributionalPlotGenerator,
    PlotParameters,
    ReturnsHeatmapPlot,
)
from .tables import DataframeGeneratorMultiple, SummaryTearSheet, TearSheet
from ..backtest import BacktestResults


STRATEGY_TABLES: list[Type[DataframeGeneratorMultiple]] = [TearSheet]
STRATEGY_GRAPHS: list[Type[DistributionalPlotGenerator]] = [
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    ReturnsHeatmapPlot,
]


def _build_strategy_page(
    strategy_name: str,
    results: Sequence[BacktestResults],
    plot_params: PlotParameters,
) -> dp.Page:
    tear_sheet = TearSheet.create(results)

    confidence_price = ConfidencePricePlot.create(results, plot_params)
    confidence_returns = ConfidenceReturnsPlot.create(results, plot_params)
    returns_heatmap = ReturnsHeatmapPlot.create(results, plot_params)

    blocks = [
        "# " + strategy_name,
        "## Tables",
        tear_sheet,
        "## Graphs",
        confidence_price,
        confidence_returns,
        returns_heatmap,
    ]
    page = dp.Page(
        title=strategy_name,
        blocks=blocks,
    )
    return page


def _build_strategy_pages(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> list[dp.Page]:
    strategy_results = _get_strategy_results(results)
    return [
        _build_strategy_page(strategy, result, plot_params)
        for strategy, result in strategy_results.items()
    ]


def _build_home_page(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> dp.Page:
    tear_sheet = SummaryTearSheet.create(results)
    blocks = [
        "# Home",
        "### Strategy Metrics",
        tear_sheet,
    ]
    page = dp.Page(title="Home", blocks=blocks)
    return page


def _build_data_page(
    results: Sequence[BacktestResults], plot_params: PlotParameters
) -> dp.Page:
    confidence_graph = ConfidenceDataPlot.create(results, plot_params)
    blocks = [
        "# Data",
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
