"""Functions to build Datapane Locally."""

from typing import Type

import datapane as dp

from .data_processing import (
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    DataframeGeneratorMultiple,
    DistributionalPlotGenerator,
    PlotParameters,
    ReturnsHeatmapPlot,
)
from .tear_sheet import TearSheet
from ..backtest import BacktestResults


STRATEGY_TABLES: list[Type[DataframeGeneratorMultiple]] = [TearSheet]
STRATEGY_GRAPHS: list[Type[DistributionalPlotGenerator]] = [
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    ReturnsHeatmapPlot,
]


def _flatten_strategies(results: BacktestResults) -> str:
    return "-".join(results.strategy_names)


def _get_strategies(results: list[BacktestResults]) -> set[str]:
    return set(_flatten_strategies(r) for r in results)


def _build_strategy_page(
    results: list[BacktestResults],
    strategy_name: str,
    plot_params: PlotParameters,
) -> dp.Page:
    strategy_results = [r for r in results if _flatten_strategies(r) == strategy_name]

    tables = [(gen.name, gen.create(strategy_results)) for gen in STRATEGY_TABLES]
    graphs = [
        (gen.name, gen.create(strategy_results, plot_params)) for gen in STRATEGY_GRAPHS
    ]

    blocks = []
    for name, table in tables:
        blocks.append("### " + name)
        blocks.append(table)
    for name, graph in graphs:
        blocks.append("### " + name)
        blocks.append(graph)

    page = dp.Page(
        title=strategy_name,
        blocks=blocks,
    )
    return page


# def _build_page(results: list[BacktestResults]) -> dp.Page:
#     """Build a page based on a list of backtest results."""
#     # if len(results) == 1:
#     metrics = DataframeGenerator.create(results[0])
#     page = dp.Page(
#         # title=f"Results{''.join(results[0].strategy_names)} Metrics",
#         blocks=[
#             "### Metrics",
#             dp.Group(metrics, metrics, columns=2),
#         ],
#     )

#     return page
# else:
#     return dp.Page(
#         title="Lol imagine having multi df support",
#         blocks=[
#             "### ...",
#         ],
#     )
# Build cumulative Metrics


# def _build_dataset_page(results: list[BacktestResults]) -> dp.Page:
#     """Build page that contains the real test data/split."""
#     raise NotImplementedError("Missing support for named datasets")


# def _build_home_page() -> dp.Page:
#     return _build_page([Mock()])  # dp.Page(dp.Text("Coin-test Dashboard"))


def build_datapane(results: list[BacktestResults]) -> None:
    """Build Datapane from large set of results.

    Args:
        results: List of BacktestResults.
    """
    plot_params = PlotParameters()

    strategies = _get_strategies(results)
    strategy_pages = [_build_strategy_page(results, s, plot_params) for s in strategies]

    page_list = strategy_pages
    dashboard = dp.App(blocks=page_list)

    dashboard.save(path="report.html")
