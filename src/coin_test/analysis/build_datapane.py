"""Functions to build Datapane Locally."""

from typing import Sequence, Type

import datapane as dp
import pandas as pd

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


def _get_strategies(results: Sequence[BacktestResults]) -> list[str]:
    return sorted(list(set(_flatten_strategies(r) for r in results)))


def _get_strategy_results(
    results: Sequence[BacktestResults],
) -> dict[str, list[BacktestResults]]:
    strategies = _get_strategies(results)
    strategy_results = {}
    for strategy in strategies:
        strategy_results[strategy] = []
    for result in results:
        strategy = _flatten_strategies(result)
        strategy_results[strategy].append(result)
    return strategy_results


def _build_strategy_page(
    strategy_name: str,
    results: list[BacktestResults],
    plot_params: PlotParameters,
) -> dp.Page:
    tables = [(gen.name, gen.create(results)) for gen in STRATEGY_TABLES]
    graphs = [(gen.name, gen.create(results, plot_params)) for gen in STRATEGY_GRAPHS]

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


def _build_home_page(
    strategy_results: dict[str, list[BacktestResults]], plot_params: PlotParameters
) -> dp.Page:
    metrics = {}
    for strategy, results in strategy_results.items():
        tear_sheet = TearSheet.create(results)
        mean = tear_sheet["Mean"].round(2).astype(str)
        std = tear_sheet["Standard Deviation"].round(2).astype(str)
        metrics[strategy] = mean + " ± " + std
    tear_sheet = pd.DataFrame.from_dict(metrics).transpose()

    page = dp.Page(
        title="Home",
        blocks=["### Tear Sheet", tear_sheet],
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

    strategy_results = _get_strategy_results(results)
    page_list = [_build_home_page(strategy_results, plot_params)]
    page_list += [
        _build_strategy_page(strategy, result, plot_params)
        for strategy, result in strategy_results.items()
    ]
    dashboard = dp.App(blocks=page_list)

    dashboard.save(path="report.html")
