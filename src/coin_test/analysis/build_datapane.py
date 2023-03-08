"""Functions to build Datapane Locally."""

import math
from typing import Sequence, Type

import datapane as dp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_processing import (
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    DataframeGeneratorMultiple,
    DistributionalPlotGenerator,
    PlotParameters,
    ReturnsHeatmapPlot,
)
from .tear_sheet import MetricsGenerator, TearSheet
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
    blocks = []
    raw_metrics = {}
    summary_metrics = {}
    for strategy, results in strategy_results.items():
        raw = MetricsGenerator.create(results)
        raw_metrics[strategy] = raw
        tear_sheet = TearSheet.create(results)
        mean = tear_sheet["Mean"].round(2).astype(str)
        std = tear_sheet["Standard Deviation"].round(2).astype(str)
        summary_metrics[strategy] = mean + " ± " + std
    tear_sheet = pd.DataFrame.from_dict(summary_metrics).transpose()
    blocks.extend(["### Tear Sheet", tear_sheet])

    raw_df = pd.concat(raw_metrics, axis=1).transpose()
    raw_df = raw_df.swaplevel()
    metrics = raw_df.index.get_level_values(0).unique()
    fig = make_subplots(
        rows=math.ceil(len(raw_df.groupby(level=0)) / 2),
        cols=2,
        subplot_titles=metrics,
    )
    for i, metric in enumerate(metrics):
        df = raw_df[raw_df.index.get_level_values(0) == metric]
        df.index = df.index.get_level_values(1)
        for strategy, data in df.iterrows():
            fig.add_trace(
                go.Violin(
                    y=data.dropna(),
                    name=strategy,
                    legendgroup="test",
                    scalemode="count",
                ),
                row=(i // 2) + 1,
                col=(i % 2) + 1,
            )
    fig.update_layout(height=1024, width=512, showlegend=True)
    blocks.append(dp.Plot(fig))

    page = dp.Page(title="Home", blocks=blocks)
    return page


def _build_data_page(
    results: list[BacktestResults], plot_params: PlotParameters
) -> dp.Page:
    confidence_graph = ConfidenceDataPlot.create(results, plot_params)
    blocks = [
        "# Data",
        confidence_graph,
    ]
    page = dp.Page(title="Data", blocks=blocks)
    return page


def build_datapane(results: list[BacktestResults]) -> None:
    """Build Datapane from large set of results.

    Args:
        results: List of BacktestResults.
    """
    plot_params = PlotParameters()

    strategy_results = _get_strategy_results(results)
    page_list = []
    page_list.append(_build_home_page(strategy_results, plot_params))
    page_list.append(_build_data_page(results, plot_params))
    page_list.extend(
        [
            _build_strategy_page(strategy, result, plot_params)
            for strategy, result in strategy_results.items()
        ]
    )
    dashboard = dp.App(blocks=page_list)

    dashboard.save(path="report.html")
