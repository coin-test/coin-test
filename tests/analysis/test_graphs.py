"""Test graph generators."""
from collections.abc import Callable

import pytest

from coin_test.analysis.graphs import (
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    DistributionalPlotGenerator,
    PlotParameters,
    ReturnsHeatmapPlot,
    SignalPricePlot,
)
from coin_test.backtest import BacktestResults


@pytest.fixture
def plot_parameters() -> PlotParameters:
    """Default plot parameters."""
    return PlotParameters()


@pytest.mark.parametrize(
    "graph_generator",
    [
        ConfidencePricePlot,
        ConfidenceReturnsPlot,
        ReturnsHeatmapPlot,
        SignalPricePlot,
    ],
)
def test_single_strategy_graphs_single_run(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
) -> None:
    """Builds graph with no errors for single run."""
    names = ["1"]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize(
    "graph_generator",
    [
        ConfidencePricePlot,
        ConfidenceReturnsPlot,
        ReturnsHeatmapPlot,
        SignalPricePlot,
    ],
)
def test_single_strategy_graphs_many_runs(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
) -> None:
    """Builds graph with no errors for many runs."""
    names = ["1" for _ in range(10)]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize(
    "graph_generator",
    [
        ConfidencePricePlot,
        ConfidenceReturnsPlot,
        ReturnsHeatmapPlot,
        SignalPricePlot,
    ],
)
def test_single_strategy_graphs_error(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
) -> None:
    """Errors on multiple strategies."""
    names = ["1", "1", "2", "2", "2", "3"]
    results = [backtest_results_factory(name) for name in names]
    with pytest.raises(ValueError):
        graph_generator.create(results, plot_parameters)


@pytest.mark.parametrize("graph_generator", [ConfidenceDataPlot])
def test_multi_strategy_graphs(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
) -> None:
    """Builds graph with no errors."""
    names = ["1", "1", "2", "2", "2", "3"]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True
