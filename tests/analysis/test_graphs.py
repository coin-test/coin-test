"""Test graph generators."""
from collections.abc import Callable
import os
from unittest.mock import Mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.analysis.graphs import (
    CandlestickPlot,
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    DistributionalPlotGenerator,
    MetricsPlot,
    PlotParameters,
    ReturnsHeatmapPlot,
    SignalHeatmapPlot,
    SignalTotalPlot,
)
from coin_test.backtest import BacktestResults


@pytest.fixture
def plot_parameters(mocker: MockerFixture) -> PlotParameters:
    """Default plot parameters."""
    mocker.patch("os.makedirs")
    params = PlotParameters("")
    params.compress_fig = Mock()
    params.compress_fig.return_value = ""
    return params


def _mock_write(mocker: MockerFixture) -> None:
    mocker.patch("plotly.graph_objects.Figure.write_image")


def test_plot_params_comrpess_fig(mocker: MockerFixture) -> None:
    """Build proper save path."""
    mocker.patch("os.makedirs")
    output_dir = "test_dir"
    name = "test"
    params = PlotParameters(output_dir)
    mocker.patch("os.path.exists")
    os.path.exists.side_effect = [True, False]
    path = params.compress_fig(Mock(), name)
    expected = os.path.join(output_dir, f"{name}(1).png")
    assert path == expected


single_strategy_graphs = [
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    ReturnsHeatmapPlot,
    SignalHeatmapPlot,
    SignalTotalPlot,
]


@pytest.mark.parametrize("graph_generator", single_strategy_graphs)
def test_single_strategy_graphs_single_run(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors for single run."""
    _mock_write(mocker)
    names = ["1"]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize("graph_generator", single_strategy_graphs)
def test_single_strategy_graphs_many_runs(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors for many runs."""
    _mock_write(mocker)
    names = ["1" for _ in range(10)]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize("graph_generator", single_strategy_graphs)
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


@pytest.mark.parametrize("graph_generator", [SignalHeatmapPlot])
def test_single_strategy_graphs_no_trades(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors for many runs."""
    _mock_write(mocker)
    names = ["1" for _ in range(10)]
    results = [backtest_results_factory(name) for name in names]
    for result in results:
        new_trades_col = pd.Series(
            [[] for _ in result.sim_data.index], index=result.sim_data.index
        )
        result.sim_data["Trades"] = new_trades_col
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize("graph_generator", [SignalHeatmapPlot])
def test_single_strategy_graphs_late_trades(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors for late trades."""
    _mock_write(mocker)
    names = ["1" for _ in range(10)]
    results = [backtest_results_factory(name) for name in names]
    for result in results:
        half = len(result.sim_data) // 2
        data = [[] for _ in range(half)]
        data += result.sim_data["Trades"][half:].to_list()
        new_trades_col = pd.Series(data, index=result.sim_data.index)
        result.sim_data["Trades"] = new_trades_col
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize("graph_generator", [SignalHeatmapPlot])
def test_single_strategy_graphs_early_trades(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors for early trades."""
    _mock_write(mocker)
    names = ["1" for _ in range(10)]
    results = [backtest_results_factory(name) for name in names]
    for result in results:
        half = len(result.sim_data) // 2
        data = [[] for _ in range(half)]
        data = result.sim_data["Trades"][:half].to_list() + data
        new_trades_col = pd.Series(data, index=result.sim_data.index)
        result.sim_data["Trades"] = new_trades_col
    graph_generator.create(results, plot_parameters)
    assert True


@pytest.mark.parametrize(
    "graph_generator", [CandlestickPlot, ConfidenceDataPlot, MetricsPlot]
)
def test_multi_strategy_graphs(
    graph_generator: DistributionalPlotGenerator,
    backtest_results_factory: Callable[[str], BacktestResults],
    plot_parameters: PlotParameters,
    mocker: MockerFixture,
) -> None:
    """Builds graph with no errors."""
    _mock_write(mocker)
    names = ["1", "1", "2", "2", "2", "3"]
    results = [backtest_results_factory(name) for name in names]
    graph_generator.create(results, plot_parameters)
    assert True
