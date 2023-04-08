"""Test app generation."""
from collections.abc import Callable
from unittest.mock import MagicMock

import datapane as dp
from pytest_mock import MockerFixture

from coin_test.analysis import build_datapane
from coin_test.analysis.graphs import (
    ConfidenceDataPlot,
    ConfidencePricePlot,
    ConfidenceReturnsPlot,
    ReturnsHeatmapPlot,
    SignalPricePlot,
)
from coin_test.analysis.tables import SummaryTearSheet, TearSheet
from coin_test.backtest import BacktestResults


def test_build_datapane(
    backtest_results_factory: Callable[[str], BacktestResults],
    mocker: MockerFixture,
) -> None:
    """Builds datapane app succesfully."""
    mocker.patch("coin_test.analysis.graphs.ConfidenceDataPlot.create")
    ConfidenceDataPlot.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.graphs.ConfidencePricePlot.create")
    ConfidencePricePlot.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.graphs.ConfidenceReturnsPlot.create")
    ConfidenceReturnsPlot.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.graphs.ReturnsHeatmapPlot.create")
    ReturnsHeatmapPlot.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.graphs.SignalPricePlot.create")
    SignalPricePlot.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.tables.SummaryTearSheet.create")
    SummaryTearSheet.create.return_value = dp.Group()
    mocker.patch("coin_test.analysis.tables.TearSheet.create")
    TearSheet.create.return_value = dp.Group()

    mock_app = MagicMock()
    mocker.patch("datapane.App")
    dp.App.return_value = mock_app

    names = ["1", "2", "3", "4", "1", "1", "4", "5", "2"]
    results = [backtest_results_factory(name) for name in names]
    build_datapane(results)

    dp.App.assert_called_once()
    mock_app.save.assert_called_once()
