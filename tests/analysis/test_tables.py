"""Test table generators."""
from collections.abc import Callable
from unittest.mock import call

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.analysis.tables import MetricsGenerator, SummaryTearSheet, TearSheet
from coin_test.backtest import BacktestResults


@pytest.fixture
def mock_metrics() -> list[str]:
    """Mock metrics."""
    return [
        "Average Annual Return (%)",
        "Sharpe Ratio",
        "Sortino Rato",
        "Max Drawdown (%)",
        "Calmar Ratio",
    ]


def test_metrics_generator_single_metrics(backtest_results: BacktestResults) -> None:
    """Successfully get metrics for a single backtest."""
    metrics = MetricsGenerator._single_metrics(backtest_results)

    assert pytest.approx(metrics["Sharpe Ratio"], 1e-3) == 3.817


def test_metrics_generator_create(
    backtest_results: BacktestResults, mock_metrics: str, mocker: MockerFixture
) -> None:
    """Returns dataframe with proper shape."""
    metrics = {m: [] for m in mock_metrics}
    mocker.patch("coin_test.analysis.tables.MetricsGenerator._single_metrics")
    MetricsGenerator._single_metrics.return_value = metrics

    n_backtests = 5
    results = [backtest_results for _ in range(n_backtests)]
    df = MetricsGenerator.create(results)

    assert len(df) == n_backtests
    assert len(df.columns) == len(mock_metrics)
    calls = [call(backtest_results) for _ in range(n_backtests)]
    MetricsGenerator._single_metrics.assert_has_calls(calls)


def test_tear_sheet_create(
    backtest_results: BacktestResults, mock_metrics: str, mocker: MockerFixture
) -> None:
    """Returns tear sheet with proper shape."""
    mocker.patch("coin_test.analysis.tables.MetricsGenerator.create")
    metrics = {m: [0] * 100 for m in mock_metrics}
    df = pd.DataFrame.from_dict(metrics)
    MetricsGenerator.create.return_value = df

    results = [backtest_results]
    tear_sheet = TearSheet.create(results)

    assert len(tear_sheet) == len(mock_metrics)  # One row per metric
    for metric in mock_metrics:
        assert metric in tear_sheet.index
    assert len(tear_sheet.columns) == 2  # "Mean" and "Std"
    for metric in ("Mean", "Standard Deviation"):
        assert metric in tear_sheet.columns
        assert type(tear_sheet[metric].iloc[0]) is str
    MetricsGenerator.create.assert_called_with(results)


def test_summary_tear_sheet_create(
    backtest_results_factory: Callable[[str], BacktestResults],
    mock_metrics: str,
    mocker: MockerFixture,
) -> None:
    """Returns summary tear sheet with proper shape."""
    mocker.patch("coin_test.analysis.tables.TearSheet.create")
    metrics = {
        "Mean": ["0"] * len(mock_metrics),
        "Standard Deviation": ["0"] * len(mock_metrics),
    }
    df = pd.DataFrame.from_dict(metrics)
    df = df.set_index(pd.Series(mock_metrics))
    TearSheet.create.return_value = df

    names = ["S1", "S1", "S2"]
    results = [backtest_results_factory(name) for name in names]
    summary_tear_sheet = SummaryTearSheet.create(results)

    assert len(summary_tear_sheet) == 2  # 2 unique names
    assert len(summary_tear_sheet.columns) == len(mock_metrics)
    for name in names:
        assert name in summary_tear_sheet.index
    for metric in mock_metrics:
        assert metric in summary_tear_sheet.columns
    calls = [call(results[0:2]), call(results[2:])]
    TearSheet.create.assert_has_calls(calls)
