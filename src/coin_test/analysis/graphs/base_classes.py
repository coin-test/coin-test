"""Graphing base classes."""

from abc import ABC, abstractmethod
from typing import Sequence, TypeAlias

import datapane as dp

from coin_test.backtest.backtest_results import BacktestResults
from .plot_parameters import PlotParameters


PLOT_RETURN_TYPE: TypeAlias = dp.Plot | dp.Select | dp.Group | dp.Toggle | dp.Media


class SinglePlotGenerator(ABC):
    """Generate a plot using a single BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results: BacktestResults, plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""


class DistributionalPlotGenerator(ABC):
    """Generate a plot using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results_list: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create distributional plot object."""
