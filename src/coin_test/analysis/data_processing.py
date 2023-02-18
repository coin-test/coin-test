"""Generate plots for analysis."""

from abc import ABC, abstractmethod
from typing import Sequence

import datapane as dp
import pandas as pd

from coin_test.backtest.backtest_results import BacktestResults


class DataframeGenerator(ABC):
    """Generate a pandas DataFrame using a single BacktestResults."""

    @abstractmethod
    @staticmethod
    def __call__(self, backtest_results: BacktestResults) -> pd.DataFrame:
        """Create dataframe."""

class DataframeGenerator(ABC):
    """Generate a pandas DataFrame using a multiple BacktestResults."""

    @abstractmethod
    @staticmethod
    def __call__(self, backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Create dataframe."""

class SinglePlotGenerator(ABC):
    """Generate a plot using a single BacktestResults."""

    @abstractmethod
    @staticmethod
    def __call__(self, backtest_results: BacktestResults) -> dp.Plot:
        """Create plot object."""

class DistributionalPlotGenerator(ABC):
    """Generate a plot using a multiple BacktestResults."""

    @abstractmethod
    @staticmethod
    def __call__(self, backtest_results_list: Sequence[BacktestResults]) -> dp.Plot:
        """Create distributional plot object."""