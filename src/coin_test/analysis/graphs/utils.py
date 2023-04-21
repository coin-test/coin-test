"""Graphing utility functions."""

from typing import Sequence

import datapane as dp
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from ..utils import get_strategies


def clamp(x: float, min_: float, max_: float) -> float:
    """Clamp value."""
    return min(max(x, min_), max_)


def get_lims(figs: Sequence[go.Figure]) -> tuple[float, float]:
    """Get shared limits for list of figures.

    Args:
        figs: Sequence of plotly figures.

    Returns:
        Limits: Minimum y value and maximium y value.
    """
    y_vals = []
    for fig in figs:
        full_fig = fig.full_figure_for_development(warn=False)
        y_vals.extend(full_fig.layout.yaxis.range)
    return min(y_vals), max(y_vals)


def is_single_strategy(results: Sequence[BacktestResults]) -> None:
    """Raise ValueError is multiple strategies exist in list of BacktestResults."""
    if len(get_strategies(results)) != 1:
        raise ValueError("Multiple strategies passed to single strategy plot!")


def make_select(options: list[dp.Plot | dp.Media]) -> dp.Select:
    """Build a DataPane select without erroring when only using a single element."""
    if len(options) == 1:
        return options[0]
    return dp.Select(*options)
