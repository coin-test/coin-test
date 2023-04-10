"""Graphing utility functions."""

from typing import Sequence

import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from ..utils import get_strategies


def clamp(x: float, min_: float, max_: float) -> float:
    """Clamp value."""
    return min(max(x, min_), max_)


def get_lims(figs: Sequence[go.Figure], x: bool = True) -> list[tuple[float, float]]:
    """Get shared limits for list of figures.

    Args:
        figs: Sequence of plotly figure
        x: Whether to get x-lims in addition to y lims. Set to False if using datetime
            values for the x axis.

    Returns:
        Limits: List of tuples, wher each tuple is (limit min, limit max). First element
            is always y limits. Second element is the x limits, and is only returned if
            x is True.
    """
    x_vals, y_vals = [], []
    for fig in figs:
        full_fig = fig.full_figure_for_development(warn=False)
        x_vals.extend(full_fig.layout.xaxis.range)
        y_vals.extend(full_fig.layout.yaxis.range)
    ret = []
    if x:
        ret.append((min(x_vals), max(x_vals)))
    ret.append((min(y_vals), max(y_vals)))
    return ret


def is_single_strategy(results: Sequence[BacktestResults]) -> None:
    """Raise ValueError is multiple strategies exist in list of BacktestResults."""
    if len(get_strategies(results)) != 1:
        raise ValueError("Multiple strategies passed to single strategy plot!")
