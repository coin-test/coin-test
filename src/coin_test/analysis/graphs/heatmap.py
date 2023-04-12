"""Generate heatmap for analysis."""

import math
from typing import Sequence

import datapane as dp
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from .utils import is_single_strategy


class ReturnsHeatmapPlot(DistributionalPlotGenerator):
    """Create strategy vs dataset returns heatmap."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)
        x = []
        y = []
        for result in backtest_results:
            portfolio_value = result.sim_data["Price"]
            portfolio_change = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
            y.append(portfolio_change)
            trading_start = portfolio_value.index[1]
            trading_end = portfolio_value.index[-1]
            asset_value = list(result.data_dict.values())[0]["Open"]
            asset_change = (
                asset_value[trading_end:trading_end].iloc[0]
                / asset_value[trading_start:trading_start].iloc[0]
            )
            x.append(asset_change)

        lb = math.floor(min(min(x), min(y)) * 10) / 10
        ub = math.ceil(max(max(x), max(y)) * 10) / 10
        step = 0.25
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[lb, ub],
                y=[lb, ub],
                mode="lines",
                showlegend=False,
                marker=dict(
                    opacity=0.5,
                    color="white",
                ),
            )
        )
        fig.add_trace(
            go.Histogram2d(
                x=x,
                y=y,
                autobinx=False,
                xbins=dict(start=lb, end=ub, size=step),
                ybins=dict(start=lb, end=ub, size=step),
            )
        )
        fig.update_layout(
            yaxis_range=[lb, ub],
            xaxis_range=[lb, ub],
        )
        PlotParameters.update_plotly_fig(
            plot_params,
            fig,
            None,
            "Dataset Return",
            "Portfolio Return",
            "Legend",
        )
        return dp.Plot(fig)
