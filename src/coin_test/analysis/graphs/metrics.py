"""Generate heatmap for analysis."""

from collections import defaultdict
from typing import Sequence

import datapane as dp
import pandas as pd
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from ..tables import MetricsGenerator
from ..utils import get_strategy_results


class MetricsPlot(DistributionalPlotGenerator):
    """Create strategy vs dataset returns heatmap."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        strategy_results = get_strategy_results(backtest_results)
        all_metrics = defaultdict(lambda: defaultdict(list))
        for strategy, results in strategy_results.items():
            metrics = MetricsGenerator.create(results)
            for metric in metrics:
                all_metrics[metric][strategy].append(metrics[metric])

        figs = {}
        for metric, strategies in all_metrics.items():
            fig = go.Figure()
            for strategy, data in strategies.items():
                data = pd.concat(data)
                fig.add_trace(go.Violin(y=data, x0=strategy, name=strategy))
            PlotParameters.update_plotly_fig(
                plot_params,
                fig,
                metric,
                "Strategies",
                "Value",
                "Legend",
            )
            figs[metric] = fig

        return dp.Group(*[dp.Plot(fig, label=metric) for metric, fig in figs.items()])
