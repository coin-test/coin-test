"""Generate distributional plots."""

from typing import Sequence

import datapane as dp
import pandas as pd
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from .utils import make_select
from ..utils import get_strategy_results


class CandlestickPlot(DistributionalPlotGenerator):
    """Create asset price plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        strategy_results = get_strategy_results(backtest_results)
        backtest_results = list(strategy_results.values())[0]

        figures = []
        for results in backtest_results[:10]:
            index = results.sim_data.index
            df = list(results.data_dict.values())[0]
            df = df.loc[index[0] : index[-1]]

            def _to_timestamp(period: pd.Period) -> pd.Timestamp:
                return period.start_time

            index = pd.Series(df.index).apply(_to_timestamp)
            figures.append(
                go.Figure(
                    go.Candlestick(
                        x=index,
                        open=df["Open"],
                        close=df["Close"],
                        low=df["Low"],
                        high=df["High"],
                        increasing=dict(line=dict(color=plot_params.line_colors[2])),
                        decreasing=dict(line=dict(color=plot_params.line_colors[1])),
                    )
                )
            )
            figures[-1].update_layout(xaxis_rangeslider_visible=False)

        for i, fig in enumerate(figures[:10]):
            PlotParameters.update_plotly_fig(
                plot_params,
                fig,
                f"Dataset {i}",
                "Time",
                "Asset Value",
            )

        return make_select(
            [
                dp.Media(
                    plot_params.compress_fig(fig, name="candlestick"),
                    label=f"Dataset {i}",
                )
                for i, fig in enumerate(figures)
            ]
        )
