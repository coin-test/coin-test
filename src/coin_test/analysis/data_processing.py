"""Generate plots for analysis."""

from abc import ABC, abstractmethod
from typing import Sequence

import datapane as dp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults


class DataframeGenerator(ABC):
    """Generate a pandas DataFrame using a single BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results: BacktestResults) -> pd.DataFrame:
        """Create dataframe."""


class DataframeGeneratorMultiple(ABC):
    """Generate a pandas DataFrame using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Create dataframe."""


class SinglePlotGenerator(ABC):
    """Generate a plot using a single BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results: BacktestResults) -> dp.Plot:
        """Create plot object."""


class DistributionalPlotGenerator(ABC):
    """Generate a plot using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> dp.Plot:
        """Create distributional plot object."""


class PricePlotSingle(SinglePlotGenerator):
    @staticmethod
    def create(backtest_results: BacktestResults) -> dp.Plot:
        """Create plot object."""

        # df = px.data.gapminder().query("continent=='Oceania'")
        # fig = px.line(backtest_results, x="Timestamp", y="lifeExp", color='country')

        # fig = go.Figure([go.Scatter(x=backtest_results.sim_data['Timestamp'], y=backtest_results.sim_data['Price'])])

        df = pd.read_csv(
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        )
        #
        # fig = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'], color_discrete_sequence="test" )])
        fig = px.line(
            x=df["Date"],
            y=df["AAPL.High"],
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig.update_layout(
            title="Plot Title",
            xaxis_title="X Axis Title",
            yaxis_title="Y Axis Title",
            legend_title="Legend Title",
            font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        )
        return dp.Plot(fig)


class PricePlotMultiple(DistributionalPlotGenerator):
    """Create Price plot from multiple Datasets"""

    @staticmethod
    def create(backtest_results: BacktestResults) -> dp.Plot:
        """Create plot object."""
        df = pd.read_csv(
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        )

        # Look here for more data: https://plotly.com/python/line-charts/
        fig = go.Figure(data=go.Scatter(x=df["Date"], y=df["AAPL.High"], name="main"))
        fig.add_scatter(x=df["Date"], y=df["AAPL.Low"], mode="lines", name="Low")
        fig.add_scatter(
            x=df["Date"], y=df["AAPL.Open"], mode="lines+markers", name="Open"
        )

        # OR
        fig = go.Figure()
        # Create and style traces
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.High"],
                name="main",
                line=dict(color="firebrick", width=4, dash="dashdot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.Low"],
                name="Low",
                line=dict(width=4, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.Open"],
                name="Open",
                line=dict(
                    width=4, dash="dash"
                ),  # dash options include 'dash', 'dot', and 'dashdot'
            )
        )

        # Should be able to update colors further here but I am unable
        # colors = ['gold', 'mediumturquoise','lightgreen']
        # fig.update_traces(hoverinfo='name', textfont_size=20,
        #           marker=dict(autocolorscale=False, line=dict(color=colors, width=2))) #color=colors

        fig.update_layout(
            title="Plot Title",
            xaxis_title="X Axis Title",
            yaxis_title="Y Axis Title",
            legend_title="Legend Title",
            font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        )
        return dp.Plot(fig)


# columns=["Timestamp", "Portfolios", "Trades", "Pending Trades", "Price"],
