"""Generate plots for analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import datapane as dp
import numpy as np
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
        df = pd.DataFrame(
            {
                "A": np.random.normal(-1, 1, 5000),
                "B": np.random.normal(1, 2, 5000),
            }
        )
        return df


class DataframeGeneratorMultiple(ABC):
    """Generate a pandas DataFrame using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Create dataframe."""


class PlotParameters:
    """Plot parameters to pass to each plot."""

    def __init__(
        self,
        line_styles: list[str] = ["solid", "dash", "dot", "dashdot"],
        line_colors: list[str] = ["firebrick", "blue", "rebeccapurple"],
        label_font_color: str = "black",
        label_font_family: str = "Courier New, monospace",
        title_font_size: int = 18,
        axes_font_size: int = 10,
        line_width: int = 2,
    ) -> None:
        self.line_styles = line_styles
        self.line_colors = line_colors
        self.line_width = line_width
        self.title_font = dict(
            family=label_font_family, size=title_font_size, color=label_font_color
        )
        self.axes_font = dict(
            family=label_font_family, size=axes_font_size, color=label_font_color
        )
        self.legend_font = self.axes_font

    @staticmethod
    def update_plotly_fig(
        plot_params: "PlotParameters",
        fig: go.Figure,
        title: str,
        x_lbl: str,
        y_lbl: str,
        legend_title: str = "",
    ) -> None:
        """Update Plotly figure"""
        fig.update_layout(
            title={"text": title, "font": plot_params.title_font},
            xaxis_title={"text": x_lbl, "font": plot_params.axes_font},
            yaxis_title={"text": y_lbl, "font": plot_params.axes_font},
            legend_title={"text": legend_title, "font": plot_params.legend_font},
        )


class SinglePlotGenerator(ABC):
    """Generate a plot using a single BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results: BacktestResults, plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""


class DistributionalPlotGenerator(ABC):
    """Generate a plot using a multiple BacktestResults."""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results_list: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create distributional plot object."""


class PricePlotSingle(SinglePlotGenerator):
    @staticmethod
    def create(
        backtest_results: BacktestResults, plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""

        df = pd.read_csv(
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        )
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
            font=plot_params.title_font,
        )
        return dp.Plot(fig)


class PricePlotMultiple(DistributionalPlotGenerator):
    """Create Price plot from multiple Datasets"""

    @staticmethod
    def create(
        backtest_results: BacktestResults, plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        df = pd.read_csv(
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        )

        fig = go.Figure()
        # Create and style traces
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.High"],
                name="main",
                line=dict(
                    color=plot_params.line_colors[0],
                    width=plot_params.line_width,
                    dash=plot_params.line_styles[0],
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.Low"],
                name="Low",
                line=dict(
                    color=plot_params.line_colors[1],
                    width=plot_params.line_width,
                    dash=plot_params.line_styles[1],
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["AAPL.Open"],
                name="Open",
                line=dict(
                    color=plot_params.line_colors[2],
                    width=plot_params.line_width,
                    dash=plot_params.line_styles[2],
                ),  # dash options include 'dash', 'dot', and 'dashdot'
            )
        )

        # Should be able to update colors further here but I am unable
        # colors = ['gold', 'mediumturquoise','lightgreen']
        # fig.update_traces(hoverinfo='name', textfont_size=20,
        #           marker=dict(autocolorscale=False, line=dict(color=colors, width=2))) #color=colors

        PlotParameters.update_plotly_fig(
            plot_params, fig, "My title", "x_lbl", "y_lbl", "Legend"
        )
        return dp.Plot(fig)
