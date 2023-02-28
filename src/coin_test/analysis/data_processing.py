"""Generate plots for analysis."""

from abc import ABC, abstractmethod
import math
from typing import Sequence

import datapane as dp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coin_test.backtest.backtest_results import BacktestResults


class DataframeGenerator(ABC):
    """Generate a pandas DataFrame using a single BacktestResults."""

    name = ""

    @staticmethod
    @abstractmethod
    def create(backtest_results: BacktestResults) -> pd.DataFrame:
        """Create dataframe."""


class DataframeGeneratorMultiple(ABC):
    """Generate a pandas DataFrame using a multiple BacktestResults."""

    name = ""

    @staticmethod
    @abstractmethod
    def create(backtest_results_list: Sequence[BacktestResults]) -> pd.DataFrame:
        """Create dataframe."""


class PlotParameters:
    """Plot parameters to pass to each plot."""

    def __init__(
        self,
        line_styles: tuple[str, ...] = ("solid", "dash", "dot", "dashdot"),
        line_colors: tuple[str, ...] = ("firebrick", "blue", "rebeccapurple"),
        label_font_color: str = "black",
        label_font_family: str = "Courier New, monospace",
        title_font_size: int = 18,
        axes_font_size: int = 10,
        line_width: int = 2,
    ) -> None:
        """Initialize plot parameters."""
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
        """Update Plotly figure."""
        fig.update_layout(
            title={"text": title, "font": plot_params.title_font},
            xaxis_title={"text": x_lbl, "font": plot_params.axes_font},
            yaxis_title={"text": y_lbl, "font": plot_params.axes_font},
            legend_title={"text": legend_title, "font": plot_params.legend_font},
        )


class SinglePlotGenerator(ABC):
    """Generate a plot using a single BacktestResults."""

    name = ""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results: BacktestResults, plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""


class DistributionalPlotGenerator(ABC):
    """Generate a plot using a multiple BacktestResults."""

    name = ""

    @staticmethod
    @abstractmethod
    def create(
        backtest_results_list: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create distributional plot object."""


class PricePlotSingle(SinglePlotGenerator):
    """Create price plot from single dataset."""

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
    """Create Price plot from multiple Datasets."""

    name = "Portfolio Value"

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        fig = go.Figure()
        # Create and style traces
        for i, result in enumerate(backtest_results):
            df = result.sim_data
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Price"],
                    name=f"Dataset {i}",
                    line=dict(
                        color=plot_params.line_colors[i % 3],
                        width=plot_params.line_width,
                        dash=plot_params.line_styles[i % 3],
                    ),
                )
            )
        PlotParameters.update_plotly_fig(
            plot_params, fig, "", "Time", "Portfolio Value", "Legend"
        )
        return dp.Plot(fig)


def _build_confidence_traces(
    name: str,
    df: pd.DataFrame,
    plot_params: PlotParameters,
) -> list[go.Scatter]:
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    upper = mean + std
    lower = mean - std
    return [
        go.Scatter(
            name=name,
            x=df.index,
            y=mean,
            mode="lines",
            line=dict(color=plot_params.line_colors[0]),
        ),
        go.Scatter(
            name="70% Confidence Upper Bound",
            x=df.index,
            y=upper,
            mode="lines",
            marker=dict(color=plot_params.line_colors[1]),
            line=dict(width=0),
            showlegend=False,
        ),
        go.Scatter(
            name="70% Confidence Lower Bound",
            x=df.index,
            y=lower,
            marker=dict(color=plot_params.line_colors[1]),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            showlegend=False,
        ),
    ]


class ConfidencePricePlot(DistributionalPlotGenerator):
    """Create Price plot with confidence band."""

    name = "Portfolio Value"

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        traces = _build_confidence_traces("Portfolio Value", price_df, plot_params)

        fig = go.Figure(traces)
        PlotParameters.update_plotly_fig(
            plot_params, fig, "", "Time", "Portfolio Value", "Legend"
        )
        return dp.Plot(fig)


class ConfidenceReturnsPlot(DistributionalPlotGenerator):
    """Create returns plot with confidence band."""

    name = "Returns"

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        returns_df = price_df.pct_change()
        traces = _build_confidence_traces("Returns", returns_df, plot_params)
        fig = go.Figure(traces)
        PlotParameters.update_plotly_fig(
            plot_params, fig, "", "Time", "Returns", "Legend"
        )
        return dp.Plot(fig)


class ReturnsHeatmapPlot(DistributionalPlotGenerator):
    """Create strategy vs dataset returns heatmap."""

    name = "Strategy vs Dataset Returns"

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        x = []
        y = []
        for result in backtest_results:
            asset_value = list(result.data_dict.values())[0]["Open"]
            asset_change = asset_value.iloc[-1] / asset_value[0]
            x.append(asset_change)
            portfolio_value = result.sim_data["Price"]
            portfolio_change = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
            y.append(portfolio_change)

        lb = math.floor(min(min(x), min(y)) * 10) / 10
        ub = math.ceil(max(max(x), max(y)) * 10) / 10
        step = 0.05
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    opacity=0.7,
                    color="white",
                    line=dict(width=1),
                ),
            )
        )
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
            plot_params, fig, "", "Dataset Return", "Portfolio Return", "Legend"
        )
        return dp.Plot(fig)


class DataPlot(DistributionalPlotGenerator):
    """Create Price plot from multiple Datasets."""

    name = "Candles vs Portfolio Value"

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        # fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Create and style traces
        for i, result in enumerate(backtest_results):
            df = result.sim_data.tail(-1)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Price"],
                    name="Portfolio Value",
                    line=dict(
                        color=plot_params.line_colors[1],
                        width=plot_params.line_width,
                        dash=plot_params.line_styles[1],
                    ),
                    visible=(i == 0),
                ),
                secondary_y=False,
            )
            candles = list(result.data_dict.values())[0]
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=candles["Open"],
                    high=candles["High"],
                    low=candles["Low"],
                    close=candles["Close"],
                    name="Candles",
                    visible=(i == 0),
                ),
                secondary_y=True,
            )
        PlotParameters.update_plotly_fig(
            plot_params, fig, "", "Time", "Portfolio Value", "Legend"
        )
        visibles = [
            [i // 2 == j for i in range(len(backtest_results) * 2)]
            for j in range(len(backtest_results))
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=[
                        dict(
                            label=f"Dataset {i}",
                            method="update",
                            args=[
                                {
                                    "visible": visibles[i],
                                    "title": f"Dataset {i}",
                                }
                            ],
                        )
                        for i in range(len(backtest_results))
                    ],
                ),
            ]
        )
        return dp.Plot(fig)
