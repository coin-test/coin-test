"""Generate plots for analysis."""

from abc import ABC, abstractmethod
import math
from typing import Sequence

import datapane as dp
import pandas as pd
from plotly.colors import n_colors
import plotly.graph_objects as go


from coin_test.backtest.backtest_results import BacktestResults


def _flatten_strategies(results: BacktestResults) -> str:
    return "-".join(results.strategy_names)


def _get_strategies(results: Sequence[BacktestResults]) -> list[str]:
    return sorted(list(set(_flatten_strategies(r) for r in results)))


def _get_strategy_results(
    results: Sequence[BacktestResults],
) -> dict[str, list[BacktestResults]]:
    strategies = _get_strategies(results)
    strategy_results = {}
    for strategy in strategies:
        strategy_results[strategy] = []
    for result in results:
        strategy = _flatten_strategies(result)
        strategy_results[strategy].append(result)
    return strategy_results


class PlotParameters:
    """Plot parameters to pass to each plot."""

    def __init__(
        self,
        line_styles: tuple[str, ...] = ("solid", "dash", "dot", "dashdot"),
        line_colors: tuple[str, ...] = (
            "rgb(5, 200, 200)",
            "rgb(200, 10, 10)",
            "rgb(150, 10, 150)",
        ),
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
        title: str | None,
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


def _build_percentiles(
    name: str,
    df: pd.DataFrame,
    plot_params: PlotParameters,
) -> go.Figure:
    mean = df.mean(axis=1)
    mid = df.quantile(0.5, axis=1)
    upper = df.quantile(0.75, axis=1)
    lower = df.quantile(0.25, axis=1)
    traces = [
        go.Scatter(
            name="Mean " + name,
            x=df.index,
            y=mean,
            mode="lines",
            line=dict(color=plot_params.line_colors[0]),
        ),
        go.Scatter(
            name="Median " + name,
            x=df.index,
            y=mid,
            mode="lines",
            line=dict(color=plot_params.line_colors[1]),
        ),
        go.Scatter(
            name="75th Percentile",
            x=df.index,
            y=upper,
            mode="lines",
            marker=dict(color=plot_params.line_colors[2]),
            line=dict(width=0),
            showlegend=False,
        ),
        go.Scatter(
            name="25th Percentile",
            x=df.index,
            y=lower,
            marker=dict(color=plot_params.line_colors[2]),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            showlegend=False,
        ),
    ]
    fig = go.Figure(traces)
    PlotParameters.update_plotly_fig(
        plot_params,
        fig,
        "Percentiles",
        "Time",
        name,
        "Legend",
    )
    return fig


def _build_ridgeline(
    name: str,
    df: pd.DataFrame,
    plot_params: PlotParameters,
) -> go.Figure:
    colors = n_colors(
        plot_params.line_colors[0],
        plot_params.line_colors[1],
        len(df),
        colortype="rgb",
    )
    df["colors"] = colors
    index = df.index
    df = df.reset_index(drop=True)

    def _make_ridgeline(series: pd.Series) -> go.Violin:
        color = series.iloc[-1]
        series = series.iloc[:-1]
        return go.Violin(y=series, name=str(series.name), line_color=color)

    traces = df.apply(_make_ridgeline, axis=1).to_list()
    fig = go.Figure(traces)
    fig.update_traces(
        width=5,
        orientation="v",
        side="positive",
        points=False,
        showlegend=False,
    )
    fig.update_layout(
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        violingroupgap=1,
    )

    tickvals = [0, len(df)]
    ticktext = [str(index[0]), str(index[-1])]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            colorscale=[[0, colors[0]], [1, colors[-1]]],
            showscale=True,
            cmin=-1,
            cmax=1,
            colorbar=dict(
                thickness=5,
                tickvals=[-1, 1],
                ticktext=[str(index[0]), str(index[-1])],
                outlinewidth=0,
            ),
        ),
        hoverinfo="none",
    )
    fig.add_trace(colorbar_trace)

    PlotParameters.update_plotly_fig(
        plot_params,
        fig,
        "Ridge Plot",
        "Time",
        name,
    )
    return fig


def _build_lines(
    name: str,
    df: pd.DataFrame,
    plot_params: PlotParameters,
) -> go.Figure:
    def _make_lines(series: pd.Series) -> go.Scatter:
        return go.Scatter(
            y=series,
            x=df.index,
            opacity=0.1,
            mode="lines",
            marker=dict(color=plot_params.line_colors[0]),
            showlegend=False,
        )

    traces = df.apply(_make_lines, axis=0).to_list()
    fig = go.Figure(traces)
    PlotParameters.update_plotly_fig(
        plot_params,
        fig,
        "Line Plot",
        "Time",
        name,
    )
    return fig


def _build_distributions_selection(
    name: str,
    df: pd.DataFrame,
    plot_params: PlotParameters,
) -> dp.Select:
    min_price, max_price = df.to_numpy().min() * 0.9, df.to_numpy().max() * 1.1
    band_fig = _build_percentiles(name, df, plot_params)
    ridge_fig = _build_ridgeline(name, df, plot_params)
    lines_fig = _build_lines(name, df, plot_params)
    band_fig.update_yaxes(range=[min_price, max_price])
    ridge_fig.update_yaxes(range=[min_price, max_price])
    lines_fig.update_yaxes(range=[min_price, max_price])
    return dp.Select(
        dp.Plot(band_fig, label="Percentiles"),
        dp.Plot(ridge_fig, label="Ridge Plot"),
        dp.Plot(lines_fig, label="Line Plot"),
    )


class ConfidencePricePlot(DistributionalPlotGenerator):
    """Create Price plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        return _build_distributions_selection("Portfolio Value", price_df, plot_params)


class ConfidenceReturnsPlot(DistributionalPlotGenerator):
    """Create returns plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        returns_df = price_df.pct_change()
        return _build_distributions_selection(
            "Portfolio Return Over Time", returns_df, plot_params
        )


class ConfidenceDataPlot(DistributionalPlotGenerator):
    """Create data plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        strategy_results = _get_strategy_results(backtest_results)
        backtest_results = list(strategy_results.values())[0]
        dfs = []
        base_index = None
        for results in backtest_results:
            index = results.sim_data.index
            df = list(results.data_dict.values())[0]
            df = df.loc[index[0] : index[-1]]["Open"]
            if base_index is None:
                base_index = df.index.to_timestamp()  # type: ignore
            dfs.append(df.reset_index(drop=True))
        data_df = pd.concat(dfs, axis=1)
        data_df.set_index(base_index, inplace=True)
        return _build_distributions_selection(
            "Asset Value Over Time", data_df, plot_params
        )


class ReturnsHeatmapPlot(DistributionalPlotGenerator):
    """Create strategy vs dataset returns heatmap."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> dp.Plot:
        """Create plot object."""
        x = []
        y = []
        for result in backtest_results:
            portfolio_value = result.sim_data["Price"]
            portfolio_change = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
            y.append(portfolio_change)
            trading_start = portfolio_value.index[0]
            trading_end = portfolio_value.index[-1]
            asset_value = list(result.data_dict.values())[0]["Open"]
            asset_change = (
                asset_value[trading_end:trading_end].iloc[0]
                / asset_value[trading_start:trading_start].iloc[0]
            )
            x.append(asset_change)

        lb = math.floor(min(min(x), min(y)) * 10) / 10
        ub = math.ceil(max(max(x), max(y)) * 10) / 10
        step = 0.05
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
