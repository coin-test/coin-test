"""Generate distributional plots."""

from typing import Sequence

import datapane as dp
import pandas as pd
from plotly.colors import n_colors
import plotly.graph_objects as go

from coin_test.backtest.backtest_results import BacktestResults
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from .utils import clamp, get_lims, is_single_strategy
from ..utils import get_strategy_results


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
            line=dict(color=plot_params.line_colors[1]),
        ),
        go.Scatter(
            name="Median " + name,
            x=df.index,
            y=mid,
            mode="lines",
            line=dict(color=plot_params.line_colors[2]),
        ),
        go.Scatter(
            name="75th Percentile",
            x=df.index,
            y=upper,
            mode="lines",
            marker=dict(color=plot_params.line_colors[0]),
            line=dict(width=0),
            showlegend=False,
        ),
        go.Scatter(
            name="25th Percentile",
            x=df.index,
            y=lower,
            marker=dict(color=plot_params.line_colors[0]),
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
    max_ridges: int = 80,
) -> go.Figure:
    df = df.copy()

    if len(df) > max_ridges:
        scale_factor = len(df) // max_ridges
        df = df[::scale_factor]

    colors = n_colors(
        plot_params.line_colors[1],
        plot_params.line_colors[2],
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

    traces = df.apply(_make_ridgeline, axis=1).tolist()
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
        hoverinfo="skip",
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
    opacity = clamp(1 / len(df.columns) * 2, 0.05, 0.5)

    def _make_lines(series: pd.Series) -> go.Scatter:
        return go.Scatter(
            y=series,
            x=df.index,
            opacity=opacity,
            mode="lines",
            marker=dict(color=plot_params.line_colors[0]),
            showlegend=False,
            hoverinfo="skip",
        )

    traces = df.apply(_make_lines, axis=0).tolist()  # type: ignore
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
    band_fig = _build_percentiles(name, df, plot_params)
    ridge_fig = _build_ridgeline(name, df, plot_params)
    lines_fig = _build_lines(name, df, plot_params)
    y_lims = get_lims((band_fig, ridge_fig, lines_fig), x=False)[0]
    band_fig.update_yaxes(range=y_lims)
    ridge_fig.update_yaxes(range=y_lims)
    lines_fig.update_yaxes(range=y_lims)
    return dp.Select(
        dp.Plot(band_fig, label="Percentiles"),
        dp.Media(
            plot_params.compress_fig(ridge_fig, name + "_ridge_plot"),
            label="Ridge Plot",
        ),
        dp.Media(
            plot_params.compress_fig(lines_fig, name + "_line_plot"),
            label="Line Plot",
        ),
    )


class ConfidencePricePlot(DistributionalPlotGenerator):
    """Create Price plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        return _build_distributions_selection("Portfolio Value", price_df, plot_params)


class ConfidenceReturnsPlot(DistributionalPlotGenerator):
    """Create returns plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)
        price_series = [results.sim_data["Price"] for results in backtest_results]
        price_df = pd.concat(price_series, axis=1)
        returns_df = price_df.pct_change()
        return _build_distributions_selection(
            "Portfolio Return Over Time", returns_df, plot_params
        )


class ConfidenceDataPlot(DistributionalPlotGenerator):
    """Create asset price plot with confidence band."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        strategy_results = get_strategy_results(backtest_results)
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
