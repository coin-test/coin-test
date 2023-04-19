"""Generate plots for analysis."""

from typing import Sequence

import datapane as dp
import numpy as np
import pandas as pd
from plotly.colors import n_colors
import plotly.graph_objects as go

from coin_test.backtest import TradeRequest
from coin_test.backtest.backtest_results import BacktestResults
from coin_test.util import Side
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from .utils import is_single_strategy


def _get_buy(trades: list[TradeRequest]) -> list[TradeRequest]:
    return [trade for trade in trades if trade.side is Side.BUY]


def _get_sell(trades: list[TradeRequest]) -> list[TradeRequest]:
    return [trade for trade in trades if trade.side is Side.SELL]


def _categorize_trades(trades: pd.Series) -> tuple[pd.Series, pd.Series]:
    buys = trades.apply(_get_buy)
    sells = trades.apply(_get_sell)
    return buys, sells


def _build_window_traces(
    backtest_results: BacktestResults,
    trades: pd.Series,
    lookback: pd.Timedelta,
    plot_params: PlotParameters,
) -> tuple[list[int], np.ndarray] | None:
    num_trades = trades.apply(len)
    trade_times = backtest_results.sim_data.index[num_trades >= 1]
    if len(trade_times) == 0:
        return
    price = list(backtest_results.data_dict.values())[0]["Open"]

    price_start, price_end = min(price.index), max(price.index)
    trades_start, trades_end = min(trade_times), max(trade_times)
    graph_start, graph_end = trades_start - lookback, trades_end + lookback
    before, after = None, None

    if graph_start < price_start.start_time:
        before_idx = pd.period_range(
            start=graph_start, end=price_start, freq=price.index.freq  # type: ignore
        )[:-1]
        before = pd.Series(index=before_idx, dtype=price.dtype)
    if graph_end > price_end.end_time:
        after_idx = pd.period_range(
            start=price_end, end=graph_end, freq=price.index.freq  # type: ignore
        )[1:]
        after = pd.Series(index=after_idx, dtype=price.dtype)
    price: pd.Series = pd.Series, pd.concat((before, price, after))  # type: ignore

    def _slice_data(timestamp: pd.Timestamp, price: pd.Series = price) -> pd.Series:
        y = price[timestamp - lookback : timestamp + lookback]
        norm = y[timestamp:timestamp].iloc[0]
        y = y.reset_index(drop=True)
        return y / norm

    sliced_data = pd.Series(trade_times).apply(_slice_data)
    y = sliced_data.to_numpy().flatten()
    shift = max(sliced_data.columns) // 2
    x = [v - shift for v in sliced_data.columns.to_list() * len(sliced_data)]
    return x, y


class SignalHeatmapPlot(DistributionalPlotGenerator):
    """Create Price plot around trade signals."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)

        buy_data = {"x": [], "y": [], "name": "Buy"}
        sell_data = {"x": [], "y": [], "name": "Sell"}
        lookback = max(backtest_results[0].strategy_lookbacks)
        for results in backtest_results:
            buys, sells = _categorize_trades(results.sim_data["Trades"])
            for trades, data in ((buys, buy_data), (sells, sell_data)):
                points = _build_window_traces(results, trades, lookback, plot_params)
                if points is not None:
                    data["x"].extend(points[0])
                    data["y"].extend(points[1])

        # Build log colorscale
        color_len = 5
        colors = n_colors(
            "rgb(200, 200, 200)",
            "rgb(0, 0, 0)",
            color_len,
            colortype="rgb",
        )
        colorscale = [[0, "rgba(255, 255, 255, 0)"]]
        for i, color in enumerate(colors):
            colorscale.append([1 / (5 ** (color_len - i - 1)), color])

        for data in (buy_data, sell_data):
            fig = go.Figure(
                go.Histogram2d(
                    x=data["x"],
                    y=data["y"],
                    xbins=dict(size=1),
                    nbinsy=50,
                    colorscale=colorscale,
                )
            )
            PlotParameters.update_plotly_fig(
                plot_params,
                fig,
                data["name"] + " Patterns",
                "Timesteps",
                "Normalized Asset Value",
            )
            fig.add_vline(
                x=0,
                line_dash="dot",
                annotation_text=data["name"] + " Signal",
                annotation_position="bottom right",
            )
            data["fig"] = fig

        return dp.Select(
            dp.Media(
                plot_params.compress_fig(buy_data["fig"], name="buy_patterns"),
                label="Buy Patterns",
            ),
            dp.Media(
                plot_params.compress_fig(sell_data["fig"], name="sell_patterns"),
                label="Sell Patterns",
            ),
        )


def _build_buy_sell_overlay_price(
    backtest_results: BacktestResults, plot_params: PlotParameters
) -> go.Figure:
    start_time = backtest_results.sim_data.index[1]
    price = list(backtest_results.data_dict.values())[0]["Open"][start_time:]
    index = price.index.map(lambda t: t.start_time)
    fig = go.Figure(
        go.Scatter(
            y=price,
            x=index,
            mode="lines",
            marker=dict(color=plot_params.line_colors[0]),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    buys, sells = _categorize_trades(backtest_results.sim_data["Trades"])

    def _build_scatter(trades: pd.Series, name: str, color: str, shape: str) -> None:
        num_trades = trades.apply(len)
        trade_times = backtest_results.sim_data.index[num_trades >= 1]
        asset_value = [
            price[trade_time:trade_time].iloc[0] for trade_time in trade_times
        ]
        fig.add_trace(
            go.Scatter(
                x=trade_times,
                y=asset_value,
                mode="markers",
                showlegend=True,
                hoverinfo="skip",
                marker=dict(color=color, symbol=shape, size=15),
                name=name,
            )
        )

    _build_scatter(buys, "Buy", plot_params.line_colors[2], "triangle-up")
    _build_scatter(sells, "Sell", plot_params.line_colors[1], "triangle-down")

    return fig


class SignalTotalPlot(DistributionalPlotGenerator):
    """Create Price plot around trade signals."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)
        figures = [
            _build_buy_sell_overlay_price(results, plot_params)
            for results in backtest_results[:10]
        ]

        for i, fig in enumerate(figures[:10]):
            PlotParameters.update_plotly_fig(
                plot_params,
                fig,
                f"Dataset {i}",
                "Time",
                "Asset Price",
            )

        return dp.Select(
            *[
                dp.Media(
                    plot_params.compress_fig(fig, name="buy_sell_price"),
                    label=f"Dataset {i}",
                )
                for i, fig in enumerate(figures)
            ]
        )
