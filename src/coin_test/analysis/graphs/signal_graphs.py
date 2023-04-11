"""Generate plots for analysis."""

from typing import Sequence

import datapane as dp
import pandas as pd
import plotly.graph_objects as go

from coin_test.backtest import TradeRequest
from coin_test.backtest.backtest_results import BacktestResults
from coin_test.util import Side
from .base_classes import DistributionalPlotGenerator, PLOT_RETURN_TYPE
from .plot_parameters import PlotParameters
from .utils import clamp, get_lims, is_single_strategy


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
) -> list[go.Scatter]:
    num_trades = trades.apply(len)
    trade_times = backtest_results.sim_data.index[num_trades >= 1]
    if len(trade_times) == 0:
        return []
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
    price: pd.Series = pd.concat((before, price, after))  # type: ignore

    def _slice_data(timestamp: pd.Timestamp, price: pd.Series = price) -> pd.Series:
        ret = pd.Series(dtype=object)
        y = price[timestamp - lookback : timestamp + lookback]
        norm = y[timestamp:timestamp].iloc[0]
        y = y.reset_index(drop=True)
        ret["y"] = y / norm
        return ret

    sliced_data = pd.Series(trade_times).apply(_slice_data)

    def _build_traces(sliced_data: pd.Series) -> go.Scatter:
        return go.Scatter(
            y=sliced_data.y,
            mode="lines",
            marker=dict(color=plot_params.line_colors[0]),
            showlegend=False,
            hoverinfo="skip",
        )

    return sliced_data.apply(_build_traces, axis=1).tolist()


class SignalWindowPlot(DistributionalPlotGenerator):
    """Create Price plot around trade signals."""

    @staticmethod
    def create(
        backtest_results: Sequence[BacktestResults], plot_params: PlotParameters
    ) -> PLOT_RETURN_TYPE:
        """Create plot object."""
        is_single_strategy(backtest_results)

        buy_traces, sell_traces = [], []
        lookback = max(backtest_results[0].strategy_lookbacks)
        for results in backtest_results:
            buys, sells = _categorize_trades(results.sim_data["Trades"])
            buy_traces.extend(
                _build_window_traces(results, buys, lookback, plot_params)
            )
            sell_traces.extend(
                _build_window_traces(results, sells, lookback, plot_params)
            )

        buy_fig = go.Figure(buy_traces)
        sell_fig = go.Figure(sell_traces)
        x_lims, y_lims = get_lims((buy_fig, sell_fig))
        mid = (x_lims[1] - x_lims[0]) // 2
        buy_opacity = clamp(1 / len(buy_traces) * 2, 0.05, 0.5) if buy_traces else 1
        sell_opacity = clamp(1 / len(sell_traces) * 2, 0.05, 0.5) if sell_traces else 1
        opacity = min(buy_opacity, sell_opacity)

        for fig, name in ((buy_fig, "Buy"), (sell_fig, "Sell")):
            PlotParameters.update_plotly_fig(
                plot_params,
                fig,
                name + " Patterns",
                "Timesteps",
                "Normalized Asset Value",
            )
            fig.update_yaxes(range=y_lims)
            fig.add_vline(
                x=mid,
                line_dash="dot",
                annotation_text=name + " Signal",
                annotation_position="bottom right",
            )
            fig.update_traces(opacity=opacity)

        return dp.Select(
            dp.Media(
                plot_params.compress_fig(buy_fig, name="buy_patterns"),
                label="Buy Patterns",
            ),
            dp.Media(
                plot_params.compress_fig(sell_fig, name="sell_patterns"),
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

    def _build_vlines(trades: pd.Series, name: str, color: str) -> None:
        num_trades = trades.apply(len)
        trade_times = backtest_results.sim_data.index[num_trades >= 1]
        for trade_time in trade_times:
            fig.add_vline(
                x=trade_time,
                line_dash="dot",
                line_color=color,
            )
        # Dummy line to generate legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                showlegend=True,
                hoverinfo="skip",
                marker=dict(color=color),
                name=name,
            )
        )

    _build_vlines(buys, "Buy", plot_params.line_colors[2])
    _build_vlines(sells, "Sell", plot_params.line_colors[1])

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
