"""Backtesting module of the coin-test library."""

from .market import ConstantSlippage, SlippageCalculator
from .portfolio import Portfolio
from .simulator import Simulator
from .strategy import Strategy
from .trade import Trade
from .trade_request import (
    LimitTradeRequest,
    MarketTradeRequest,
    StopLimitTradeRequest,
    TradeRequest,
)

__all__ = [
    "ConstantSlippage",
    "LimitTradeRequest",
    "MarketTradeRequest",
    "Portfolio",
    "SlippageCalculator",
    "Simulator",
    "StopLimitTradeRequest",
    "Strategy",
    "Trade",
    "TradeRequest",
]
