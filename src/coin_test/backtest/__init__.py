"""Backtesting module of the coin-test library."""

from .portfolio import Portfolio
from .simulator import Simulator
from .strategy import TestStrategy, UserDefinedStrategy
from .trade import Trade
from .trade_request import (
    LimitTradeRequest,
    MarketTradeRequest,
    StopLimitTradeRequest,
    TradeRequest,
)

__all__ = [
    "LimitTradeRequest",
    "MarketTradeRequest",
    "Portfolio",
    "Simulator",
    "StopLimitTradeRequest",
    "UserDefinedStrategy",
    "TestStrategy",
    "Trade",
    "TradeRequest",
]
