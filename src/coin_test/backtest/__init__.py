"""Backtesting module of the coin-test library."""

from .backtest_results import BacktestResults
from .market import (
    ConstantSlippage,
    ConstantTransactionFeeCalculator,
    GaussianSlippage,
    SlippageCalculator,
    TransactionFeeCalculator,
)
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
    "BacktestResults",
    "ConstantSlippage",
    "ConstantTransactionFeeCalculator",
    "GaussianSlippage",
    "LimitTradeRequest",
    "MarketTradeRequest",
    "Portfolio",
    "SlippageCalculator",
    "Simulator",
    "StopLimitTradeRequest",
    "Strategy",
    "Trade",
    "TradeRequest",
    "TransactionFeeCalculator",
]
