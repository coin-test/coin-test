"""Enum objects for use in the coin-test package."""

from enum import Enum


class Side(Enum):
    """The side for a trade.

    BUY, SELL
    """

    SELL = 0
    BUY = 1


class TradeType(Enum):
    """The type of trade being performed.

    Currently the only option is MARKET
    """

    MARKET = 0
