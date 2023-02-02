"""Define the TradeRequest abstract class and subclasses."""

from abc import ABC, abstractmethod
from statistics import mean
from typing import cast

import pandas as pd

from .market import SlippageCalculator, TransactionFeeCalculator
from .trade import Trade
from ..util import AssetPair, Side


class TradeRequest(ABC):
    """Request a trade with given specifications."""

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a TradeRequest.

        Args:
            asset_pair: The AssetPair for the asset being traded
            side: The direction of the trade
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None

        Raises:
            ValueError: If arguments are not inputted correctly
        """
        self.asset_pair = asset_pair
        self.side = side
        self.notional = notional
        self.qty = qty

        if notional is not None and qty is not None:
            raise ValueError("Notional and qty cannot be specified together.")
        elif notional is None and qty is None:
            raise ValueError("Must specify either notional or qty")

    # TODO: Refactor to use timestamp data instead of price
    @abstractmethod
    def should_execute(self, price: float) -> bool:
        """Determine if a trade can execute given the current price.

        Args:
            price: The current price of the asset

        Returns:
            bool: True if the trade can execute
        """

    @abstractmethod
    def build_trade(
        self,
        current_asset_price: dict[AssetPair, pd.DataFrame],
        slippage_calculator: SlippageCalculator,
        transaction_fee_calculator: TransactionFeeCalculator,
    ) -> Trade:
        """Build Trade that represents a TradeRequest.

        Args:
            current_asset_price: Current price data from composer
            slippage_calculator: Slippage Calculator implementation
            transaction_fee_calculator: TransactionFeeCalculator implementation

        Returns:
            Trade that the TradeRequest represents
        """

    @staticmethod
    def _calculate_slippage(
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
        slippage_calculator: SlippageCalculator,
    ) -> float:
        """Add slippage to transaction price.

        Args:
            asset_pair: The asset pair for the trade
            side: The side for the trade
            current_asset_price: Current price data from composer
            slippage_calculator: Slippage Calculator implementation

        Returns:
            float: The slippage-adjusted rate for the transaction.
        """
        curr_price = current_asset_price[asset_pair]
        average_price = mean(curr_price[["Open", "High", "Low", "Close"]].iloc[0])

        slippage = slippage_calculator.calculate(asset_pair, side, current_asset_price)
        transaction_price = average_price + slippage
        return transaction_price


class MarketTradeRequest(TradeRequest):
    """A TradeRequest implementation for market (GTC) orders."""

    def should_execute(self, price: float) -> bool:
        """A MarketTrade object should always execute."""
        return True

    def build_trade(
        self,
        current_asset_price: dict[AssetPair, pd.DataFrame],
        slippage_calculator: SlippageCalculator,
        transaction_fee_calculator: TransactionFeeCalculator,
    ) -> Trade:
        """Build Trade that represents a TradeRequest for a MarketTradeRequest.

        Args:
            current_asset_price: Current price data from composer
            slippage_calculator: Slippage Calculator implementation
            transaction_fee_calculator: TransactionFeeCalculator implementation

        Returns:
            Trade that the TradeRequest represents
        """
        price = TradeRequest._calculate_slippage(
            self.asset_pair, self.side, current_asset_price, slippage_calculator
        )

        amount = self.qty
        if amount is None:
            amount = cast(float, self.notional) / price

        transaction_fee = transaction_fee_calculator(self.asset_pair, amount, price)

        return Trade(self.asset_pair, self.side, amount, price, transaction_fee)


class LimitTradeRequest(MarketTradeRequest):
    """A TradeRequest implementation for limit orders.

    If buying, buy when the current price is less than the limit price.
    If selling, sell when the current price is greater than the limit price.
    """

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        limit_price: float,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a LimitTradeRequest.

        Args:
            asset_pair: The TradingPair for the asset being traded
            side: The direction of the trade
            limit_price: The limit price for triggering the trade
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None
        """
        super().__init__(asset_pair, side, notional, qty)
        self.limit_price = limit_price

    def should_execute(self, price: float) -> bool:
        """Execute when the limit price condition is reached."""
        if self.side == Side.BUY:
            return self.limit_price > price
        else:
            # self.side == Side.SELL
            return self.limit_price < price


class StopLimitTradeRequest(MarketTradeRequest):
    """A TradeRequest implementation for stop limit orders.

    If buying, buy when the current price is greater than the limit price.
    If selling, sell when the current price is less than the limit price.
    """

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        stop_limit_price: float,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a LimitTradeRequest.

        Args:
            asset_pair: The TradingPair for the asset being traded
            side: The direction of the trade
            stop_limit_price: The limit price for triggering the trade
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None
        """
        super().__init__(asset_pair, side, notional, qty)
        self.stop_limit_price = stop_limit_price

    def should_execute(self, price: float) -> bool:
        """Execute when the stop limit price condition is reached."""
        if self.side == Side.BUY:
            return self.stop_limit_price < price
        else:
            # self.side == Side.SELL
            return self.stop_limit_price > price
