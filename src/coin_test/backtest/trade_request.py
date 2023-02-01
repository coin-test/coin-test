"""Define the TradeRequest abstract class and subclasses."""

from abc import ABC, abstractmethod
from statistics import mean
from typing import cast, Type

import pandas as pd

from .trade import Trade
from ..util import AssetPair, Side


class SlippageCalculator(ABC):
    """Calculate the slippage of an asset."""

    @staticmethod
    @abstractmethod
    def calculate(
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> float:
        """Calculate slippage for current a current asset.

        Args:
            asset_pair: The asset pair for the trade
            side: The side for the trade
            current_asset_price: Current price data from composer

        Returns:
            float: The slippage for the transaction.
        """


class ConstantSlippage(SlippageCalculator):
    """A Constant slippage Calculator."""

    @staticmethod
    def calculate(
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> float:
        """Calculate slippage for current asset.

        Args:
            asset_pair: The asset pair for the trade
            side: The side for the trade
            current_asset_price: Current price data from composer

        Returns:
            float: The slippage for the transaction.
        """
        curr_price = current_asset_price[asset_pair]
        average_price = mean(curr_price[["Open", "High", "Low", "Close"]].iloc[0])

        BASIS_POINT_ADJ = 10

        if side == Side.BUY:
            return average_price * (BASIS_POINT_ADJ / 10000)
        else:
            return average_price * (-BASIS_POINT_ADJ / 10000)


class TradeRequest(ABC):
    """Request a trade with given specifications."""

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        slippage_calculator: Type[SlippageCalculator],
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a TradeRequest.

        Args:
            asset_pair: The AssetPair for the asset being traded
            side: The direction of the trade
            slippage_calculator: Slippage Calculator implementation
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
        self.slippage_calculator = slippage_calculator

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
    def build_trade(self, current_asset_price: dict[AssetPair, pd.DataFrame]) -> Trade:
        """Build Trade that represents a TradeRequest.

        Args:
            current_asset_price: Current price data from composer

        Returns:
            Trade that the TradeRequest represents
        """

    @staticmethod
    def _calculate_slippage(
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
        slippage_calculator: Type[SlippageCalculator],
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

    @staticmethod
    def _generate_transaction_fee(amount: float, adjusted_price: float) -> float:
        """Generate a transaction fee for a given trade request.

        Args:
            amount: The quantity of the currency being traded
            adjusted_price: Price of the trade

        Returns:
            float: The transaction fee in the base currency.
        """
        TRANSACTION_FEE_BP = 50

        return amount * adjusted_price * (TRANSACTION_FEE_BP / 10000)


class MarketTradeRequest(TradeRequest):
    """A TradeRequest implementation for market (GTC) orders."""

    def should_execute(self, price: float) -> bool:
        """A MarketTrade object should always execute."""
        return True

    def build_trade(self, current_asset_price: dict[AssetPair, pd.DataFrame]) -> Trade:
        """Build Trade that represents a TradeRequest for a MarketTradeRequest.

        Args:
            current_asset_price: Current price data from composer

        Returns:
            Trade that the TradeRequest represents
        """
        price = TradeRequest._calculate_slippage(
            self.asset_pair, self.side, current_asset_price, self.slippage_calculator
        )

        amount = self.qty
        if amount is None:
            amount = cast(float, self.notional) / price

        transaction_fee = TradeRequest._generate_transaction_fee(amount, price)

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
        slippage_calculator: Type[SlippageCalculator],
        limit_price: float,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a LimitTradeRequest.

        Args:
            asset_pair: The TradingPair for the asset being traded
            side: The direction of the trade
            slippage_calculator: Slippage Calculator implementation
            limit_price: The limit price for triggering the trade
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None
        """
        super().__init__(asset_pair, side, slippage_calculator, notional, qty)
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
        slippage_calculator: Type[SlippageCalculator],
        stop_limit_price: float,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a LimitTradeRequest.

        Args:
            asset_pair: The TradingPair for the asset being traded
            side: The direction of the trade
            slippage_calculator: Slippage Calculator implementation
            stop_limit_price: The limit price for triggering the trade
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None
        """
        super().__init__(asset_pair, side, slippage_calculator, notional, qty)
        self.stop_limit_price = stop_limit_price

    def should_execute(self, price: float) -> bool:
        """Execute when the stop limit price condition is reached."""
        if self.side == Side.BUY:
            return self.stop_limit_price < price
        else:
            # self.side == Side.SELL
            return self.stop_limit_price > price
