"""Define the TradeRequest abstract class and subclasses."""

from abc import ABC, abstractmethod

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
            asset_pair: The TradingPair for the asset being traded
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

    @abstractmethod
    def can_execute(self, price: float) -> bool:
        """Determine if a trade can execute given the current price.

        Args:
            price: The current price of the asset

        Returns:
            bool: True if the trade can execute
        """


class MarketTradeRequest(TradeRequest):
    """A TradeRequest implementation for market (GTC) orders."""

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a MarketTradeRequest."""
        super().__init__(asset_pair, side, notional, qty)

    def can_execute(self, price: float) -> bool:
        """A MarketTrade object should always execute."""
        return True


class LimitTradeRequest(TradeRequest):
    """A TradeRequest implementation for limit orders.

    If buying, buy when the current price is less than the limit price.
    If selling, sell when the current price is higher than the limit price.
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

    def can_execute(self, price: float) -> bool:
        """Execute when the limit price condition is reached."""
        if self.side == Side.BUY:
            return self.limit_price > price
        else:
            # self.side == Side.SELL
            return self.limit_price < price
