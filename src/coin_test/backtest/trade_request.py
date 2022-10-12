"""Define the TradeRequest class."""

from ..util import Side, TradeType


class TradeRequest:
    """Request a trade with given specifications."""

    def __init__(
        self,
        symbol: str,
        side: Side,
        type_: TradeType = TradeType.MARKET,
        notional: float | None = None,
        qty: float | None = None,
    ) -> None:
        """Initialize a TradeRequest.

        Args:
            symbol: The symbol of the asset being traded
            side: The direction of the trade
            type_: The type of trade, default is TradeType.MARKET
            notional: The amount of money to trade, default None
            qty: The amount of shares to trade, can't be used with notional,
                default None

        Raises:
            ValueError: If arguments are not inputted correctly
        """
        self.symbol = symbol
        self.side = side
        self.type_ = type_
        self.notional = notional
        self.qty = qty

        if notional is not None and qty is not None:
            raise ValueError("Notional and qty cannot be specified together.")
        elif notional is None and qty is None:
            raise ValueError("Must specify either notional or qty")
