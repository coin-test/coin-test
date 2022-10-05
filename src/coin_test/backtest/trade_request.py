"""Trade request object."""


class TradeRequest:
    """A request object for creating a trade."""

    def __init__(self, symbol: str = "BTC") -> None:
        """Create a new TradeRequest object.

        Args:
            symbol: The symbol of the asset being traded

        """
        self.symbol = symbol
