"""Define the Trade class."""

from datetime import datetime

from ..util import Side


class Trade:
    """Store the details of a trade."""

    def __init__(
        self, symbol: str, side: Side, price: float, timestamp: datetime
    ) -> None:
        """Initialize a Trade object.

        Args:
            symbol: The symbol of the asset being traded
            side: The direction of the trade
            price: The price per share of the asset
            timestamp: When the trade takes place
        """
        self.symbol = symbol
        self.side = side
        self.price = price
        self.timestamp = timestamp
