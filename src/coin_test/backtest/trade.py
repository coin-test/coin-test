"""Trade object."""

from datetime import datetime


class Trade:
    """An object containing the details of a trade."""

    def __init__(
        self, symbol: str, side: str, price: float, timestamp: datetime
    ) -> None:
        """Initialize a Trade object.

        Args:
            symbol: The symbol of the asset being traded
            side: buy or sell
            price: The price per share of the asset
            timestamp: When the trade takes place
        """
        self.symbol = symbol
        self.side = side
        self.price = price
        self.timestamp = timestamp
