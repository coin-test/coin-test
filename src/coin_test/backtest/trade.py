"""Define the Trade class."""

import datetime as dt

from ..util import AssetPair, Side


class Trade:
    """Store the details of a trade."""

    def __init__(
        self, asset_pair: AssetPair, side: Side, price: float, timestamp: dt.datetime
    ) -> None:
        """Initialize a Trade object.

        Args:
            asset_pair: The asset pair being traded
            side: The direction of the trade
            price: The price per share of the asset
            timestamp: When the trade takes place
        """
        self.asset_pair = asset_pair
        self.side = side
        self.price = price
        self.timestamp = timestamp
