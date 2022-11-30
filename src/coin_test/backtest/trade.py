"""Define the Trade class."""

from ..util import AssetPair, Side


class Trade:
    """Store the details of a trade."""

    def __init__(
        self, asset_pair: AssetPair, side: Side, amount: float, price: float
    ) -> None:
        """Initialize a Trade object.

        Args:
            asset_pair: The asset pair being traded
            side: The direction of the trade
            amount: The number of shares of the asset
            price: The price per share of the asset
        """
        self.asset_pair = asset_pair
        self.side = side
        self.amount = amount
        self.price = price

    def __repr__(self) -> str:
        """Build string representation."""
        currency_amt = self.amount * self.price
        if self.side == Side.BUY:
            return (
                f"{currency_amt} {self.asset_pair.currency} -> "
                f"{self.amount} {self.asset_pair.asset}"
            )
        else:
            return (
                f"{self.amount} {self.asset_pair.asset} -> "
                f"{currency_amt} {self.asset_pair.currency}"
            )
