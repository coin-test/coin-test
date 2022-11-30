"""Define the Trade class."""

from ..util import AssetPair, Side


class Trade:
    """Store the details of a trade."""

    def __init__(
        self,
        asset_pair: AssetPair,
        side: Side,
        amount: float,
        price: float,
        transaction_fee: float = 0,
    ) -> None:
        """Initialize a Trade object.

        Args:
            asset_pair: The asset pair being traded
            side: The direction of the trade
            amount: The number of shares of the asset
            price: The price per share of the asset
            transaction_fee: The trasaction fee, always in the base currency
        """
        self.asset_pair = asset_pair
        self.side = side
        self.amount = amount
        self.price = price
        self.transaction_fee = transaction_fee
