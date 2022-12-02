"""Define the Portfolio class."""

from typing import Union

from .trade import Trade
from ..util import Money, Side, Ticker


class Portfolio:
    """Manage a portfolio."""

    def __init__(self, base_currency: Ticker, assets: dict[Ticker, Money]) -> None:
        """Initialize a Portfolio.

        Args:
            base_currency: The base currency of the portfolio
            assets: A dictionary of all holdings

        Raises:
            ValueError: If the parameters are not inputted properly
        """
        self.base_currency = base_currency
        self.assets = assets if assets is not None else {}
        self.reserved = {a: Money(a, 0) for a in assets}

        if self.base_currency not in self.assets:
            raise ValueError("Base currency must exist in assets.")

        for asset, money in self.assets.items():
            if money.ticker != asset:
                raise ValueError("Money must match ticker in assets.")

    # Currently just returns cash in a given asset TODO: support shorting
    def available_assets(self, asset: Ticker) -> Money:
        """Return the available assets for a given ticker in a portfolio.

        Args:
            asset: The desired ticker

        Returns:
            Money: The amount of money available for the given ticker

        Raises:
            ValueError: If the ticker is not in the portfolio
        """
        if asset not in self.assets:
            raise ValueError("This asset does not exist in the portfolio.")

        return self.assets[asset] - self.reserved[asset]

    def adjust(self, trade: Trade) -> Union["Portfolio", None]:
        """Adjust the portfolio after a given Trade is performed.

        Args:
            trade: The Trade object that is completed

        Returns:
            Portfolio | None: The new adjusted portfolio or None if insufficient Money
        """
        adjusted_assets = self.assets.copy()

        currency = trade.asset_pair.currency
        asset = trade.asset_pair.asset
        if trade.side == Side.BUY:
            # Check for overspending
            if adjusted_assets[currency] < Money(currency, trade.amount * trade.price):
                return None

            adjusted_assets[currency] -= Money(currency, trade.amount * trade.price)
            adjusted_assets[asset] += Money(asset, trade.amount)
        else:
            if adjusted_assets[asset] < Money(asset, trade.amount):
                return None
            adjusted_assets[currency] += Money(currency, trade.amount * trade.price)
            adjusted_assets[asset] -= Money(asset, trade.amount)

        return Portfolio(self.base_currency, adjusted_assets)

    def __repr__(self) -> str:
        """Build string representation."""
        return f"{self.base_currency} - {self.assets}"
