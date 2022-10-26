"""Define the Portfolio class."""

from ..util import Money, Ticker


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
