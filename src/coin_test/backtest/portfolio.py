"""Create a Portfolio class."""


class Portfolio:
    """A class that manages a portfolio."""

    def __init__(self, cash: float = 0, assets: dict[str, float] | None = None) -> None:
        """Initialize a Portfolio.

        Args:
            cash: The amount of cash in the portfolio
            assets: A dictionary of all assets and quantity
        """
        self.cash = cash
        self.assets = assets if assets is not None else {}

    # Currently just returns cash TODO: support shorting
    def get_free_cash(self) -> float:
        """Returns the free cash available in the portfolio."""
        return self.cash
