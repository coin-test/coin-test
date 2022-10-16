"""Ticker objects for use in the coin-test package."""


class Ticker:
    """Represents an asset."""

    def __init__(self, symbol: str) -> None:
        """Initialize a Ticker.

        Args:
            symbol: Symbol of an asset

        Raises:
            ValueError: Invalid symbol
        """
        cleaned_symbol = symbol.strip().lower()
        if len(cleaned_symbol) == 0:
            raise ValueError(
                f"""Expecting symbol to contain more than whitespace got: {symbol}."""
            )
        self.symbol = cleaned_symbol

    def __eq__(self, other: "Ticker") -> bool:
        """Check equality between symbols."""
        if not isinstance(other, Ticker):
            raise NotImplementedError
        else:
            return self.symbol == other.symbol
