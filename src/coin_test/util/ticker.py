"""Define ticker classes for use in the coin-test package."""

from typing import NamedTuple


class Ticker:
    """Represent an asset."""

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
                f"""Expecting symbol to contain more than whitespace got: '{symbol}'."""
            )
        self.symbol = cleaned_symbol

    def __eq__(self, other: "Ticker") -> bool:
        """Check equality between symbols."""
        if not isinstance(other, Ticker):
            raise NotImplementedError
        else:
            return self.symbol == other.symbol

    def __hash__(self) -> int:
        """Hash a ticker based on the symbol name."""
        return hash(self.symbol)

    def __repr__(self) -> str:
        """Build string representation."""
        return f'Ticker("{self.symbol}")'


class AssetPair(NamedTuple):
    """Pair of tickers that can be traded."""

    asset: Ticker
    currency: Ticker
