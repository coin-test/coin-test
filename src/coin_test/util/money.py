"""Initializes the Money class."""

from .ticker import Ticker


class Money:
    """Amount of money in a given currency."""

    def __init__(self, currency: Ticker, qty: float) -> None:
        """Initialize a Money object.

        Args:
            currency: The currency the money is in.
            qty: The amount of money that exists.
        """
        self.currency = currency
        self.qty = qty

    def _can_compare(self, other: "Money") -> None:
        """Check if Money objects can be compared."""
        if not isinstance(other, Money):
            raise NotImplementedError

        if self.currency != other.currency:
            raise ValueError("Can't compare equality across different tickers")

    def __eq__(self, other: "Money") -> bool:
        """Check for equality between Money objects."""
        self._can_compare(other)

        return self.qty == other.qty

    def __gt__(self, other: "Money") -> bool:
        """Check if this Money object is greater than another."""
        self._can_compare(other)

        return self.qty > other.qty

    def __lt__(self, other: "Money") -> bool:
        """Check if this Money object is less than another."""
        return not (self > other or self == other)

    def __ne__(self, other: "Money") -> bool:
        """Check if this Money object is not equal to another."""
        return not (self == other)

    def __add__(self, other: "Money") -> "Money":
        """Add two amounts of Money."""
        self._can_compare(other)

        return Money(self.currency, self.qty + other.qty)

    def __sub__(self, other: "Money") -> "Money":
        """Subtract two amounts of Money."""
        self._can_compare(other)

        return Money(self.currency, self.qty - other.qty)
