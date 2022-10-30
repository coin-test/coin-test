"""Define the MetaData class."""

from typing import NamedTuple


class MetaData(NamedTuple):
    """Historical data metadata."""

    asset: str
    currency: str
    freq: str
