"""Define the MetaData class."""

from typing import NamedTuple

from ..util import AssetPair


class MetaData(NamedTuple):
    """Historical data metadata."""

    pair: AssetPair
    freq: str
