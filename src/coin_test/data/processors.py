"""Define the Processor classes."""

from abc import ABC, abstractmethod

from pandas import DataFrame


class Processor(ABC):
    """Transform a DataFrame."""

    @abstractmethod
    def process(self, df: DataFrame) -> DataFrame:
        """Process a dataframe."""


class IdentityProcessor(Processor):
    """Identity Transform."""

    def process(self, df: DataFrame) -> DataFrame:
        """Identity tranfsorm."""
        return df
