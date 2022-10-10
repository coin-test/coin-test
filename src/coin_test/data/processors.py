"""Dataset processors."""

from abc import ABC, abstractmethod

from pandas import DataFrame


class Processor(ABC):
    """A base class for processors that transform a dataframe."""

    @abstractmethod
    def process(self, df: DataFrame) -> DataFrame:
        """Process a dataframe."""
        pass


class IdentityProcessor(Processor):
    """Identify processor for tests."""

    def process(self, df: DataFrame) -> DataFrame:
        """Identify tranfsorm."""
        return df
