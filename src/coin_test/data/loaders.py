"""Define the Loader classes."""

from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    """Load some data into a DataFrame.

    Use to provide the Dataset class a consistent interface to access data
    loaded from different sources.
    """

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """The loaded dataframe."""


class DataFrameLoader(Loader):
    """Load a DataFrame.

    Use when an appropriate DataFrame has already been created. Does no
    loading, but fulfills the Loader interface.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a DataFrame Loader."""
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """Return stored DataFrame."""
        return self._df
