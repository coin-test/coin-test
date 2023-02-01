"""Define the Processor classes."""

from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame


class Processor(ABC):
    """Transform a DataFrame."""

    @abstractmethod
    def __call__(self, df: DataFrame) -> DataFrame:
        """Process a dataframe."""


class IdentityProcessor(Processor):
    """Identity Transform."""

    def __call__(self, df: DataFrame) -> DataFrame:
        """Identity transform."""
        return df


class FillProcessor(Processor):
    """Fill missing periods for a dataset."""

    def __init__(self, freq: str, method: str | None = "pad") -> None:
        """Initialize a FillProcessor.

        Args:
            freq: Frequency to fill periods for.
            method: Fill method to use in pandas .fillna() function. Must be one
                of {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}.
        """
        self.freq = freq
        self.method = method

    def __call__(self, df: DataFrame) -> DataFrame:
        """Identity tranfsorm."""
        full_index = pd.period_range(
            df.index[0], df.index[-1], freq=self.freq  # type: ignore
        )
        missing_index = full_index.difference(df.index)
        missing_df = pd.DataFrame(index=missing_index, columns=df.columns)
        full_df = pd.concat([df, missing_df]).sort_index()
        full_df = full_df.fillna(method=self.method)  # type: ignore
        return full_df
