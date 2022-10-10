"""Dataset object."""

import pandas as pd
from pandas import DataFrame

from .processors import Processor


class Dataset:
    """Object that manages a single dataset."""

    def __init__(
        self, src: str, asset: str, currency: str, processors: list[Processor]
    ) -> None:
        """Intialize a dataset.

        Args:
            src: CSV to load data from. Can be a URL or a file path
            asset: Asset being traded
            currency: Currency being traded against
            processors: List of Processor objects to be used to transform the
                loaded dataframe
        """
        self.asset = asset
        self.currency = currency

        df = self._load_df(src)
        self.df = self._clean(df, processors)

        self.interval = None

    def _load_df(self, src: str) -> DataFrame:
        """Load the dataframe from src."""
        return pd.read_csv(src)

    def _clean(self, df: pd.DataFrame, processors: list[Processor]) -> pd.DataFrame:
        """Clean the dataframe."""
        for processor in processors:
            df = processor.process(df)
        return df
