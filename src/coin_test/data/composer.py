"""Define the Dataset class."""

import pandas as pd

from .loaders import PriceDataLoader
from .processors import Processor


class Composer:
    """Manages a dataset."""

    def __init__(
        self,
        price_loader: PriceDataLoader,
        processors: list[Processor],
    ) -> None:
        """Intialize a dataset.

        Args:
            price_loader: PriceDataLoader containing price data DataFrame
                and asset MetaData
            processors: List of Processor objects to transform the data
        """
        self.df = self._clean(price_loader.df, processors)
        self.metadata = price_loader.metadata

    @staticmethod
    def _clean(df: pd.DataFrame, processors: list[Processor]) -> pd.DataFrame:
        """Clean the dataframe."""
        for processor in processors:
            df = processor.process(df)
        return df
