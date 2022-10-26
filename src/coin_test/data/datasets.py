"""Define the Dataset classes."""

from abc import ABCMeta, abstractmethod
from collections import Counter

import pandas as pd

from .metadata import MetaData
from .processors import Processor


class DatasetMetaclass(type):
    """Check class defines neccesary attributes.

    Use to enforce the existence of specific attributes after object
    instantiation. Unlike ABCMeta, does _not_ require the usage of @property,
    and therefore works with setters and getter properly.
    """

    def __call__(cls, *args, **kwargs):  # noqa: ANN002,ANN003,ANN204
        """Initialize class then check attributes."""
        instance = super().__call__(*args, *kwargs)
        if not hasattr(instance, "df"):
            raise ValueError("`df` not defined in Dataset!")
        return instance


class Dataset(metaclass=DatasetMetaclass):
    """Load some data into a DataFrame.

    Use to provide the Dataset class a consistent interface to access data
    loaded from different sources. Note that extending classes MUST set the `df`
    attribute in their `__init__` method, else an error will be raised. Also
    note that setting `df` will trigger validation of the new dataframe.
    """

    SCHEMA = {}
    _df: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        """The loaded dataframe."""
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """The loaded dataframe."""
        if not self._validate_df(df):
            raise ValueError(
                f"""
                DataFrame has incorrect column names or column types.
                Expecting {self.SCHEMA}, got {df.dtypes}.
                """
            )
        self._df = df

    @classmethod
    def _validate_df(cls, df: pd.DataFrame) -> bool:
        """Validate dataframe has correct columns.

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if DataFrame has required column names with required types,
                false otherwise
        """
        # TODO: Validate period index
        col_names = Counter(df.columns)
        for required_col, required_type in cls.SCHEMA.items():
            if col_names.get(required_col, 0) != 1:
                return False
            if df[required_col].dtype != required_type:
                return False
        return True

    def process(self, processors: list[Processor]) -> "Dataset":
        """Process the dataset."""
        df = self.df
        for processor in processors:
            df = processor.process(df)
        self.df = df
        return self  # Return self for caller's convenience


class PriceDatasetMetaclass(DatasetMetaclass, ABCMeta):  # noqa: B024
    """Combine DatasetMetaclass and Abstract Base Class metaclass."""


class PriceDataset(Dataset, metaclass=PriceDatasetMetaclass):
    """Load a DataFrame with associated MetaData."""

    SCHEMA = {
        "Open Time": int,
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
        "Close Time": int,
    }

    @property
    @abstractmethod
    def metadata(self) -> MetaData:
        """The dataframe metadata."""


class CustomDataset(PriceDataset):
    """Load a DataFrame in the expected format of price data."""

    def __init__(self, df: pd.DataFrame, asset: str, currency: str) -> None:
        """Initialize a PriceDataLoader.

        Validate DataFrame format correctness and infer timestep interval.

        Args:
            df: DataFrame to validate and infer interval for
            asset: Asset to store in MetaData
            currency: Currency to store in MetaData
        """
        self.df = df

        interval = self._infer_interval(self.df["Open Time"])
        self._metadata = MetaData(asset, currency, interval)

    @staticmethod
    def _infer_interval(timestamps: pd.Series) -> int:
        """Infer interval from dataframe.

        Args:
            timestamps: Series of timestamps in seconds

        Returns:
            int: Inferred interval in seconds

        Raises:
            ValueError: Intervals are not consistent across timesteps
        """
        intervals = timestamps.diff().iloc[1:].astype(int)
        if not all(intervals == intervals.iloc[0]):
            raise ValueError("Timestamps intervals are not consistent")
        return int(intervals.iloc[0])

    @property
    def metadata(self) -> MetaData:
        """Price data metadata."""
        return self._metadata
