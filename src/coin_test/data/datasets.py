"""Define the Dataset classes."""

from abc import ABCMeta, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

from .metadata import MetaData
from .processors import Processor
from ..util import AssetPair


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

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Validate setting of `df` attribute."""
        if name == "df" and not self._validate_df(value):
            raise ValueError(
                f"""
                DataFrame has incorrect column names or column types.
                Expecting {self.SCHEMA}, got {value.dtypes}.
                """
            )
        super().__setattr__(name, value)

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
            df = processor(df)
        self.df = df
        return self  # Return self for caller's convenience


class PriceDatasetMetaclass(DatasetMetaclass, ABCMeta):  # noqa: B024
    """Combine DatasetMetaclass and Abstract Base Class metaclass."""


class PriceDataset(Dataset, metaclass=PriceDatasetMetaclass):
    """Load a DataFrame with associated MetaData."""

    SCHEMA = {
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
    }
    OPEN_TIME_NAME = "Open Time"

    @property
    @abstractmethod
    def metadata(self) -> MetaData:
        """The dataframe metadata."""


class CustomDataset(PriceDataset):
    """Load a DataFrame in the expected format of price data."""

    def __init__(self, df: pd.DataFrame, freq: str, pair: AssetPair) -> None:
        """Initialize a PriceDataLoader.

        Validate DataFrame format correctness and infer timestep interval.

        Args:
            df: DataFrame to load
            freq: Pandas period string representing price data interval
            pair: AssetPair represented in datal
        """
        self.df = self._add_period_index(df, freq)
        self._metadata = MetaData(pair, freq)

    @classmethod
    def _add_period_index(cls, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Add a Period index to passed Dataframe.

        Immediately return if index already exists. Otherwise, check for
        existence of cls.OPEN_TIME_NAME column, then build Period index based on
        column type.

        Args:
            df: DataFrame to add Period index to.
            freq: Pandas period string representing price data inteval

        Returns:
            DataFrame: DataFrame with Period index.

        Raises:
            ValueError: Column is missing or type is un-supported.
        """
        if isinstance(df.index, pd.PeriodIndex):
            return df
        if cls.OPEN_TIME_NAME not in df.columns:
            raise ValueError(
                f"""
                Custom price DataFrame requires either a PeriodIndex or a column
                of name "{cls.OPEN_TIME_NAME}".
                """
            )

        series = df[cls.OPEN_TIME_NAME]
        if (
            pd.api.types.is_period_dtype(series.dtype)
            or pd.api.types.is_datetime64_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            # Types that pandas can auto-infer from
            index = pd.PeriodIndex(data=series, freq=freq)  # type: ignore
        elif series.dtype == int:
            # If int, assume time since epoch
            index = pd.PeriodIndex(
                data=[datetime.fromtimestamp(d) for d in series],
                freq=freq,  # type: ignore
            )
        else:
            raise ValueError(
                f"""
                Was not able to convert series of type {series.dtype} to a
                pandas PeriodIndex.
                """
            )
        return df.set_index(index, verify_integrity=True)

    @property
    def metadata(self) -> MetaData:
        """Price data metadata."""
        return self._metadata
