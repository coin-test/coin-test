"""Define the Loader classes."""

from abc import ABC, abstractmethod
from collections import Counter

import pandas as pd

from .metadata import MetaData


REQUIRED_COLS = {
    "Open Time": int,
    "Open": float,
    "High": float,
    "Low": float,
    "Close": float,
    "Volume": float,
    "Close Time": int,
}


class Loader(ABC):
    """Load some data into a DataFrame.

    Use to provide the Dataset class a consistent interface to access data
    loaded from different sources.
    """

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """The loaded dataframe."""


class PriceDataLoader(Loader):
    """Load a DataFrame with associated MetaData.

    DataFrame contained in PriceDataLoader expected to be validated using
    _validate_df().
    """

    @property
    @abstractmethod
    def metadata(self) -> MetaData:
        """The dataframe metadata."""

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> bool:
        """Validate dataframe has correct columns.

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if DataFrame has required column names with required types,
                false otherwise
        """
        col_names = Counter(df.columns)
        for required_col, required_type in REQUIRED_COLS.items():
            if col_names.get(required_col, 0) != 1:
                return False
            if df[required_col].dtype != required_type:
                return False
        return True


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


class PriceDataFrameLoader(PriceDataLoader):
    """Load a DataFrame in the expected format of price data."""

    def __init__(self, df: pd.DataFrame, asset: str, currency: str) -> None:
        """Initialize a PriceDataLoader.

        Validate DataFrame format correctness and infer timestep interval.

        Args:
            df: DataFrame to validate and infer interval for
            asset: Asset to store in MetaData
            currency: Currency to store in MetaData

        Raises:
            ValueError: DataFrame column names or types are incorrect
        """
        if not self._validate_df(df):
            raise ValueError(
                f"""
                DataFrame has incorrect column names or column types.
                Expecting {REQUIRED_COLS}, got {df.dtypes}.
                """
            )
        interval = self._infer_interval(df["Open Time"])

        self._df = df
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
    def df(self) -> pd.DataFrame:
        """Validated price data DataFrame."""
        return self._df

    @property
    def metadata(self) -> MetaData:
        """Price data metadata."""
        return self._metadata
