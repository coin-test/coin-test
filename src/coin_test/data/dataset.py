"""Define the Dataset class."""

from collections import Counter

import pandas as pd

from .processors import Processor


REQUIRED_COLS = {
    "Open Time": int,
    "Open": float,
    "High": float,
    "Low": float,
    "Close": float,
    "Volume": float,
    "Close Time": int,
}


class Dataset:
    """Manages a dataset."""

    def __init__(
        self,
        asset: str,
        currency: str,
        processors: list[Processor],
        df: pd.DataFrame,
        interval: int | None = None,
    ) -> None:
        """Intialize a dataset.

        Args:
            asset: Asset being traded
            currency: Currency being traded against
            processors: List of Processor objects to transform the data
            df: Dataframe to validate and clean
            interval: Interval in seconds between timesteps. Inferred if `None`

        Raises:
            ValueError: DataFrame column names or types are incorrect
        """
        self.asset = asset
        self.currency = currency

        if not self.validate_df(df):
            raise ValueError(
                f"""
                Dataframe has incorrect column names or column types.
                Expecting {REQUIRED_COLS}, got {df.dtypes}.
                """
            )
        self.df = self._clean(df, processors)

        if interval is None:
            interval = self.infer_interval(df["Open Time"])
        self.interval = interval

    @staticmethod
    def infer_interval(timestamps: pd.Series) -> int:
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

    @staticmethod
    def validate_df(df: pd.DataFrame) -> bool:
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

    @staticmethod
    def _clean(df: pd.DataFrame, processors: list[Processor]) -> pd.DataFrame:
        """Clean the dataframe."""
        for processor in processors:
            df = processor.process(df)
        return df
