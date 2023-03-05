"""Define the Dataset classes."""

from abc import ABCMeta, abstractmethod
from collections import Counter
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
        instance = super().__call__(*args, **kwargs)
        if not hasattr(instance, "df"):
            raise ValueError("`df` not defined in Dataset!")
        if not hasattr(instance, "name"):
            raise ValueError("`name` not defined in Dataset!")
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
        if not isinstance(df.index, pd.PeriodIndex):
            return False
        col_names = Counter(df.columns)
        if len(col_names) != len(cls.SCHEMA):
            return False
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

    @staticmethod
    def _calculate_split_index(
        dataset: "Dataset",
        timestamp: pd.Timestamp | None = None,
        length: pd.Timedelta | pd.DateOffset | None = None,
        percent: float | None = None,
    ) -> pd.Timestamp | int:
        """Validate splitting parameters and calulate the index to split on.

        Args:
            dataset (Dataset): Dataset used to validate the parameters
            timestamp (pd.Timestamp | None, optional): Timestamp to split on.
            length (pd.Timedelta | None, optional): Timedelta from
                beginning to split on.
            percent (float | None, optional): Percentage of the dataset
                between [0.0, 1.0] to split.

        Returns:
            pd.Timestamp | int: Timestamp or the index to split on

        Raises:
            ValueError: Must specify one and only one of timestamp, length, or percent
        """
        first_date = dataset.df.index[0].start_time  # type: ignore
        last_date = dataset.df.index[-1].start_time  # type: ignore
        if timestamp is None and length is None and percent is None:
            raise ValueError("Must specify how to split dataset")
        elif timestamp is not None and length is None and percent is None:
            if first_date > timestamp or timestamp > last_date:
                raise ValueError(
                    f"Timestamp given({timestamp}) must be within dataset \
                        bounds ({first_date}:{last_date})"
                )
            index = timestamp
        elif timestamp is None and length is not None and percent is None:
            split_timestamp = first_date + length

            if first_date > split_timestamp or split_timestamp > last_date:
                raise ValueError(
                    f"Start time + Timeperiod({split_timestamp}) must be \
                        within dataset bounds ({first_date}:{last_date})"
                )
            index = split_timestamp
        elif timestamp is None and length is None and percent is not None:
            if 0.0 >= percent or percent >= 1:
                raise ValueError(
                    f"Percentage({percent}) must be greater than 0 and less than 1"
                )
            index = int(dataset.df.shape[0] * percent)
        else:
            raise ValueError("Cannot specify multiple methods of splitting a dataset")

        return index

    def split(
        self,
        timestamp: pd.Timestamp | None = None,
        length: pd.Timedelta | None = None,
        percent: float | None = None,
        pre_name: str = "_pre",
        post_name: str = "_post",
    ) -> tuple["Dataset", "Dataset"]:
        """Split Dataset into Pre and Post split Datasets.

        Args:
            timestamp: [Optional] Timestamp to split Dataset on
            length: [Optional] pd.Timedelta to specify length of the pre-split dataset
            percent: [Optional] float percentage of the data to split on
            pre_name: Suffix to append to dataset name for pre section of the split
            post_name: Suffix to append to dataset name for post section of the split

        Returns:
            tuple: Pre and post split datasets of the same type as the original dataset
        """
        index = Dataset._calculate_split_index(self, timestamp, length, percent)
        pre_df = self.df[:index]
        post_df = self.df[index:].tail(-1)

        pre_dataset = self._dataset_from_split(
            self.name + pre_name, pre_df, self  # type: ignore
        )
        post_dataset = self._dataset_from_split(
            self.name + post_name, post_df, self  # type: ignore
        )
        return pre_dataset, post_dataset

    @staticmethod
    @abstractmethod
    def _dataset_from_split(
        name: str, df: pd.DataFrame, dataset: "Dataset"
    ) -> "Dataset":
        """Create Dataset from Dataframe and a corresponding Dataset."""


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

    @staticmethod
    def _dataset_from_split(
        name: str, df: pd.DataFrame, dataset: "PriceDataset"
    ) -> "PriceDataset":
        """Create Dataset from Dataframe and a corresponding PriceDataset.

        Args:
            name: Name to be assigned to the new dataset
            df: pd.DataFrame To create a new dataset around
            dataset: Dataset to use for metadata

        Returns:
            Dataset: Datasets of the same type as the original dataset with the new df
        """
        return CustomDataset(name, df, dataset.metadata.freq, dataset.metadata.pair)


class CustomDataset(PriceDataset):
    """Load a DataFrame in the expected format of price data."""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        freq: str,
        pair: AssetPair,
        synthetic: bool = False,
    ) -> None:
        """Initialize a PriceDataLoader.

        Validate DataFrame format correctness and infer timestep interval.

        Args:
            name: Name to be assigned to the new dataset
            df: DataFrame to load
            freq: Pandas period string representing price data interval
            pair: AssetPair represented in datal
            synthetic: Boolean indicating if this data is synthetic
        """
        self.name = name
        self.df = self._add_period_index(df, freq)
        self._metadata = MetaData(pair, freq)
        self.synthetic = synthetic

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
                data=pd.to_datetime(series, unit="s", utc=True),
                freq=freq,  # type: ignore
            )
        else:
            raise ValueError(
                f"""
                Was not able to convert series of type {series.dtype} to a
                pandas PeriodIndex.
                """
            )
        df = df.drop(columns=[cls.OPEN_TIME_NAME])
        return df.set_index(index, verify_integrity=True)

    @property
    def metadata(self) -> MetaData:
        """Price data metadata."""
        return self._metadata
