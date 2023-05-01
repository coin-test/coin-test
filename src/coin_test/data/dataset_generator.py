"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from math import ceil
from random import Random
from typing import cast, Literal

from arch import arch_model
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .datasets import CustomDataset

logger = logging.getLogger(__name__)


class DatasetGenerator(ABC):
    """Create synthetic datasets."""

    @abstractmethod
    def generate(
        self, timedelta: pd.Timedelta, seed: int | None = None, n: int = 1
    ) -> list[CustomDataset]:
        """Create synthetic datasets from the given dataset.

        Args:
            timedelta: A time range for the new datasets
            seed: A random seed for the generated datasets
            n: The number of datasets to generate

        Returns:
            list[DATASET_TYPE]: The synthetic datasets
        """


class WindowStepDatasetGenerator(DatasetGenerator):
    """Windows of data as separate datasets."""

    DATASET_TYPE = CustomDataset

    def __init__(self, dataset: "WindowStepDatasetGenerator.DATASET_TYPE") -> None:
        """Create a window step dataset generator."""
        self.dataset = dataset
        self.metadata = dataset.metadata

    def generate(
        self,
        timedelta: pd.Timedelta,
        seed: int | None = None,
        n: int = 1,
    ) -> list["WindowStepDatasetGenerator.DATASET_TYPE"]:
        """Create uniformly distributed window+step datasets.

        Given a length of time for each dataset, along with a number of
        datasets to create, make new datasets of a given length that are
        equally spaced from each other, creating overlapping datasets
        if necessary.

        Args:
            timedelta: A time range for the new datasets
            seed: Irrelevant for this implementation
            n: The number of datasets to generate

        Returns:
            list[DATASET_TYPE]: The synthetic datasets
        """
        del seed

        chunks = self.extract_windows(
            self.dataset.df,
            self.metadata.freq,
            timedelta,
            n,
        )

        datasets = []

        for i, chunk in enumerate(chunks):
            datasets.append(
                self.DATASET_TYPE(
                    f"{type(self).__name__}_{i}",
                    chunk,
                    self.metadata.freq,
                    self.metadata.pair,
                    synthetic=False,
                )
            )

        return datasets

    @staticmethod
    def extract_windows(
        df_total: pd.DataFrame,
        freq: str,
        timedelta: pd.Timedelta,
        n: int,
    ) -> list[pd.DataFrame]:
        """Take a DataFrame and create windows from it.

        Args:
            df_total: Entire original DataFrame
            freq: The frequency of the DataFrame PeriodIndex
            timedelta: The length in time per window
            n: The number of windows

        Returns:
            list[pd.DataFrame]: The windows
        """
        window_length = WindowStepDatasetGenerator.calc_window_length(freq, timedelta)

        slices = WindowStepDatasetGenerator.make_slices(len(df_total), window_length, n)

        return [cast(pd.DataFrame, df_total.iloc[s]) for s in slices]

    @staticmethod
    def make_slices(total_length: int, window_length: int, n: int) -> list[slice]:
        """Given the total dataset length and window length, make slices for the df."""
        last_sliceable = total_length - window_length
        if last_sliceable < 0:
            raise ValueError("Windows are larger than original dataset")

        # calculate the num of intervals between each chunk's start
        window_sep = last_sliceable / (n - 1)
        window_overlap_perc = (window_length - window_sep) / window_length

        # check valid chunk sizes
        if window_sep < 1:
            raise ValueError(f"Impossible to make {n} windows from the given dataset.")
        if window_overlap_perc > 0.5:
            logger.warning(f"Windows overlap by {window_overlap_perc*100:.0f}%")

        start_indices = (
            np.linspace(start=0, stop=last_sliceable, num=n).round().astype(int)
        )

        return list(map(slice, start_indices, start_indices + window_length))

    @staticmethod
    def calc_window_length(freq: str, timedelta: pd.Timedelta) -> int:
        """Calculate the number of rows in each window of length timedelta."""
        temp_start = pd.Period("1970-1-1", freq=freq)
        temp_idx = pd.period_range(
            start=temp_start,
            end=temp_start + timedelta,
            freq=freq,
        )
        return len(temp_idx)


class SamplingDatasetGenerator(DatasetGenerator):
    """ABC for sampling dataset generators."""

    @staticmethod
    def create_index(
        start: pd.Period, timedelta: pd.Timedelta, freq: str
    ) -> pd.PeriodIndex:
        """Create a PeriodIndex given a start time, timedelta, and frequency."""
        return pd.period_range(start=start, end=start + timedelta, freq=freq)

    @staticmethod
    def normalize_row_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the row data so that it can be sampled with returns."""
        df = df.reset_index(drop=True)

        # Normalize high, low, and close from open prices
        df["High"] = df["High"] / df["Open"]
        df["Low"] = df["Low"] / df["Open"]
        df["Close_Norm"] = df["Close"] / df["Open"]

        # Normalize open prices from previous close prices
        df["Open"] /= df["Close"].shift(1, fill_value=df.iloc[0]["Open"])

        # Restructure df
        del df["Close"]
        df.rename(columns={"Close_Norm": "Close"}, inplace=True)

        return df

    @staticmethod
    def unnormalize(synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """Take a normalized Dataframe and unnormalize it.

        Essentially, convert columns from representing percentage increases
        to actual prices.

        Args:
            synthetic_df: Normalized Dataframe

        Returns:
            pd.DataFrame: The unnormalized Dataframe
        """
        # Stack Open and Close data in one column
        oc_stacked = synthetic_df.melt(
            id_vars=["High", "Low"],
            value_vars=["Open", "Close"],
            var_name="O/C",
            value_name="Perc_Change",
            ignore_index=False,
        ).reset_index()
        oc_stacked.sort_values(
            by=["index", "O/C"], ascending=[True, False], inplace=True
        )

        # Calculate Open and Close data
        oc_stacked["Price"] = oc_stacked["Perc_Change"].cumprod()

        # Unstack Open and Close data
        oc_unstacked = oc_stacked.pivot(index="index", columns="O/C", values="Price")

        # Put unstacked data in sample data
        synthetic_df["Open"] = oc_unstacked["Open"]
        synthetic_df["Close"] = oc_unstacked["Close"]
        synthetic_df["High"] = oc_unstacked["Open"] * synthetic_df["High"]
        synthetic_df["Low"] = oc_unstacked["Open"] * synthetic_df["Low"]

        return synthetic_df


class StitchedChunkDatasetGenerator(SamplingDatasetGenerator):
    """Synthetic Dataset Generator with chunks of data."""

    DATASET_TYPE = CustomDataset

    def __init__(
        self,
        dataset: "StitchedChunkDatasetGenerator.DATASET_TYPE",
        chunk_size: int = 10,
    ) -> None:
        """Initialize a chunk synthetic dataset generator."""
        self.dataset = dataset
        self.start = cast(pd.Period, dataset.df.index[0])
        self.metadata = dataset.metadata
        self.chunk_size = chunk_size

        if chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        elif len(self.dataset.df) < chunk_size:
            raise ValueError("Chunk size mustn't be larger than the dataset")

    def generate(
        self,
        timedelta: pd.Timedelta,
        seed: int | None = None,
        n: int = 1,
    ) -> list["StitchedChunkDatasetGenerator.DATASET_TYPE"]:
        """Create chunk-based synthetic datasets from the given dataset.

        Args:
            timedelta: A time range for the new datasets
            seed: A random seed for the generated datasets
            n: The number of datasets to generate

        Returns:
            list[DATASET_TYPE]: The synthetic datasets
        """
        rng = np.random.default_rng(seed)

        df = self.dataset.df
        starting_prices = df.sample(n=n, random_state=rng, replace=True)["Open"]
        df_norm = self.normalize_row_data(df)

        period_index = self.create_index(self.start, timedelta, self.metadata.freq)
        num_rows = len(period_index)

        new_datasets = []
        for i in range(n):
            synthetic_df = self.select_data(
                df_norm,
                starting_prices[i],  # type: ignore
                num_rows,
                self.chunk_size,
                rng,
            )
            synthetic_df.index = period_index.copy()
            new_datasets.append(
                self.DATASET_TYPE(
                    f"{type(self).__name__}_{i}",
                    synthetic_df,
                    self.metadata.freq,
                    self.metadata.pair,
                    synthetic=True,
                )
            )

        return new_datasets

    @staticmethod
    def select_data(
        df_norm: pd.DataFrame,
        starting_price: float,
        num_rows: int,
        chunk_size: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Take a normalized Dataframe and create a synthetic dataset from it.

        Args:
            df_norm: Normalized Dataframe of original data
            starting_price: The first open price for the data
            num_rows: The number of rows in the dataset
            chunk_size: The amount of rows to combine in each chunk
            rng: A numpy random number generator

        Returns:
            pd.DataFrame: The synthetic dataset
        """
        # Select Chunk Length Windows
        chunks = tuple(df_norm.rolling(chunk_size))[chunk_size - 1 :]

        rows_per_chunk = len(chunks[0])
        num_chunks = ceil(num_rows / rows_per_chunk)

        # Select data and concatenate chunks
        r = Random(rng.bytes(16))  # type: ignore
        sampled_chunks: list[pd.DataFrame] = r.sample(chunks, num_chunks)
        sampled_data: pd.DataFrame = pd.concat(sampled_chunks)

        # shrink to size of dataset
        sampled_data = sampled_data.head(num_rows).reset_index(drop=True)

        sampled_data.loc[0, "Open"] = starting_price
        unnormalized_df = StitchedChunkDatasetGenerator.unnormalize(sampled_data)

        return unnormalized_df


class ReturnsDatasetGenerator(SamplingDatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    DATASET_TYPE = CustomDataset

    def __init__(self, dataset: "ReturnsDatasetGenerator.DATASET_TYPE") -> None:
        """Initialize a ResultsDatasetGenerator object."""
        self.dataset = dataset
        self.start = cast(pd.Period, dataset.df.index[0])
        self.metadata = dataset.metadata

    def generate(
        self,
        timedelta: pd.Timedelta,
        seed: int | None = None,
        n: int = 1,
    ) -> list["ReturnsDatasetGenerator.DATASET_TYPE"]:
        """Create returns-based synthetic datasets from the given dataset.

        Args:
            timedelta: A time range for the new datasets
            seed: A random seed for the generated datasets
            n: The number of datasets to generate

        Returns:
            list[DATASET_TYPE]: The synthetic datasets
        """
        rng = np.random.default_rng(seed)

        df = self.dataset.df
        starting_prices = df.sample(n=n, random_state=rng, replace=True)["Open"]
        df_norm = self.normalize_row_data(df)

        period_index = self.create_index(self.start, timedelta, self.metadata.freq)

        new_datasets = []
        for i in range(n):
            synthetic_df = self.select_data(
                df_norm, starting_prices[i], len(period_index), rng  # type: ignore
            )
            synthetic_df.index = period_index.copy()
            new_datasets.append(
                self.DATASET_TYPE(
                    f"{type(self).__name__}_{i}",
                    synthetic_df,
                    self.metadata.freq,
                    self.metadata.pair,
                    synthetic=True,
                )
            )

        return new_datasets

    @staticmethod
    def select_data(
        df_norm: pd.DataFrame,
        starting_price: float,
        num_rows: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Take a normalized Dataframe and create a synthetic dataset from it.

        Args:
            df_norm: Normalized Dataframe of original data
            starting_price: The first open price for the data
            num_rows: The number of rows in the dataset
            rng: A numpy random number generator

        Returns:
            pd.DataFrame: The synthetic dataset
        """
        # Select data
        sampled_data = df_norm.sample(
            n=num_rows, replace=True, random_state=rng
        ).reset_index(drop=True)
        sampled_data.loc[0, "Open"] = starting_price

        # Stack Open and Close data in one column
        oc_stacked = sampled_data.melt(
            id_vars=["High", "Low"],
            value_vars=["Open", "Close"],
            var_name="O/C",
            value_name="Perc_Change",
            ignore_index=False,
        ).reset_index()
        oc_stacked.sort_values(
            by=["index", "O/C"], ascending=[True, False], inplace=True
        )

        # Calculate Open and Close data
        oc_stacked["Price"] = oc_stacked["Perc_Change"].cumprod()

        # Unstack Open and Close data
        oc_unstacked = oc_stacked.pivot(index="index", columns="O/C", values="Price")

        # Put unstacked data in sample data
        sampled_data["Open"] = oc_unstacked["Open"]
        sampled_data["Close"] = oc_unstacked["Close"]
        sampled_data["High"] = oc_unstacked["Open"] * sampled_data["High"]
        sampled_data["Low"] = oc_unstacked["Open"] * sampled_data["Low"]

        return sampled_data


@dataclass
class GarchSettings:
    """Class for keeping track of settings to intialize a GARCH model."""

    mean: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ]
    lags: int | NDArray | list[int] | None
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"]
    p: int | list[int]
    o: int
    q: int
    power: float
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ]
    hold_back: int | None
    rescale: bool | None


class GarchDatasetGenerator(DatasetGenerator):
    """Synthetic Dataset Generator with GARCH.

    Use Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.
    Default values initialize a GARCH(1,1) model with constant mean.

    Close prices are simulated with univariate GARCH model.
    Open prices are set as previous day's close.
    High and Low prices are randomly sampled chunks transformed into and reverted from
    percent changes relative to min/max Open/Close of each period bar.
    """

    DATASET_TYPE = CustomDataset

    def __init__(
        self,
        dataset: "GarchDatasetGenerator.DATASET_TYPE",
        chunk_size: int = 1,
        mean: Literal[
            "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
        ] = "Constant",
        lags: int | list[int] | NDArray | None = 0,
        vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH",
        p: int | list[int] = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2,
        dist: Literal[
            "normal",
            "gaussian",
            "t",
            "studentst",
            "skewstudent",
            "skewt",
            "ged",
            "generalized error",
        ] = "normal",
        hold_back: int | None = None,
        rescale: bool | None = None,
    ) -> None:
        """Initializes a GARCHDatasetGenerator."""
        self.dataset = dataset
        self.start: pd.Period = dataset.df.index[0]  # type: ignore
        self.metadata = dataset.metadata
        self.chunk_size = chunk_size

        if chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        elif len(self.dataset.df) < chunk_size:
            raise ValueError("Chunk size mustn't be larger than the dataset")

        self.garch_settings = GarchSettings(
            mean=mean,
            lags=lags,
            vol=vol,
            p=p,
            o=o,
            q=q,
            power=power,
            dist=dist,
            hold_back=hold_back,
            rescale=rescale,
        )

    @staticmethod
    def get_garch_model_parameters(
        univariate_series: pd.Series,
        garch_settings: GarchSettings,
    ) -> pd.Series:
        """Gets GARCH model parameters estimated from given univariate series.

        Args:
            univariate_series: series of data
            garch_settings: arguments to contruct the GARCH model with

        Returns:
            res_garch_model.params: dictionary of model parameters
                estimated from fitting to univariate series.
        """
        res_garch_model = arch_model(
            univariate_series,
            mean=garch_settings.mean,
            lags=garch_settings.lags,
            vol=garch_settings.vol,
            p=garch_settings.p,
            o=garch_settings.o,
            q=garch_settings.q,
            power=garch_settings.power,
            dist=garch_settings.dist,
            hold_back=garch_settings.hold_back,
            rescale=garch_settings.rescale,
        ).fit(disp="off")

        return cast(pd.Series, res_garch_model.params)

    @staticmethod
    def sample_series(
        series: pd.Series,
        num_rows: int,
        chunk_size: int,
        rng: np.random.Generator,
    ) -> pd.Series:
        """Randomly samples a new series from a given series.

        Args:
            series: series to sample from
            num_rows: The number of rows in the new series
            chunk_size: The amount of rows to combine in each chunk
            rng: A numpy random number generator

        Returns:
            pd.Series: a new sampled series
        """
        # Select chunk length windows.
        chunks = tuple(series.rolling(chunk_size))[chunk_size - 1 :]

        rows_per_chunk = len(chunks[0])
        num_chunks = ceil(num_rows / rows_per_chunk)

        # Select data and concatenate chunks.
        r = Random(rng.bytes(16))  # type: ignore
        sampled_chunks: list[pd.Series] = r.sample(chunks, num_chunks)
        sampled_series: pd.Series = pd.concat(sampled_chunks)

        # Shrink to size of dataset
        sampled_series = sampled_series.head(num_rows).reset_index(drop=True)

        # Remove series name.
        sampled_series = sampled_series.rename(None)

        return sampled_series

    @staticmethod
    def to_pct_change_based_on_extreme_bar_open_and_close(
        series_to_pct_change: pd.Series,
        extreme: Literal["max", "min"],
        open_series: pd.Series,
        close_series: pd.Series,
    ) -> pd.Series:
        """Calculates percent change relative to min/max of Open and Close.

        Args:
            series_to_pct_change: a series of values to turn into percent change
            extreme: selects either min or max when calculating extreme series
            open_series: series of open price values
            close_series: series of close price values

        Returns:
            pct_change_series: difference of series_to_pct_change and extreme_series
            as proportion of extreme_series.

        """
        if extreme == "max":
            extreme_series = close_series.combine(open_series, max, fill_value=0)
        else:
            extreme_series = close_series.combine(open_series, min, fill_value=0)

        # Remove all series index before calculation.
        series_to_pct_change = series_to_pct_change.reset_index(drop=True)
        extreme_series = extreme_series.reset_index(drop=True)

        pct_change_series = (series_to_pct_change - extreme_series) / extreme_series

        return pct_change_series

    @staticmethod
    def from_pct_change_based_on_extreme_bar_open_and_close(
        pct_change_series: pd.Series,
        extreme: Literal["max", "min"],
        open_series: pd.Series,
        close_series: pd.Series,
    ) -> pd.Series:
        """Calculates values from percent change and min/max of Open and Close.

        Args:
            pct_change_series: a series of percent change to turn into values
            extreme: selects either min or max when calculating extreme series
            open_series: series of open price values
            close_series: series of close price values

        Returns:
            series: extreme_series + extreme_series * pct_change_series

        Raises:
            ValueError: open_series should be same length as close_series.
            ValueError: pct_change_series should be same length as open_series.
        """
        if len(close_series) != len(open_series):
            raise ValueError("open_series should be same length as close_series.")

        if len(pct_change_series) != len(open_series):
            raise ValueError("pct_change_series should be same length as open_series.")

        if extreme == "max":
            extreme_series = close_series.combine(open_series, max, fill_value=0)
        else:
            extreme_series = close_series.combine(open_series, min, fill_value=0)

        # Remove all series index before calculation.
        pct_change_series = pct_change_series.reset_index(drop=True)
        extreme_series = extreme_series.reset_index(drop=True)

        return extreme_series + extreme_series * pct_change_series

    @staticmethod
    def from_pct_change_based_on_starting_value(
        pct_change_series: pd.Series,
        starting_price: float,
    ) -> pd.Series:
        """Generate synthetic series from GARCH model fit to univariate data."""
        # Values are one row longer than by-item percent changes.
        num_rows = len(pct_change_series) + 1
        # Remove all series index before calculation.
        pct_change_series = pct_change_series.reset_index(drop=True)

        # Transform generated percent change values to price values.
        synthetic_series = pd.Series(0, index=list(range(num_rows)))
        # Intialize series_0 from starting value.
        synthetic_series[0] = starting_price
        # set series_j = series_j-1 + series_j-1 * pct_change_j-1 / 100.
        for j in range(1, num_rows):
            synthetic_series[j] = (
                synthetic_series[j - 1]
                + synthetic_series[j - 1] * pct_change_series[j - 1] / 100
            )

        synthetic_series = synthetic_series.reset_index(drop=True).rename(None)
        return synthetic_series

    def generate(
        self,
        timedelta: pd.Timedelta,
        seed: int | None = None,
        n: int = 1,
    ) -> list["GarchDatasetGenerator.DATASET_TYPE"]:
        """Create synthetic datasets from GARCH model fit to given dataset.

        Args:
            timedelta: A time range for the new datasets
            seed: A random seed for the generated datasets
            n: The number of datasets to generate

        Returns:
            list[PriceDataset]: The synthetic datasets
        """
        rng = np.random.default_rng(seed)

        df = self.dataset.df
        starting_prices = df.sample(n=n, random_state=rng, replace=True)[
            ["Open", "Close"]
        ]

        period_index = StitchedChunkDatasetGenerator.create_index(
            self.start, timedelta, self.metadata.freq
        )
        num_rows = len(period_index)

        # Estimate GARCH model parameters from univariate percent change series.
        pct_change_close_series = 100 * df["Close"].pct_change().dropna()
        garch_model_params = GarchDatasetGenerator.get_garch_model_parameters(
            univariate_series=pct_change_close_series,
            garch_settings=self.garch_settings,
        )

        pct_change_high_series = self.to_pct_change_based_on_extreme_bar_open_and_close(
            series_to_pct_change=df["High"],
            extreme="max",
            open_series=df["Open"],
            close_series=df["Close"],
        )

        pct_change_low_series = self.to_pct_change_based_on_extreme_bar_open_and_close(
            series_to_pct_change=df["Low"],
            extreme="min",
            open_series=df["Open"],
            close_series=df["Close"],
        )

        new_datasets = []
        for i in range(n):
            # Initalize a GARCH model distribution and simulate Close data.
            synthetic_pct_change_close_series = arch_model(
                None,
                mean=self.garch_settings.mean,
                lags=self.garch_settings.lags,
                vol=self.garch_settings.vol,
                p=self.garch_settings.p,
                o=self.garch_settings.o,
                q=self.garch_settings.q,
                power=self.garch_settings.power,
                dist=self.garch_settings.dist,
                hold_back=self.garch_settings.hold_back,
                rescale=self.garch_settings.rescale,
            ).simulate(
                garch_model_params, num_rows - 1  # type: ignore
            )[
                "data"
            ]
            synthetic_close_series = (
                GarchDatasetGenerator.from_pct_change_based_on_starting_value(
                    pct_change_series=synthetic_pct_change_close_series,
                    starting_price=starting_prices["Close"][i],  # type: ignore
                )
            )

            synthetic_open_series = synthetic_close_series.shift(periods=1)
            synthetic_open_series[0] = starting_prices["Open"][i]

            pct_change_high_series_sampled = GarchDatasetGenerator.sample_series(
                series=pct_change_high_series,
                num_rows=num_rows,
                chunk_size=self.chunk_size,
                rng=rng,
            )
            synthetic_high_series = (
                self.from_pct_change_based_on_extreme_bar_open_and_close(
                    pct_change_series=pct_change_high_series_sampled,
                    extreme="max",
                    close_series=synthetic_close_series,
                    open_series=synthetic_open_series,
                )
            )

            pct_change_low_series_sampled = GarchDatasetGenerator.sample_series(
                series=pct_change_low_series,
                num_rows=num_rows,
                chunk_size=self.chunk_size,
                rng=rng,
            )
            synthetic_low_series = (
                self.from_pct_change_based_on_extreme_bar_open_and_close(
                    pct_change_series=pct_change_low_series_sampled,
                    extreme="min",
                    close_series=synthetic_close_series,
                    open_series=synthetic_open_series,
                )
            )

            synthetic_df = pd.concat(
                {
                    "Open": synthetic_open_series,
                    "High": synthetic_high_series,
                    "Low": synthetic_low_series,
                    "Close": synthetic_close_series,
                    "Volume": pd.Series(0, index=list(range(num_rows)), dtype=float),
                },
                axis=1,
            )
            synthetic_df.index = period_index.copy()

            new_datasets.append(
                self.DATASET_TYPE(
                    f"{type(self).__name__}_{i}",
                    synthetic_df,
                    self.metadata.freq,
                    self.metadata.pair,
                    synthetic=True,
                )
            )

        return new_datasets
