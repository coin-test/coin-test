"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from random import Random
from typing import cast, List, Literal

from arch import arch_model
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .datasets import CustomDataset


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
            list[PriceDataset]: The synthetic datasets
        """


class StitchedChunkDatasetGenerator(DatasetGenerator):
    """Synthetic Dataset Generator with chunks of data."""

    DATASET_TYPE = CustomDataset

    def __init__(
        self,
        dataset: "StitchedChunkDatasetGenerator.DATASET_TYPE",
        chunk_size: int = 10,
    ) -> None:
        """Initialize a chunk synthetic dataset generator."""
        self.dataset = dataset
        self.start: pd.Period = dataset.df.index[0]  # type: ignore
        self.metadata = dataset.metadata
        self.chunk_size = chunk_size

        if chunk_size < 1:
            raise ValueError("Chunk size must be a positive integer")
        elif len(self.dataset.df) < chunk_size:
            raise ValueError("Chunk size mustn't be larger than the dataset")

    @staticmethod
    def create_index(
        start: pd.Period, timedelta: pd.Timedelta | pd.DateOffset, freq: str
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

    def generate(
        self,
        timedelta: pd.Timedelta | pd.DateOffset,
        seed: int | None = None,
        n: int = 1,
    ) -> list["StitchedChunkDatasetGenerator.DATASET_TYPE"]:
        """Create chunk-based synthetic datasets from the given dataset.

        Args:
            timedelta: A time range for the new datasets
            seed: A random seed for the generated datasets
            n: The number of datasets to generate

        Returns:
            list[PriceDataset]: The synthetic datasets
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
        unnormalized_df = ReturnsDatasetGenerator.unnormalize(sampled_data)

        return unnormalized_df


class ReturnsDatasetGenerator(StitchedChunkDatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    DATASET_TYPE = CustomDataset

    def __init__(self, dataset: "ReturnsDatasetGenerator.DATASET_TYPE") -> None:
        """Initialize a ResultsDatasetGenerator object."""
        super().__init__(dataset, chunk_size=1)


@dataclass
class GarchSettings:
    """Class for keeping track of settings to intialize a GARCH model.

    Defaults initialize a GARCH(1,1) model with constant mean.
    """

    mean: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ] = "Constant"
    lags: int | NDArray | List[int] | None = 0
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH"
    p: int | List[int] = 1
    o: int = 0
    q: int = 1
    power: float = 2
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ] = "normal"
    hold_back: int | None = None
    rescale: bool | None = None


class GARCHDatasetGenerator(DatasetGenerator):
    """Synthetic Dataset Generator with GARCH.

    Using Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.

    Close prices are simulated with univariate GARCH model.
    Open prices are set as previous day's close.
    High and Low prices are randomly sampled chunks normalized as
    percent changes relative to min/max Open/Close of given historical dataset.
    """

    DATASET_TYPE = CustomDataset

    def __init__(
        self,
        dataset: "GARCHDatasetGenerator.DATASET_TYPE",
        chunk_size: int = 1,
        mean: Literal[
            "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
        ] = "Constant",
        lags: int | List[int] | NDArray | None = 0,
        vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH",
        p: int | List[int] = 1,
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
        """Initialize a GARCHDatasetGenerator."""
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
    def get_GARCH_model_parameters(
        univariate_series: pd.Series, garch_settings: GarchSettings
    ) -> pd.Series:
        """Get GARCH model parameters estimated from given univariate series.

        Args:
            univariate_series: series of data
            garch_settings: settings to contruct the GARCH model with

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
        """Samples a new series from a pandas series.

        Args:
            series: series to sample from
            num_rows: The number of rows in the series
            chunk_size: The amount of rows to combine in each chunk
            rng: A numpy random number generator

        Returns:
            pd.Series: a new sampled series
        """
        # Select Chunk Length Windows
        chunks = tuple(series.rolling(chunk_size))[chunk_size - 1 :]

        rows_per_chunk = len(chunks[0])
        num_chunks = ceil(num_rows / rows_per_chunk)

        # Select data and concatenate chunks
        r = Random(rng.bytes(16))  # type: ignore
        sampled_chunks: list[pd.Series] = r.sample(chunks, num_chunks)
        sampled_data: pd.Series = pd.concat(sampled_chunks)

        # shrink to size of dataset
        sampled_data = sampled_data.head(num_rows).reset_index(drop=True)

        return sampled_data

    def generate(
        self,
        timedelta: pd.Timedelta | pd.DateOffset,
        seed: int | None = None,
        n: int = 1,
    ) -> list["GARCHDatasetGenerator.DATASET_TYPE"]:
        """Create chunk-based synthetic datasets from the given dataset.

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

        # Define univariate price trend to simulate with GARCH model.
        returns = 100 * df["Close"].pct_change().dropna()

        # Normalize high prices
        df["max_open_close"] = df[["Open", "Close"]].max(axis=1)
        df["pct_diff_high"] = (df["High"] - df["max_open_close"]) / df["max_open_close"]

        # Normalize low prices
        df["min_open_close"] = df[["Open", "Close"]].min(axis=1)
        df["pct_diff_low"] = (df["Low"] - df["min_open_close"]) / df["min_open_close"]

        garch_model_params = self.get_GARCH_model_parameters(
            returns, self.garch_settings
        )

        # Construct an empty GARCH model for simulating data.
        sim_garch_model = arch_model(
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
        )

        new_datasets = []
        for i in range(n):
            synthetic_series_close_pct_change = sim_garch_model.simulate(
                garch_model_params, num_rows  # type: ignore
            )["data"]

            # Transform generated Close percent change values to Close prices.
            # Intialize Close_0 from starting value.
            # Then for each row,
            # set Close_j = Close_j-1 + Close_j-1 * pct_change_j-1 / 100.
            synthetic_series_close = pd.Series(0, index=list(range(num_rows)))
            synthetic_series_close[0] = starting_prices["Close"][i]
            for j in range(1, num_rows):
                synthetic_series_close[j] = (
                    synthetic_series_close[j - 1]
                    + synthetic_series_close[j - 1]
                    * synthetic_series_close_pct_change[j - 1]
                    / 100
                )

            # Set Open price as previous Close price.
            synthetic_series_open = synthetic_series_close.shift(periods=1)
            synthetic_series_open[0] = starting_prices["Open"][i]

            # Set High price as percent change on max between Open and Close.
            synthetic_series_high_pct_change = self.sample_series(
                df["pct_diff_high"],
                num_rows,
                chunk_size=self.chunk_size,
                rng=rng,
            )

            synthetic_series_max_open_close = synthetic_series_close.combine(
                synthetic_series_open, max, fill_value=0
            )

            synthetic_series_high = (
                synthetic_series_max_open_close
                + synthetic_series_max_open_close * synthetic_series_high_pct_change
            )

            # Set Low price as percent change on min between Open and Close.
            synthetic_series_low_pct_change = self.sample_series(
                df["pct_diff_close"],
                num_rows,
                chunk_size=self.chunk_size,
                rng=rng,
            )

            synthetic_series_min_open_close = synthetic_series_close.combine(
                synthetic_series_open, min, fill_value=0
            )

            synthetic_series_low = (
                synthetic_series_min_open_close
                + synthetic_series_min_open_close * synthetic_series_low_pct_change
            )

            # Build dataframe.
            synthetic_index_labels = period_index.copy()
            synthetic_series_open.index = synthetic_index_labels
            synthetic_series_high.index = synthetic_index_labels
            synthetic_series_low.index = synthetic_index_labels
            synthetic_series_close.index = synthetic_index_labels
            synthetic_df = pd.concat(
                {
                    "Open": synthetic_series_open,
                    "High": synthetic_series_high,
                    "Low": synthetic_series_low,
                    "Close": synthetic_series_close,
                },
                axis=1,
            )

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
