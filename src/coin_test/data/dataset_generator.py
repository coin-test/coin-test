"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod
from math import ceil
from random import Random

import numpy as np
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
    def unnormalize(normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Take a normalized Dataframe and unnormalize it.

        Essentially, convert columns from representing percentage increases
        to actual prices.

        Args:
            normalized_df: Normalized Dataframe

        Returns:
            pd.DataFrame: The unnormalized Dataframe
        """
        # Stack Open and Close data in one column
        oc_stacked = normalized_df.melt(
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
        normalized_df["Open"] = oc_unstacked["Open"]
        normalized_df["Close"] = oc_unstacked["Close"]
        normalized_df["High"] = oc_unstacked["Open"] * normalized_df["High"]
        normalized_df["Low"] = oc_unstacked["Open"] * normalized_df["Low"]

        return normalized_df

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
        print("num_rows in period_index", num_rows)

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
        print("\n\n".join(repr(x) for x in chunks))
        rows_per_chunk = len(chunks[0])
        num_chunks = ceil(num_rows / rows_per_chunk)

        # Select data and concatenate chunks
        r = Random(rng.bytes(16))  # type: ignore
        sampled_chunks: list[pd.DataFrame] = r.sample(chunks, num_chunks)
        sampled_data: pd.DataFrame = pd.concat(sampled_chunks)

        # shrink to size of dataset
        sampled_data = sampled_data.head(num_rows).reset_index(drop=True)

        print(sampled_data)

        sampled_data.loc[0, "Open"] = starting_price
        unnormalized_df = ReturnsDatasetGenerator.unnormalize(sampled_data)
        print(unnormalized_df)
        return unnormalized_df


class ReturnsDatasetGenerator(StitchedChunkDatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    DATASET_TYPE = CustomDataset

    def __init__(self, dataset: "ReturnsDatasetGenerator.DATASET_TYPE") -> None:
        """Initialize a ResultsDatasetGenerator object."""
        super().__init__(dataset, chunk_size=1)

    # @staticmethod
    # def select_data(
    #     df_norm: pd.DataFrame,
    #     starting_price: float,
    #     num_rows: int,
    #     rng: np.random.Generator,
    # ) -> pd.DataFrame:
    #     """Take a normalized Dataframe and create a synthetic dataset from it.

    #     Args:
    #         df_norm: Normalized Dataframe of original data
    #         starting_price: The first open price for the data
    #         num_rows: The number of rows in the dataset
    #         rng: A numpy random number generator

    #     Returns:
    #         pd.DataFrame: The synthetic dataset
    #     """
    #     # Select data
    #     sampled_data = df_norm.sample(
    #         n=num_rows, replace=True, random_state=rng
    #     ).reset_index(drop=True)
    #     sampled_data.loc[0, "Open"] = starting_price

    #     unnormalized_df = ReturnsDatasetGenerator.unnormalize(sampled_data)
    #     return unnormalized_df
