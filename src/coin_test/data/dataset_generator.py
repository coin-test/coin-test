"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .datasets import CustomDataset


class DatasetGenerator(ABC):
    """Create synthetic datasets."""

    @abstractmethod
    def generate(self, seed: int | None = None, n: int = 1) -> list[CustomDataset]:
        """Create synthetic datasets from the given dataset.

        Args:
            seed: A random seed for the generated datasets (optional)
            n: The number of datasets to generate

        Returns:
            list[PriceDataset]: The synthetic datasets
        """


class ResultsDatasetGenerator(DatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    def __init__(self, dataset: CustomDataset) -> None:
        """Initialize a ResultsDatasetGenerator object."""
        self.dataset = dataset

    def generate(self, seed: int | None = None, n: int = 1) -> list[CustomDataset]:
        """Create synthetic datasets from the given dataset.

        Args:
            seed: A random seed for the generated datasets (optional)
            n: The number of datasets to generate

        Returns:
            list[PriceDataset]: The synthetic datasets
        """
        rng = np.random.default_rng(seed)

        df = self.dataset.df
        starting_prices = df.sample(n=n, random_state=rng, replace=True)["Open"]
        df_norm = self.normalize_row_data(df)

        new_datasets = []
        for i in range(n):
            new_datasets.append(
                CustomDataset(
                    self.select_data(
                        df_norm, starting_prices[i], rng.df.index  # type: ignore
                    ),
                    self.dataset._metadata.freq,
                    self.dataset._metadata.pair,
                )
            )

        return new_datasets

    @staticmethod
    def select_data(
        df_norm: pd.DataFrame,
        starting_price: float,
        rng: np.random.Generator,
        index: pd.PeriodIndex,
    ) -> pd.DataFrame:
        """Take a normalized Dataframe and create a synthetic dataset from it.

        Args:
            df_norm: Normalized Dataframe of original data
            starting_price: The first open price for the data
            rng: A numpy random number generator
            index: The datetime index of the data

        Returns:
            pd.DataFrame: The synthetic dataset
        """
        # Select data
        sampled_data = df_norm.sample(
            frac=1, replace=True, random_state=rng
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

        # Add time index
        sampled_data.set_index(index, inplace=True)

        return sampled_data

    @staticmethod
    def normalize_row_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the row data so that it can be sampled with returns."""
        df = df.copy()

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
