"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod

import pandas as pd

try:
    from icecream import ic
except ImportError:
    ic = lambda x: x

from .datasets import CustomDataset, PriceDataset


class DatasetGenerator(ABC):
    """Create synthetic datasets."""

    @abstractmethod
    def generate(
        self,
        seed: int | None = None,
        n: int = 1,
        **kwargs,
    ) -> list[PriceDataset]:
        """Create synthetic datasets from the given dataset."""


class ResultsDatasetGenerator(DatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    def __init__(self, dataset: CustomDataset):
        self.dataset = dataset

    def generate(self, seed: int | None = None, n: int = 1, **kwargs) -> list[PriceDataset]:
        """Create synthetic datasets from the given dataset."""
        if seed:
            pass

        df = self.dataset.df
        starting_prices = df.sample(n=n, random_state=seed, replace=True)["Open"]
        df_norm = ResultsDatasetGenerator.normalize_row_data(df)

        new_datasets = []
        for i in range(n):
            new_datasets.append(
                CustomDataset(
                    ResultsDatasetGenerator.select_data(df_norm, starting_prices[i]), # type: ignore
                    freq = self.dataset.freq,
                    pair = self.dataset.pair
                )
            )

        return new_datasets

    @staticmethod
    def select_data(df_norm: pd.DataFrame, starting_price: float) -> pd.DataFrame:
        """"""
        # Select data
        sampled_data = df_norm.sample(frac=1, replace=True).reset_index(drop=True)
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

        # Calculate Unnormalized Open and Close data
        oc_stacked["Price"] = oc_stacked["Perc_Change"].cumprod()

        ic(oc_stacked)

        # Unstack Open and Close data
        oc_unstacked = oc_stacked.pivot(index="index", columns="O/C", values="Price")

        ic(oc_unstacked)

        # Put unstacked data in sample data
        sampled_data["Open"] = oc_unstacked["Open"]
        sampled_data["Close"] = oc_unstacked["Close"]
        sampled_data["High"] = oc_unstacked["Open"] * sampled_data["High"]
        sampled_data["Low"] = oc_unstacked["Open"] * sampled_data["Low"]

        ic(sampled_data)

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
