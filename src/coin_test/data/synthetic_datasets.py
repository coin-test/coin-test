"""Define the SyntheticDatasets class and subclasses."""

from abc import ABC, abstractmethod

from pandas import DataFrame

from .datasets import PriceDataset


class DatasetGenerator(ABC):
    """Create synthetic datasets."""

    @abstractmethod
    def generate(
        seed: int | None = None,
        n: int = 1,
        *args,
    ) -> list[PriceDataset]:
        """Create synthetic datasets from the given dataset."""


class ResultsDatasetGenerator(DatasetGenerator):
    """Create synthetic datasets by shuffling the percentage gains each day."""

    def __init__(self, dataset: PriceDataset):
        self.dataset = dataset

    @staticmethod
    def generate(
        dataset: PriceDataset, seed: int | None = None, *args
    ) -> list[PriceDataset]:
        """Create synthetic datasets from the given dataset."""
        if seed:
            pass

        df = dataset.df
        starting_price = df.sample(n=1, random_state=seed)["Open"]
        df_norm = ResultsDatasetGenerator.normalize_row_data(df)

        return [dataset]

    @staticmethod
    def select_data(df_norm: DataFrame, starting_price: float) -> DataFrame:
        """"""
        shuffled_gains = df_norm.sample(frac=1, replace=True)
        oc_data = shuffled_gains.melt(
            id_vars=["High", "Low"], value_vars=["Open", "Close"]
        )
        return df_norm

    @staticmethod
    def normalize_row_data(df: DataFrame) -> DataFrame:
        """Normalize the row data so that it can be sampled with returns."""
        df = df.copy()
        # Normalize high, low, and close from open prices
        df[["High", "Low", "Close_norm"]] = df[["High", "Low", "Close"]] / df["Open"]

        # Normalize open prices from previous close prices
        df.loc[1:, "Open"] /= df.loc[:-1, "Close"]
        df.loc[0, "Open"] = 1

        # Restructure df
        df.rename(columns={"Close_norm": "Close"})

        return df
