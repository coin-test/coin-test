"""Test the DatasetGenerator classes."""

import pandas as pd

from coin_test.data import ResultsDatasetGenerator
from coin_test.data.datasets import CustomDataset, PriceDataset
from coin_test.util import AssetPair, Ticker


def test_dataset_generator_initialized(hour_data_indexed_df: pd.DataFrame) -> None:
    """Test that the ResultDatasetGenerator can be initialized."""
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    price_dataset = CustomDataset(hour_data_indexed_df, "H", pair)
    ResultsDatasetGenerator(price_dataset)


def test_dataset_generator_normalize_raw_data(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """"""
    norm_df = ResultsDatasetGenerator.normalize_row_data(hour_data_indexed_df)
    selected_df = ResultsDatasetGenerator.select_data(norm_df, 1000.0)
    assert False
