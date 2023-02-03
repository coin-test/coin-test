"""Test the DatasetGenerator classes."""

import numpy as np
import pandas as pd
import pytest

from coin_test.data import ResultsDatasetGenerator
from coin_test.data.datasets import CustomDataset
from coin_test.util import AssetPair, Ticker


def test_dataset_generator_initialized(hour_data_indexed_df: pd.DataFrame) -> None:
    """Initialize the ResultDatasetGenerator."""
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    price_dataset = CustomDataset(hour_data_indexed_df, "H", pair)
    ResultsDatasetGenerator(price_dataset)


def test_normalize_dataset(
    hour_data_indexed_df: pd.DataFrame, hour_data_norm_indexed_df: pd.DataFrame
) -> None:
    """Normalize data with ResultsDatasetGenerator."""
    norm_df = ResultsDatasetGenerator.normalize_row_data(hour_data_indexed_df)

    norm_df.reset_index(drop=True, inplace=True)
    hour_data_norm_indexed_df.reset_index(drop=True, inplace=True)

    df_diff = norm_df - hour_data_norm_indexed_df
    assert df_diff.max().max() < 1e-3  # all values are very close

    # with open("tests/data/assets/norm_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     norm_df.to_csv(outfile)


def test_dataset_select_data(
    hour_data_norm_indexed_df: pd.DataFrame, hour_data_sel_indexed_df: pd.DataFrame
) -> None:
    """Select data with ResultsDatasetGenerator."""
    rng = np.random.default_rng(int("stonks", 36))
    starting_price = 7.48

    selected_df = ResultsDatasetGenerator.select_data(
        hour_data_norm_indexed_df, starting_price, rng, hour_data_norm_indexed_df.index
    )

    # with open("tests/data/assets/sel_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     selected_df.to_csv(outfile, index=False)

    selected_df.reset_index(drop=True, inplace=True)
    hour_data_sel_indexed_df.reset_index(drop=True, inplace=True)

    df_diff = selected_df - hour_data_sel_indexed_df
    assert df_diff.max().max() < 1e-3  # all values are very close


def test_dataset_generator_create_datasets(hour_data_indexed_df: pd.DataFrame) -> None:
    """Create synthetic datasets with ResultsDatasetGenerator."""
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    price_dataset = CustomDataset(hour_data_indexed_df, "H", pair)
    gen = ResultsDatasetGenerator(price_dataset)

    datasets = gen.generate(seed=int("bonks", 36), n=2)

    assert len(datasets) == 2
    for dataset in datasets:
        assert dataset.df.shape == price_dataset.df.shape
        # assert dataset._metadata == price_dataset._metadata


@pytest.fixture
def hour_data_norm() -> str:
    """Hourly normalized data CSV filepath."""
    return "tests/data/assets/norm_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_sel() -> str:
    """Hourly normalized data CSV filepath."""
    return "tests/data/assets/sel_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_norm_indexed_df(hour_data_norm: str) -> pd.DataFrame:
    """Hourly data contents with period index."""
    dtypes = {
        "Open": float,
        "High": float,
        "Low": float,
        "Volume": float,
        "Close": float,
    }
    df = pd.read_csv(
        hour_data_norm,
        dtype=dtypes,  # type: ignore
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    df.index.to_period(freq="H", inplace=True)  # type: ignore
    return df


@pytest.fixture
def hour_data_sel_indexed_df(hour_data_sel: str) -> pd.DataFrame:
    """Hourly data contents with period index."""
    dtypes = {
        "Open": float,
        "High": float,
        "Low": float,
        "Volume": float,
        "Close": float,
    }
    df = pd.read_csv(
        hour_data_sel,
        dtype=dtypes,  # type: ignore
    )
    return df
