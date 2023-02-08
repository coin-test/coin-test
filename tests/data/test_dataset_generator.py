"""Test the DatasetGenerator classes."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import ReturnsDatasetGenerator
from coin_test.data.datasets import CustomDataset
from coin_test.data.metadata import MetaData
from coin_test.util import AssetPair, Ticker


def test_dataset_generator_initialized(hour_data_indexed_df: pd.DataFrame) -> None:
    """Initialize the ResultDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    gen = ReturnsDatasetGenerator(mock_dataset)

    assert gen.dataset == mock_dataset
    assert gen.metadata == metadata
    assert isinstance(gen.start, pd.Period)


def test_normalize_dataset(
    hour_data_indexed_df: pd.DataFrame, hour_data_norm_df: pd.DataFrame
) -> None:
    """Normalize data with ResultsDatasetGenerator."""
    norm_df = ReturnsDatasetGenerator.normalize_row_data(hour_data_indexed_df)

    pd.testing.assert_frame_equal(norm_df, hour_data_norm_df)

    # with open("tests/data/assets/norm_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     norm_df.to_csv(outfile, index=False)


def test_dataset_select_data(
    hour_data_norm_df: pd.DataFrame, hour_data_sel_df: pd.DataFrame
) -> None:
    """Select data with ResultsDatasetGenerator."""
    rng = np.random.default_rng(int("stonks", 36))
    starting_price = 7.48
    num_rows = 4

    selected_df = ReturnsDatasetGenerator.select_data(
        hour_data_norm_df, starting_price, num_rows, rng
    )

    pd.testing.assert_frame_equal(selected_df, hour_data_sel_df)

    # with open("tests/data/assets/sel_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     selected_df.to_csv(outfile, index=False)


def test_properly_index_data() -> None:
    """Create a proper PeriodIndex for generated data."""
    freq = "H"
    start = pd.Period("2023-01-01 10:00", freq)
    timedelta = pd.Timedelta(days=1)

    index = ReturnsDatasetGenerator.create_index(start, timedelta, freq)

    assert isinstance(index, pd.PeriodIndex)
    assert len(index) == 25  # 24 hours between first and last point
    assert index[0] == start
    assert index.freq == freq


def test_dataset_generator_create_datasets(
    hour_data_indexed_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with ResultsDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "H"
    metadata = MetaData(pair, freq)
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    gen = ReturnsDatasetGenerator(mock_dataset)
    timedelta = pd.Timedelta(hours=3)

    mocker.patch("coin_test.data.CustomDataset.__new__")

    gen.generate(seed=int("bonks", 36), timedelta=timedelta, n=2)

    dataset_params = CustomDataset.__new__.call_args_list

    assert len(dataset_params) == 2

    (_, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": True}
    assert len(s_df) == 4  # 3 hours between first and last timestamps


@pytest.fixture
def hour_data_norm() -> str:
    """Hourly normalized data CSV filepath."""
    return "tests/data/assets/norm_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_sel() -> str:
    """Hourly normalized data CSV filepath."""
    return "tests/data/assets/sel_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_norm_df(hour_data_norm: str) -> pd.DataFrame:
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
    )
    return df


@pytest.fixture
def hour_data_sel_df(hour_data_sel: str) -> pd.DataFrame:
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
