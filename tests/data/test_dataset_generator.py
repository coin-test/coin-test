"""Test the DatasetGenerator classes."""

import logging
from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import (
    CustomDataset,
    MetaData,
    ReturnsDatasetGenerator,
    SamplingDatasetGenerator,
    StitchedChunkDatasetGenerator,
    WindowStepDatasetGenerator,
)
from coin_test.util import AssetPair, Ticker


def test_window_dataset_generator_initialized(dataset: CustomDataset) -> None:
    """Properly initialize a window step dataset generator."""
    metadata = dataset.metadata

    gen = WindowStepDatasetGenerator(dataset)

    assert gen.dataset == dataset
    assert gen.metadata == metadata


def test_make_slices() -> None:
    """Properly make slices of windows."""
    slices = WindowStepDatasetGenerator.make_slices(10, 4, 3)
    assert slices == [
        slice(0, 4),
        slice(3, 7),
        slice(6, 10),
    ]


def test_warn_make_overlapping_slices(caplog: pytest.LogCaptureFixture) -> None:
    """Warn when making slices of overlapping windows."""
    caplog.set_level(logging.WARN)

    slices = WindowStepDatasetGenerator.make_slices(10, 4, 5)

    assert slices == [
        slice(0, 4),
        slice(2, 6),
        slice(3, 7),
        slice(4, 8),
        slice(6, 10),
    ]

    assert caplog.record_tuples == [
        ("coin_test.data.dataset_generator", logging.WARN, "Windows overlap by 70%")
    ]


def test_err_too_many_slices() -> None:
    """Error when making too many slices."""
    with pytest.raises(ValueError):
        WindowStepDatasetGenerator.make_slices(10, 1, 11)


def test_err_too_big_slices() -> None:
    """Error when making too big slices."""
    with pytest.raises(ValueError):
        WindowStepDatasetGenerator.make_slices(10, 11, 1)


def test_calc_window_length() -> None:
    """Properly calculate the length of a window."""
    freq_1 = "min"
    timedelta_1 = pd.Timedelta(minutes=30)

    freq_2 = "H"
    timedelta_2 = pd.Timedelta(days=40)

    assert WindowStepDatasetGenerator.calc_window_length(freq_1, timedelta_1) == 31
    assert WindowStepDatasetGenerator.calc_window_length(freq_2, timedelta_2) == 961


def test_extract_windows(hour_data_indexed_df: pd.DataFrame) -> None:
    """Properly extract windows from a DataFrame."""
    freq = "H"
    timedelta = pd.Timedelta(hours=4)
    n = 3

    windows = WindowStepDatasetGenerator.extract_windows(
        hour_data_indexed_df, freq, timedelta, n
    )

    assert len(windows) == n
    pd.testing.assert_frame_equal(
        windows[0], cast(pd.DataFrame, hour_data_indexed_df.iloc[:5])
    )
    pd.testing.assert_frame_equal(
        windows[-1], cast(pd.DataFrame, hour_data_indexed_df.iloc[-5:])
    )


def test_create_window_steps(
    dataset: CustomDataset, hour_data_indexed_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Create window + step datasets."""
    metadata = dataset.metadata
    freq = metadata.freq
    pair = metadata.pair
    dataset.name = f"{WindowStepDatasetGenerator.__name__}_0"

    timedelta = pd.Timedelta(hours=3)
    n = 2

    mocker.patch("coin_test.data.WindowStepDatasetGenerator.DATASET_TYPE")

    gen = WindowStepDatasetGenerator(dataset)

    mocker.patch("coin_test.data.WindowStepDatasetGenerator.extract_windows")
    WindowStepDatasetGenerator.extract_windows.return_value = dfs = [
        cast(pd.DataFrame, hour_data_indexed_df.iloc[:10]),
        cast(pd.DataFrame, hour_data_indexed_df.iloc[-10:]),
    ]

    gen.generate(timedelta=timedelta, seed=None, n=n)

    WindowStepDatasetGenerator.extract_windows.assert_called_with(
        dataset.df, freq, timedelta, n
    )

    dataset_params = (
        WindowStepDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore
    )

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert s_name == dataset.name
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": False}
    pd.testing.assert_frame_equal(s_df, dfs[0])


def test_normalize_dataset(
    hour_data_indexed_df: pd.DataFrame, hour_data_norm_df: pd.DataFrame
) -> None:
    """Normalize data with SamplingDatasetGenerator."""
    norm_df = SamplingDatasetGenerator.normalize_row_data(hour_data_indexed_df)

    pd.testing.assert_frame_equal(norm_df, hour_data_norm_df)

    # with open("tests/data/assets/norm_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     norm_df.to_csv(outfile, index=False)


def test_unnormalize_dataset(
    hour_data_df: pd.DataFrame, hour_data_norm_df: pd.DataFrame
) -> None:
    """Unnormalize data with SamplingDatasetGenerator."""
    del hour_data_df["Open Time"]
    hour_data_norm_df.iloc[0, 0] = hour_data_df.iloc[0, 0]  # type: ignore
    unnorm_df = SamplingDatasetGenerator.unnormalize(hour_data_norm_df)

    pd.testing.assert_frame_equal(unnorm_df, hour_data_df, check_like=True)


def test_properly_index_data() -> None:
    """Create a proper PeriodIndex for generated data."""
    freq = "H"
    start = pd.Period("2023-01-01 10:00", freq)
    timedelta = pd.Timedelta(days=1)

    index = SamplingDatasetGenerator.create_index(start, timedelta, freq)

    assert isinstance(index, pd.PeriodIndex)
    assert len(index) == 25  # 24 hours between first and last point
    assert index[0] == start
    assert index.freq == freq


def test_returns_dataset_generator_initialized(
    dataset: CustomDataset,
) -> None:
    """Initialize the ReturnsDatasetGenerator."""
    metadata = dataset.metadata

    gen = ReturnsDatasetGenerator(dataset)

    assert gen.dataset == dataset
    assert gen.metadata == metadata
    assert isinstance(gen.start, pd.Period)


def test_returns_dataset_select_data(
    hour_data_norm_df: pd.DataFrame, hour_data_sel_df: pd.DataFrame
) -> None:
    """Select data with ReturnsDatasetGenerator."""
    rng = np.random.default_rng(int("stonks", 36))
    starting_price = 7.48
    num_rows = 4

    selected_df = ReturnsDatasetGenerator.select_data(
        hour_data_norm_df, starting_price, num_rows, rng
    )

    print(selected_df)
    print(hour_data_sel_df)
    pd.testing.assert_frame_equal(selected_df, hour_data_sel_df)

    # with open("tests/data/assets/sel_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     selected_df.to_csv(outfile, index=False)


def test_returns_dataset_generator_create_datasets(
    dataset: CustomDataset, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with ReturnsDatasetGenerator."""
    metadata = dataset.metadata
    freq = metadata.freq
    pair = metadata.pair
    dataset.name = f"{ReturnsDatasetGenerator.__name__}_0"

    gen = ReturnsDatasetGenerator(dataset)
    timedelta = pd.Timedelta(hours=3)

    mocker.patch("coin_test.data.ReturnsDatasetGenerator.DATASET_TYPE")

    gen.generate(seed=int("bonks", 36), timedelta=timedelta, n=2)

    dataset_params = ReturnsDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_name == dataset.name
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": True}
    assert len(s_df) == 4  # 3 hours between first and last timestamps


def test_chunked_dataset_generator_initialized(
    dataset: CustomDataset,
) -> None:
    """Initialize the StitchedChunkDatasetGenerator."""
    metadata = dataset.metadata
    chunk_size = 5

    gen = StitchedChunkDatasetGenerator(dataset, chunk_size)

    assert gen.dataset == dataset
    assert gen.metadata == metadata
    assert gen.chunk_size == chunk_size
    assert isinstance(gen.start, pd.Period)


def test_chunked_dataset_generator_err_on_initialization(
    dataset: CustomDataset,
) -> None:
    """Fail to initialize the StitchedChunkDatasetGenerator."""
    with pytest.raises(ValueError):
        StitchedChunkDatasetGenerator(dataset, chunk_size=-1)

    with pytest.raises(ValueError):
        StitchedChunkDatasetGenerator(dataset, chunk_size=2**32)


def test_chunked_dataset_select_data(
    hour_data_norm_df: pd.DataFrame, hour_data_sel_chunk_df: pd.DataFrame
) -> None:
    """Select data with StitchedChunkDatasetGenerator."""
    rng = np.random.default_rng(int("stonks", 36))
    starting_price = 7.48
    num_rows = 5
    chunk_size = 2

    selected_df = StitchedChunkDatasetGenerator.select_data(
        hour_data_norm_df, starting_price, num_rows, chunk_size, rng
    )

    pd.testing.assert_frame_equal(selected_df, hour_data_sel_chunk_df)

    # with open("tests/data/assets/sel_chunk_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     selected_df.to_csv(outfile, index=False)


def test_chunk_dataset_generator_create_datasets(
    dataset: CustomDataset, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with StitchedChunkDatasetGenerator."""
    metadata = dataset.metadata
    freq = metadata.freq
    pair = metadata.pair
    dataset.name = f"{StitchedChunkDatasetGenerator.__name__}_0"

    chunk_size = 3  # 3 rows of data
    gen = StitchedChunkDatasetGenerator(dataset, chunk_size)
    timedelta = pd.Timedelta(hours=10)

    mocker.patch("coin_test.data.StitchedChunkDatasetGenerator.DATASET_TYPE")

    gen.generate(seed=int("chonks", 36), timedelta=timedelta, n=2)

    dataset_params = (
        StitchedChunkDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore
    )

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_name == dataset.name
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": True}
    assert len(s_df) == 11  # 10 hours between first and last timestamps


@pytest.fixture
def hour_data_norm() -> str:
    """Hourly normalized data CSV filepath."""
    return "tests/data/assets/norm_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_norm_df(hour_data_norm: str) -> pd.DataFrame:
    """Hourly normalized data contents with period index."""
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
def dataset(hour_data_indexed_df: pd.DataFrame) -> CustomDataset:
    """A mock dataset with data and metadata."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "H"
    metadata = MetaData(pair, freq)
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    return mock_dataset


@pytest.fixture
def hour_data_sel() -> str:
    """Hourly selected data CSV filepath."""
    return "tests/data/assets/sel_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_sel_df(hour_data_sel: str) -> pd.DataFrame:
    """Hourly selected data contents with period index."""
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


@pytest.fixture
def hour_data_sel_chunk() -> str:
    """Hourly selected (by chunk) data CSV filepath."""
    return "tests/data/assets/sel_chunk_eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_sel_chunk_df(hour_data_sel_chunk: str) -> pd.DataFrame:
    """Hourly selected (by chunk) data contents with period index."""
    dtypes = {
        "Open": float,
        "High": float,
        "Low": float,
        "Volume": float,
        "Close": float,
    }
    df = pd.read_csv(
        hour_data_sel_chunk,
        dtype=dtypes,  # type: ignore
    )
    return df
