"""Test the DatasetGenerator classes."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import (
    GarchDatasetGenerator,
    GarchSettings,
    ReturnsDatasetGenerator,
    StitchedChunkDatasetGenerator,
)
from coin_test.data.metadata import MetaData
from coin_test.util import AssetPair, Ticker


def test_chunked_dataset_generator_initialized(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Initialize the StitchedChunkDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df
    chunk_size = 5

    gen = StitchedChunkDatasetGenerator(mock_dataset, chunk_size)

    assert gen.dataset == mock_dataset
    assert gen.metadata == metadata
    assert gen.chunk_size == chunk_size
    assert isinstance(gen.start, pd.Period)


def test_returns_dataset_generator_initialized(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Initialize the ReturnsDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    gen = ReturnsDatasetGenerator(mock_dataset)

    assert gen.dataset == mock_dataset
    assert gen.metadata == metadata
    assert gen.chunk_size == 1
    assert isinstance(gen.start, pd.Period)


def test_chunked_dataset_generator_err_on_initialization(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Fail to initialize the StitchedChunkDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    with pytest.raises(ValueError):
        StitchedChunkDatasetGenerator(mock_dataset, chunk_size=-1)

    with pytest.raises(ValueError):
        StitchedChunkDatasetGenerator(mock_dataset, chunk_size=1000000000000000)


def test_normalize_dataset(
    hour_data_indexed_df: pd.DataFrame, hour_data_norm_df: pd.DataFrame
) -> None:
    """Normalize data with ResultsDatasetGenerator."""
    norm_df = StitchedChunkDatasetGenerator.normalize_row_data(hour_data_indexed_df)

    pd.testing.assert_frame_equal(norm_df, hour_data_norm_df)

    # with open("tests/data/assets/norm_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     norm_df.to_csv(outfile, index=False)


def test_chunked_dataset_select_data(
    hour_data_norm_df: pd.DataFrame, hour_data_sel_df: pd.DataFrame
) -> None:
    """Select data with ResultsDatasetGenerator."""
    rng = np.random.default_rng(int("stonks", 36))
    starting_price = 7.48
    num_rows = 5
    chunk_size = 2

    selected_df = StitchedChunkDatasetGenerator.select_data(
        hour_data_norm_df, starting_price, num_rows, chunk_size, rng
    )

    pd.testing.assert_frame_equal(selected_df, hour_data_sel_df)

    # with open("tests/data/assets/sel_eth_usdc_1h_9_28.csv", "w") as outfile:
    #     selected_df.to_csv(outfile, index=False)


def test_properly_index_data() -> None:
    """Create a proper PeriodIndex for generated data."""
    freq = "H"
    start = pd.Period("2023-01-01 10:00", freq)
    timedelta = pd.Timedelta(days=1)

    index = StitchedChunkDatasetGenerator.create_index(start, timedelta, freq)

    assert isinstance(index, pd.PeriodIndex)
    assert len(index) == 25  # 24 hours between first and last point
    assert index[0] == start
    assert index.freq == freq


def test_returns_dataset_generator_create_datasets(
    hour_data_indexed_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with ResultsDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "H"
    metadata = MetaData(pair, freq)
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df
    mock_dataset.name = f"{ReturnsDatasetGenerator.__name__}_0"

    gen = ReturnsDatasetGenerator(mock_dataset)
    timedelta = pd.Timedelta(hours=3)

    mocker.patch("coin_test.data.ReturnsDatasetGenerator.DATASET_TYPE")

    gen.generate(seed=int("bonks", 36), timedelta=timedelta, n=2)

    dataset_params = ReturnsDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_name == mock_dataset.name
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": True}
    assert len(s_df) == 4  # 3 hours between first and last timestamps


def test_chunk_dataset_generator_create_datasets(
    hour_data_indexed_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with StitchedChunkDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "H"
    chunk_size = 3  # 3 rows of data
    metadata = MetaData(pair, freq)

    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df
    mock_dataset.name = f"{StitchedChunkDatasetGenerator.__name__}_0"

    gen = StitchedChunkDatasetGenerator(mock_dataset, chunk_size)
    timedelta = pd.Timedelta(hours=10)

    mocker.patch("coin_test.data.StitchedChunkDatasetGenerator.DATASET_TYPE")

    gen.generate(seed=int("chonks", 36), timedelta=timedelta, n=2)

    dataset_params = (
        StitchedChunkDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore
    )

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_name == mock_dataset.name
    assert s_freq == freq
    assert s_pair == pair
    assert s_opts == {"synthetic": True}
    assert len(s_df) == 11  # 10 hours between first and last timestamps


def test_garch_dataset_generator_initialized(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Initialize the GarchDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df
    chunk_size = 5
    garch_settings = GarchSettings(
        mean="Constant",
        lags=0,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        power=2,
        dist="normal",
        hold_back=None,
        rescale=None,
    )

    gen = GarchDatasetGenerator(mock_dataset, chunk_size)

    assert gen.dataset == mock_dataset
    assert gen.metadata == metadata
    assert gen.chunk_size == chunk_size
    assert isinstance(gen.start, pd.Period)
    assert gen.garch_settings.mean == garch_settings.mean
    assert gen.garch_settings.lags == garch_settings.lags
    assert gen.garch_settings.vol == garch_settings.vol
    assert gen.garch_settings.p == garch_settings.p
    assert gen.garch_settings.o == garch_settings.o
    assert gen.garch_settings.q == garch_settings.q
    assert gen.garch_settings.power == garch_settings.power
    assert gen.garch_settings.dist == garch_settings.dist
    assert gen.garch_settings.hold_back == garch_settings.hold_back
    assert gen.garch_settings.rescale == garch_settings.rescale


def test_garch_dataset_generator_err_on_initialization(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Fail to initialize the GarchDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    metadata = MetaData(pair, freq="H")
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df

    with pytest.raises(ValueError):
        GarchDatasetGenerator(mock_dataset, chunk_size=-1)

    with pytest.raises(ValueError):
        GarchDatasetGenerator(mock_dataset, chunk_size=1000000000000000)


def test_garch_get_model_parameters(
    hour_data_indexed_df: pd.DataFrame, garch_parameters: pd.Series
) -> None:
    """Test getting GARCH model parameters estimated from fitting to series."""
    univariate_series = 100 * hour_data_indexed_df["Close"].pct_change().dropna()
    garch_settings = GarchSettings(
        mean="Constant",
        lags=0,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        power=2,
        dist="normal",
        hold_back=None,
        rescale=None,
    )

    np.random.RandomState(int("stonks", 36))
    test_garch_model_params = GarchDatasetGenerator.get_garch_model_parameters(
        univariate_series, garch_settings
    )

    pd.testing.assert_series_equal(test_garch_model_params, garch_parameters)


def test_garch_dataset_sample_series(
    hour_data_indexed_df: pd.DataFrame, hour_data_sampled_series: pd.Series
) -> None:
    """Test sampling chunks from series."""
    rng = np.random.default_rng(int("stonks", 36))
    num_rows = 5
    chunk_size = 2
    series = hour_data_indexed_df["High"]

    test_sampled_series = GarchDatasetGenerator.sample_series(
        series, num_rows, chunk_size, rng
    )

    pd.testing.assert_series_equal(test_sampled_series, hour_data_sampled_series)


def test_garch_high_to_pct_change_extreme_open_and_close(
    hour_data_indexed_df: pd.DataFrame,
    pct_change_high_series: pd.Series,
) -> None:
    """Test transform high series to percent change of max of each open and close."""
    open_series = hour_data_indexed_df["Open"]
    close_series = hour_data_indexed_df["Close"]
    high_series = hour_data_indexed_df["High"]
    extreme = "max"

    test_pct_change_high_series = (
        GarchDatasetGenerator.to_pct_change_based_on_extreme_bar_open_and_close(
            series_to_pct_change=high_series,
            open_series=open_series,
            close_series=close_series,
            extreme=extreme,
        )
    )

    pd.testing.assert_series_equal(test_pct_change_high_series, pct_change_high_series)


def test_garch_low_to_pct_change_extreme_open_and_close(
    hour_data_indexed_df: pd.DataFrame,
    pct_change_low_series: pd.Series,
) -> None:
    """Test transform low series to percent change of min of each open and close."""
    open_series = hour_data_indexed_df["Open"]
    close_series = hour_data_indexed_df["Close"]
    low_series = hour_data_indexed_df["Low"]
    extreme = "min"

    test_pct_change_low_series = (
        GarchDatasetGenerator.to_pct_change_based_on_extreme_bar_open_and_close(
            series_to_pct_change=low_series,
            open_series=open_series,
            close_series=close_series,
            extreme=extreme,
        )
    )

    pd.testing.assert_series_equal(test_pct_change_low_series, pct_change_low_series)


def test_garch_high_from_pct_change_extreme_open_and_close(
    pct_change_high_series: pd.Series,
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Test transform high percent change to value using max of open and close."""
    extreme = "max"
    open_series = hour_data_indexed_df["Open"]
    close_series = hour_data_indexed_df["Close"]
    high_series = hour_data_indexed_df["High"].reset_index(drop=True).rename(None)

    test_high_series = (
        GarchDatasetGenerator.from_pct_change_based_on_extreme_bar_open_and_close(
            pct_change_series=pct_change_high_series,
            extreme=extreme,
            open_series=open_series,
            close_series=close_series,
        )
    )

    pd.testing.assert_series_equal(test_high_series, high_series)


def test_garch_low_from_pct_change_extreme_open_and_close(
    pct_change_low_series: pd.Series,
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Test transform low percent change to value using min of open and close."""
    extreme = "min"
    open_series = hour_data_indexed_df["Open"]
    close_series = hour_data_indexed_df["Close"]
    low_series = hour_data_indexed_df["Low"].reset_index(drop=True).rename(None)

    test_low_series = (
        GarchDatasetGenerator.from_pct_change_based_on_extreme_bar_open_and_close(
            pct_change_series=pct_change_low_series,
            extreme=extreme,
            open_series=open_series,
            close_series=close_series,
        )
    )

    pd.testing.assert_series_equal(test_low_series, low_series)


def test_garch_err_len_open_v_close_from_pct_change_extreme_open_and_close() -> None:
    """Test transform pct change errors if open and close series different lengths."""
    extreme = "min"
    pct_change_series = pd.Series([1, 2, 3])
    open_series = pd.Series([1, 2, 3])
    close_series = pd.Series([4, 5])

    with pytest.raises(ValueError):
        GarchDatasetGenerator.from_pct_change_based_on_extreme_bar_open_and_close(
            pct_change_series=pct_change_series,
            extreme=extreme,
            open_series=open_series,
            close_series=close_series,
        )


def test_garch_err_len_open_v_series_from_pct_change_extreme_open_and_close() -> None:
    """Test transform pct change errors if open and series different lengths."""
    extreme = "min"
    pct_change_series = pd.Series([1, 2])
    open_series = pd.Series([1, 2, 3])
    close_series = pd.Series([4, 5, 6])

    with pytest.raises(ValueError):
        GarchDatasetGenerator.from_pct_change_based_on_extreme_bar_open_and_close(
            pct_change_series=pct_change_series,
            extreme=extreme,
            open_series=open_series,
            close_series=close_series,
        )


def test_garch_from_pct_change_based_on_starting_value(
    hour_data_indexed_df: pd.DataFrame,
) -> None:
    """Test transform percent change series to value series based on starting value."""
    series = hour_data_indexed_df["Close"].reset_index(drop=True).rename(None)
    starting_price = hour_data_indexed_df["Close"].iloc[0]

    pct_change_series = 100 * series.pct_change().dropna()

    np.random.RandomState(int("stonks", 36))
    test_series = GarchDatasetGenerator.from_pct_change_based_on_starting_value(
        pct_change_series=pct_change_series,
        starting_price=starting_price,
    )

    pd.testing.assert_series_equal(test_series, series)


def test_garch_dataset_generator_create_datasets(
    hour_data_indexed_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Create synthetic datasets with GarchDatasetGenerator."""
    mock_dataset = Mock()
    pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "H"
    metadata = MetaData(pair, freq)
    mock_dataset.metadata = metadata
    mock_dataset.df = hour_data_indexed_df
    mock_dataset.name = f"{GarchDatasetGenerator.__name__}_0"

    gen = GarchDatasetGenerator(mock_dataset)
    timedelta = pd.Timedelta(hours=3)

    mocker.patch("coin_test.data.GarchDatasetGenerator.DATASET_TYPE")

    gen.generate(seed=int("bonks", 36), timedelta=timedelta, n=2)

    dataset_params = GarchDatasetGenerator.DATASET_TYPE.call_args_list  # type: ignore

    assert len(dataset_params) == 2

    (s_name, s_df, s_freq, s_pair), s_opts = dataset_params[0]

    assert isinstance(s_df.index, pd.PeriodIndex)
    assert s_name == mock_dataset.name
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


@pytest.fixture
def hour_data_sampled_series() -> pd.Series:
    """Hourly data High series randomly sampled contents."""
    return pd.Series([1293.11, 1286.32, 1288.3, 1289.02, 1286.32])


@pytest.fixture
def garch_parameters() -> pd.Series:
    """GARCH(1,1) parameters estimated from hourly close series pct change."""
    return pd.Series(
        {
            "mu": 2.229744e-01,
            "omega": 3.962079e-01,
            "alpha[1]": 6.750067e-01,
            "beta[1]": 7.326199e-13,
        },
        dtype=float,
        name="params",
    )


@pytest.fixture
def pct_change_high_series() -> pd.Series:
    """High series percent change relative to max of Open and Close series."""
    return pd.Series(
        [
            0.00494225,
            0.00189632,
            0.00154954,
            0.00094011,
            0.00147616,
            0.0006687,
            0.0050286,
            0.0006301,
            0.00015557,
            0.0008108,
            0.00604475,
            0.00226453,
            0.00571091,
            0.00583515,
            0.00422396,
            0.00066812,
            0.00524637,
            0.00434662,
            0.00130429,
            0.00691837,
            0.00409833,
            0.00285029,
            0.00071338,
            0.00305295,
        ],
        dtype=float,
    )


@pytest.fixture
def pct_change_low_series() -> pd.Series:
    """Low series percent change relative to min of Open and Close series."""
    return pd.Series(
        [
            0.00000000e00,
            -3.53262528e-03,
            -7.28372049e-03,
            -9.63119568e-04,
            -2.21188969e-03,
            -4.97016344e-03,
            -8.60713140e-05,
            -9.12006257e-03,
            -4.22638449e-03,
            -2.74142617e-03,
            -1.50220496e-02,
            -5.68405607e-03,
            -1.32022884e-03,
            -4.15688801e-03,
            -4.27200464e-03,
            -2.83555301e-03,
            -6.09361600e-04,
            -8.23683355e-03,
            -6.06385955e-03,
            -7.53092254e-03,
            -8.76292907e-04,
            -2.49015090e-03,
            -2.99114621e-03,
            -1.35308893e-03,
        ],
        dtype=float,
    )
