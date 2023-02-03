"""Test the Composer class."""

from typing import cast
from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from pytest_mock import MockerFixture

from coin_test.data import Composer, MetaData
from coin_test.util import AssetPair, Ticker


def _mock_dataset(
    df: pd.DataFrame | None, metadata: MetaData | None
) -> tuple[Mock, PropertyMock, PropertyMock]:
    dataset = Mock()
    df_mock = PropertyMock(return_value=df)
    metadata_mock = PropertyMock(return_value=metadata)
    type(dataset).df = df_mock
    type(dataset).metadata = metadata_mock
    return dataset, df_mock, metadata_mock


def _patch_composer_val(currency: Ticker, mocker: MockerFixture) -> None:
    mocker.patch("coin_test.data.Composer._validate_params")
    mocker.patch("coin_test.data.Composer._get_min_freq")
    Composer._validate_params.return_value = currency


@pytest.fixture
def period_index_df(hour_data_df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe with PeriodIndex."""
    years = [
        "2000",
        "2001",
        "2002",
        "2003",
        "2004",
        "2005",
        "2006",
        "2007",
        "2008",
        "2009",
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
    ]
    index = pd.PeriodIndex(years, freq="Y")  # type: ignore
    hour_data_df.set_index(index, inplace=True)
    return hour_data_df


@pytest.fixture
def mocked_dataset(period_index_df: pd.DataFrame) -> Mock:
    """Mock that contains a PeriodIndex DataFrame."""
    dataset, _, _ = _mock_dataset(period_index_df, None)
    return dataset


def test_is_within_range_true(mocked_dataset: Mock) -> None:
    """Validate dataset in time range."""
    start_time = pd.Timestamp("2000-11")
    end_time = pd.Timestamp("2023-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time)


def test_is_within_range_start_off(mocked_dataset: Mock) -> None:
    """Reject dataset missing end date."""
    start_time = pd.Timestamp("2000-11")
    end_time = pd.Timestamp("2040-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time) is False


def test_is_within_range_end_off(mocked_dataset: Mock) -> None:
    """Reject dataset missing start date."""
    start_time = pd.Timestamp("1999-11")
    end_time = pd.Timestamp("2021-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time) is False


def test_missing_data_true(period_index_df: pd.DataFrame) -> None:
    """Return true on full dataset."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    assert Composer._validate_missing_data(dataset)


def test_missing_data_missing(period_index_df: pd.DataFrame) -> None:
    """Return false on missing data."""
    period_index_df.drop(index=period_index_df.index[3], inplace=True)
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    print(period_index_df)
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    assert not Composer._validate_missing_data(dataset)


def test_get_shared_currency() -> None:
    """Return shared currency."""
    datasets = []
    base = Ticker("USDT")
    for asset in ("BTC", "ETH", "DOGE"):
        metadata = MetaData(AssetPair(Ticker(asset), base), "H")
        dataset, _, _ = _mock_dataset(None, metadata)
        datasets.append(dataset)
    shared = Composer._get_shared_currency(datasets)
    assert shared == base


def test_get_shared_currency_invalid() -> None:
    """Return None when no shared currency."""
    datasets = []
    for asset, currency in (("BTC", "USDT"), ("ETH", "USDC")):
        metadata = MetaData(AssetPair(Ticker(asset), Ticker(currency)), "H")
        dataset, _, _ = _mock_dataset(None, metadata)
        datasets.append(dataset)
    shared = Composer._get_shared_currency(datasets)
    assert shared is None


@pytest.mark.parametrize(
    "freqs,target",
    [
        (("Y",), to_offset("Y")),
        (("Y", "M"), to_offset("M")),
        (("Y", "M", "W"), to_offset("W")),
        (("Y", "M", "D"), to_offset("D")),
        (("Y", "H", "D"), to_offset("H")),
        (("H", "T", "S"), to_offset("S")),
    ],
)
def test_get_min_freq(freqs: tuple[str], target: pd.DateOffset) -> None:
    """Return smallest timedelta."""
    datasets = []
    for freq in freqs:
        metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USD")), freq)
        dataset, _, _ = _mock_dataset(None, metadata)
        datasets.append(dataset)
    min_delta = Composer._get_min_freq(datasets)
    assert min_delta == target


def test_validate_params(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Call validation functions."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(simple_df, metadata)

    mocker.patch("coin_test.data.Composer._is_within_range")
    mocker.patch("coin_test.data.Composer._validate_missing_data")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    Composer._get_shared_currency.return_value = metadata.pair.currency

    ticker = Composer._validate_params([dataset], start_time, end_time)

    Composer._is_within_range.assert_called_once_with(dataset, start_time, end_time)
    Composer._validate_missing_data.assert_called_once_with(dataset)
    Composer._get_shared_currency.assert_called_once_with([dataset])
    assert ticker == metadata.pair.currency


def test_validate_params_invalid_range() -> None:
    """Error on invalid time range."""
    start_time = pd.Timestamp("2021")
    end_time = pd.Timestamp("2020")

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], start_time, end_time)
        assert "earlier than end time" in str(e)


def test_validate_params_no_datasets() -> None:
    """Error on no datasets passed."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    with pytest.raises(ValueError) as e:
        Composer._validate_params([], start_time, end_time)
        assert "At least one dataset must be defined." in str(e)


def test_validate_params_not_within_range(mocker: MockerFixture) -> None:
    """Error on dataset not within range."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    mocker.patch("coin_test.data.Composer._is_within_range")
    Composer._is_within_range.return_value = False

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], start_time, end_time)
        assert "Not all datasets cover requested time range" in str(e)


def test_validate_params_not_shared_currency(mocker: MockerFixture) -> None:
    """Error on dataset not sharing currency."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    mocker.patch("coin_test.data.Composer._is_within_range")
    mocker.patch("coin_test.data.Composer._validate_missing_data")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    Composer._is_within_range.return_value = True
    Composer._validate_missing_data.return_value = True
    Composer._get_shared_currency.return_value = None

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], start_time, end_time)
        assert "Not all datasets share a single currency." in str(e)


def test_validate_params_missing_data(mocker: MockerFixture) -> None:
    """Error on dataset missing data."""
    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")

    mocker.patch("coin_test.data.Composer._is_within_range")
    mocker.patch("coin_test.data.Composer._validate_missing_data")
    Composer._is_within_range.return_value = True
    Composer._validate_missing_data.return_value = False

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], start_time, end_time)
    assert "does not have data for every period." in str(e)


def test_composer_init(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initialize correctly."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, df_mock, metadata_mock = _mock_dataset(simple_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    composer = Composer([dataset], start_time, end_time)

    pd.testing.assert_frame_equal(composer.datasets[metadata.pair].df, simple_df)

    df_mock.assert_called()
    metadata_mock.assert_called()
    Composer._validate_params.assert_called_once_with([dataset], start_time, end_time)
    Composer._get_min_freq.assert_called_once_with([dataset])

    assert hasattr(composer, "datasets")
    assert hasattr(composer, "freq")
    assert composer.start_time == start_time
    assert composer.end_time == end_time
    assert composer.currency == metadata.pair.currency


def test_composer_get_range(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Get range of data."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    data = composer.get_range(start_time, end_time, mask=False)

    assert len(data) == 1
    assert metadata.pair in data

    df = data[metadata.pair]
    assert len(df) == 4


def test_composer_get_range_higher_resolution(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Get range of data with higher resolution query."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    start_time = pd.Timestamp("2019-12-30")
    end_time = pd.Timestamp("2022-1-15")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    data = composer.get_range(start_time, end_time, mask=False)

    assert len(data) == 1
    assert metadata.pair in data

    df = data[metadata.pair]
    assert len(df) == 4


def test_composer_get_range_single(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Get single timestep."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2019")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    data = composer.get_range(start_time, end_time, mask=False)

    df = data[metadata.pair]
    assert len(df) == 1


def test_composer_get_range_masked(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Mask final timestep of data."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    data = composer.get_range(start_time, end_time, mask=True)

    final_row = data[metadata.pair].iloc[-1]
    for col, val in final_row.items():
        if col == "Open":
            assert not np.isnan(val)
        else:
            assert np.isnan(val)


def test_composer_get_range_skip_bad_key(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Ignore non-existent keys."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    pair = AssetPair(Ticker("Test"), Ticker("Test"))
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    data = composer.get_range(start_time, end_time, keys=[pair])
    assert len(data) == 0


def test_composer_get_timestep(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Calls get_range correctly."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    timestep = pd.Timestamp("2019")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    composer.get_range = Mock()

    composer.get_timestep(timestep, mask=True)
    composer.get_range.assert_called_once_with(timestep, timestep, keys=None, mask=True)


def test_composer_get_lookback(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Calls get_range correctly."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(metadata.pair.currency, mocker)

    timestep = pd.Timestamp("2019")
    composer = Composer([dataset], pd.Timestamp("2008"), pd.Timestamp("2022"))
    composer.freq = cast(pd.DateOffset, to_offset("Y"))
    composer.get_range = Mock()

    composer.get_lookback(timestep, 8, mask=True)
    composer.get_range.assert_called_once_with(
        pd.Timestamp("2011-12-31"), timestep, keys=None, mask=True
    )
