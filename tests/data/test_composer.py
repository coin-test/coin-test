"""Test the Dataset class."""

from unittest.mock import Mock, PropertyMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import Composer, MetaData
from coin_test.util import AssetPair, Ticker


@pytest.fixture
def mocked_dataset(hour_data_df: pd.DataFrame) -> Mock:
    """Mock that contains a PeriodIndex DataFrame."""
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

    dataset = Mock()
    df_mock = PropertyMock(return_value=hour_data_df)
    type(dataset).df = df_mock
    return dataset


def test_is_within_range_true(mocked_dataset: Mock) -> None:
    """Validates dataset in time range."""
    start_time = pd.Timestamp("2000-11")
    end_time = pd.Timestamp("2023-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time)


def test_is_within_range_start_off(mocked_dataset: Mock) -> None:
    """Validates dataset in time range."""
    start_time = pd.Timestamp("2000-11")
    end_time = pd.Timestamp("2040-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time) is False


def test_is_within_range_end_off(mocked_dataset: Mock) -> None:
    """Validates dataset in time range."""
    start_time = pd.Timestamp("1999-11")
    end_time = pd.Timestamp("2021-4")
    assert Composer._is_within_range(mocked_dataset, start_time, end_time) is False


def test_get_shared_currency() -> None:
    """Return shared currency."""
    datasets = []
    base = Ticker("USDT")
    for asset in ("BTC", "ETH", "DOGE"):
        metadata = MetaData(AssetPair(Ticker(asset), base), "H")
        dataset = Mock()
        metadata_mock = PropertyMock(return_value=metadata)
        type(dataset).metadata = metadata_mock
        datasets.append(dataset)
    shared = Composer._get_shared_currency(datasets)
    assert shared == base


def test_get_shared_currency_invalid() -> None:
    """Return shared currency."""
    datasets = []
    for asset, currency in (("BTC", "USDT"), ("ETH", "USDC")):
        metadata = MetaData(AssetPair(Ticker(asset), Ticker(currency)), "H")
        dataset = Mock()
        metadata_mock = PropertyMock(return_value=metadata)
        type(dataset).metadata = metadata_mock
        datasets.append(dataset)
    shared = Composer._get_shared_currency(datasets)
    assert shared is None


@pytest.mark.parametrize(
    "freqs,target",
    [(("Y", "M", "D"), "1D"), (("W", "M", "Q"), "1W"), (("H", "T", "S"), "1S")],
)
def test_get_min_freq(freqs: tuple[str], target: str) -> None:
    """Return smallest timedelta."""
    datasets = []
    for freq in freqs:
        metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USD")), freq)
        dataset = Mock()
        metadata_mock = PropertyMock(return_value=metadata)
        type(dataset).metadata = metadata_mock
        datasets.append(dataset)

    min_delta = Composer._get_min_freq(datasets)
    assert min_delta == pd.to_timedelta(target)


def test_composer_init(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initializes correctly."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    dataset = Mock()
    df_mock = PropertyMock(return_value=simple_df)
    metadata_mock = PropertyMock(return_value=metadata)
    mocker.patch("coin_test.data.Composer._is_within_range")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    mocker.patch("coin_test.data.Composer._get_min_freq")

    type(dataset).df = df_mock
    type(dataset).metadata = metadata_mock
    Composer._is_within_range.return_value = True
    Composer._get_shared_currency.return_value = metadata.pair.currency

    composer = Composer([dataset], start_time, end_time)

    pd.testing.assert_frame_equal(composer.datasets[metadata.pair].df, simple_df)

    df_mock.assert_called()
    metadata_mock.assert_called()
    Composer._is_within_range.assert_called_once_with(dataset, start_time, end_time)
    Composer._get_shared_currency.assert_called_once_with([dataset])
    Composer._get_min_freq.assert_called_once_with([dataset])

    assert hasattr(composer, "datasets")
    assert hasattr(composer, "freq")
    assert composer.start_time == start_time
    assert composer.end_time == end_time
    assert composer.currency == metadata.pair.currency


def test_composer_invalid_range() -> None:
    """Errors on invalid time range."""
    start_time = pd.Timestamp("2021")
    end_time = pd.Timestamp("2020")

    with pytest.raises(ValueError) as e:
        Composer([Mock()], start_time, end_time)
        assert "earlier than end time" in str(e)


def test_composer_no_datasets() -> None:
    """Error on no datasets passed."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    with pytest.raises(ValueError) as e:
        Composer([], start_time, end_time)
        assert "At least one dataset must be defined." in str(e)


def test_composer_not_within_range(mocker: MockerFixture) -> None:
    """Error on dataset not within range."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    mocker.patch("coin_test.data.Composer._is_within_range")
    Composer._is_within_range.return_value = False

    with pytest.raises(ValueError) as e:
        Composer([Mock()], start_time, end_time)
        assert "Not all datasets cover requested time range" in str(e)


def test_composer_not_shared_currency(mocker: MockerFixture) -> None:
    """Errors on dataset not sharing currency."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    mocker.patch("coin_test.data.Composer._is_within_range")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    Composer._is_within_range.return_value = True
    Composer._get_shared_currency.return_value = None

    with pytest.raises(ValueError) as e:
        Composer([Mock()], start_time, end_time)
        assert "Not all datasets share a single currency." in str(e)
