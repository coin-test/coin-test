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


def _patch_composer_val(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    currency: Ticker,
    mocker: MockerFixture,
) -> None:
    mocker.patch("coin_test.data.Composer._validate_params")
    mocker.patch("coin_test.data.Composer._get_min_freq")
    Composer._validate_params.return_value = (start_time, end_time, currency)


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


def test_missing_data_true(period_index_df: pd.DataFrame) -> None:
    """Return true on full dataset."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    assert Composer._validate_missing_data(dataset)


def test_missing_data_missing(period_index_df: pd.DataFrame) -> None:
    """Return false on missing data."""
    period_index_df.drop(index=period_index_df.index[3], inplace=True)
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    assert not Composer._validate_missing_data(dataset)


def test_get_start_end_single(period_index_df: pd.DataFrame) -> None:
    """Return start and end time."""
    length = pd.DateOffset(years=1)
    dataset, _, _ = _mock_dataset(period_index_df, None)
    start_end = Composer._get_start_end([dataset], length)

    assert start_end is not None
    start_time, end_time = start_end
    assert start_time == pd.Timestamp("2000")
    assert end_time == pd.Timestamp("2001")


def test_get_start_end_single_exact(period_index_df: pd.DataFrame) -> None:
    """Return start and end time."""
    length = pd.DateOffset(years=23)
    dataset, _, _ = _mock_dataset(period_index_df, None)
    start_end = Composer._get_start_end([dataset], length)

    assert start_end is not None
    start_time, end_time = start_end
    assert start_time == pd.Timestamp("2000")
    assert end_time == pd.Timestamp("2023")


def test_get_start_end_single_too_long(period_index_df: pd.DataFrame) -> None:
    """Return start and end time."""
    length = pd.DateOffset(years=24)
    dataset, _, _ = _mock_dataset(period_index_df, None)
    start_end = Composer._get_start_end([dataset], length)
    assert start_end is None


def test_get_start_end_multiple(period_index_df: pd.DataFrame) -> None:
    """Return start and end time."""
    length = pd.DateOffset(years=1)
    dataset, _, _ = _mock_dataset(period_index_df, None)

    shorter_df = period_index_df.drop(index=period_index_df.index[:3])
    shorter_dataset, _, _ = _mock_dataset(shorter_df, None)

    start_end = Composer._get_start_end([dataset, shorter_dataset], length)

    assert start_end is not None
    start_time, end_time = start_end
    assert start_time == pd.Timestamp("2003")
    assert end_time == pd.Timestamp("2004")


def test_get_start_end_multiple_too_long(period_index_df: pd.DataFrame) -> None:
    """Return start and end time."""
    length = pd.DateOffset(years=21)
    dataset, _, _ = _mock_dataset(period_index_df, None)
    shorter_df = period_index_df.drop(index=period_index_df.index[:3])
    shorter_dataset, _, _ = _mock_dataset(shorter_df, None)
    start_end = Composer._get_start_end([dataset, shorter_dataset], length)
    assert start_end is None


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


def test_validate_start_end_true() -> None:
    """Do not error on correct times."""
    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    Composer._validate_start_end(start_time, end_time)


def test_validate_start_end_equal() -> None:
    """Do not error on equal times."""
    start_time = pd.Timestamp("2019")
    Composer._validate_start_end(start_time, start_time)


def test_validate_start_end_false() -> None:
    """Error on incorrect times."""
    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    with pytest.raises(ValueError) as e:
        Composer._validate_start_end(end_time, start_time)
    assert "must be earlier than" in str(e)


def test_validate_params(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Call validation functions."""
    length = pd.DateOffset(years=3)
    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(simple_df, metadata)

    mocker.patch("coin_test.data.Composer._validate_missing_data")
    mocker.patch("coin_test.data.Composer._get_start_end")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    Composer._get_start_end.return_value = (start_time, end_time)
    Composer._get_shared_currency.return_value = metadata.pair.currency

    start, end, ticker = Composer._validate_params([dataset], length)

    Composer._validate_missing_data.assert_called_once_with(dataset)
    Composer._get_start_end.assert_called_once_with([dataset], length)
    Composer._get_shared_currency.assert_called_once_with([dataset])
    assert start == start_time
    assert end == end_time
    assert ticker == metadata.pair.currency


def test_validate_params_no_datasets() -> None:
    """Error on no datasets passed."""
    length = pd.DateOffset(years=3)

    with pytest.raises(ValueError) as e:
        Composer._validate_params([], length)
    assert "At least one dataset must be defined." in str(e)


def test_validate_params_not_start_end(mocker: MockerFixture) -> None:
    """Error on dataset not sharing currency."""
    length = pd.DateOffset(years=3)

    mocker.patch("coin_test.data.Composer._validate_missing_data")
    mocker.patch("coin_test.data.Composer._get_start_end")
    Composer._get_start_end.return_value = None

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], length)
    assert "Overlapping time for datasets" in str(e)


def test_validate_params_not_shared_currency(mocker: MockerFixture) -> None:
    """Error on dataset not sharing currency."""
    length = pd.DateOffset(years=3)

    mocker.patch("coin_test.data.Composer._validate_missing_data")
    mocker.patch("coin_test.data.Composer._get_start_end")
    mocker.patch("coin_test.data.Composer._get_shared_currency")
    Composer._get_start_end.return_value = Mock(), Mock()
    Composer._get_shared_currency.return_value = None

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], length)
    assert "Not all datasets share a single currency." in str(e)


def test_validate_params_missing_data(mocker: MockerFixture) -> None:
    """Error on dataset missing data."""
    length = pd.DateOffset(years=3)

    mocker.patch("coin_test.data.Composer._validate_missing_data")
    Composer._validate_missing_data.return_value = False

    with pytest.raises(ValueError) as e:
        Composer._validate_params([Mock()], length)
    assert "does not have data for every period." in str(e)


def test_composer_init(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initialize correctly."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")
    length = pd.DateOffset(years=1)

    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, df_mock, metadata_mock = _mock_dataset(simple_df, metadata)
    _patch_composer_val(start_time, end_time, metadata.pair.currency, mocker)

    composer = Composer([dataset], length)

    pd.testing.assert_frame_equal(composer.datasets[metadata.pair].df, simple_df)

    df_mock.assert_called()
    metadata_mock.assert_called()
    Composer._validate_params.assert_called_once_with([dataset], length)
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
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
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
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)

    start_time = pd.Timestamp("2019-12-30")
    end_time = pd.Timestamp("2022-1-15")
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
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2019")
    data = composer.get_range(start_time, end_time, mask=False)

    df = data[metadata.pair]
    assert len(df) == 1


def test_composer_get_range_masked(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Mask final timestep of data."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
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
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)

    start_time = pd.Timestamp("2019")
    end_time = pd.Timestamp("2022")
    pair = AssetPair(Ticker("Test"), Ticker("Test"))
    data = composer.get_range(start_time, end_time, keys=[pair])
    assert len(data) == 0


def test_composer_get_timestep(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Calls get_range correctly."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "H")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)
    composer.get_range = Mock()

    timestep = pd.Timestamp("2019")
    composer.get_timestep(timestep, mask=True)
    composer.get_range.assert_called_once_with(timestep, timestep, keys=None, mask=True)


def test_composer_get_lookback(
    period_index_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Calls get_range correctly."""
    metadata = MetaData(AssetPair(Ticker("BTC"), Ticker("USDT")), "Y")
    dataset, _, _ = _mock_dataset(period_index_df, metadata)
    _patch_composer_val(
        pd.Timestamp("2008"), pd.Timestamp("2022"), metadata.pair.currency, mocker
    )

    length = pd.DateOffset(years=14)
    composer = Composer([dataset], length)
    composer.get_range = Mock()
    composer.freq = cast(pd.DateOffset, to_offset("Y"))

    timestep = pd.Timestamp("2019")
    composer.get_lookback(timestep, 8, mask=True)
    composer.get_range.assert_called_once_with(
        pd.Timestamp("2011-12-31"), timestep, keys=None, mask=True
    )
