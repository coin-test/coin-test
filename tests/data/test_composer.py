"""Test the Dataset class."""

from unittest.mock import Mock, PropertyMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import Composer, MetaData


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


def test_composer_init(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initializes correctly."""
    metadata = MetaData("BTC", "USD", "H")
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    dataset = Mock()
    df_mock = PropertyMock(return_value=simple_df)
    metadata_mock = PropertyMock(return_value=metadata)
    mocker.patch("coin_test.data.Composer._is_within_range")

    type(dataset).df = df_mock
    type(dataset).metadata = metadata_mock
    Composer._is_within_range.return_value = True

    composer = Composer([dataset], start_time, end_time)

    pd.testing.assert_frame_equal(composer.datasets[metadata].df, simple_df)

    df_mock.assert_called_once_with()
    metadata_mock.assert_called_once_with()
    Composer._is_within_range.assert_called_once_with(dataset, start_time, end_time)


def test_composer_invalid_range() -> None:
    """Errors on invalid time range."""
    start_time = pd.Timestamp("2021")
    end_time = pd.Timestamp("2020")

    with pytest.raises(ValueError) as e:
        Composer([Mock()], start_time, end_time)
        assert "earlier than end time" in str(e)


def test_composer_not_within_range(mocker: MockerFixture) -> None:
    """Errors on dataset not within range."""
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    mocker.patch("coin_test.data.Composer._is_within_range")
    Composer._is_within_range.return_value = False

    with pytest.raises(ValueError) as e:
        Composer([Mock()], start_time, end_time)
        assert "Not all datasets cover requested time range" in str(e)
