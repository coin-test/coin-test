"""Test the Loader classes."""

from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import CustomDataset, Dataset, PriceDataset


def test_dataset_construction_fails() -> None:
    """Raises error when creating base class."""
    with pytest.raises(ValueError):
        Dataset()


def test_validate_df_correct(hour_data_df: pd.DataFrame) -> None:
    """Validates a correctly formatted df."""
    assert PriceDataset._validate_df(hour_data_df)


def test_validate_df_missing_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df = hour_data_df.drop(columns=["Open"])
    assert not PriceDataset._validate_df(hour_data_df)


def test_validate_df_duplicate_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df.insert(0, "Open", hour_data_df["Open"], allow_duplicates=True)
    assert not PriceDataset._validate_df(hour_data_df)


def test_validate_df_incorrect_type(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df with an incorrect type."""
    hour_data_df["Open"] = hour_data_df["Open"].astype(int)
    assert not PriceDataset._validate_df(hour_data_df)


def test_add_period_index_existing_index(hour_data_df: pd.DataFrame) -> None:
    """Does nothing if period index exists."""
    years = ["20" + str(i) for i in range(10, len(hour_data_df) + 10)]
    index = pd.PeriodIndex(years, freq="Y")  # type: ignore
    hour_data_df.set_index(index, inplace=True)
    with_index = CustomDataset._add_period_index(hour_data_df, "Y")
    pd.testing.assert_frame_equal(hour_data_df, with_index)


def test_add_period_index_missing_col(simple_df: pd.DataFrame) -> None:
    """Errors on missing Open Time column."""
    with pytest.raises(ValueError):
        CustomDataset._add_period_index(simple_df, "Y")


def test_add_period_index_wrong_type(hour_data_df: pd.DataFrame) -> None:
    """Errors on missing Open Time column."""
    hour_data_df["Open Time"] = hour_data_df["Open Time"].astype(float)
    with pytest.raises(ValueError):
        CustomDataset._add_period_index(hour_data_df, "H")


_dates = [datetime(2000, 1, 1), datetime(2001, 1, 1), datetime(2002, 1, 1)]


@pytest.mark.parametrize(
    "data,freq",
    [
        (["2000", "2001", "2002"], "Y"),
        (_dates, "Y"),
        ([pd.Timestamp(date) for date in _dates], "Y"),
        ([pd.Period(date, freq="Y") for date in _dates], "Y"),  # type: ignore
        ([int(date.timestamp()) for date in _dates], "Y"),
    ],
)
def test_add_period_index(
    data: list[str | datetime | pd.Timestamp | pd.Period],
    freq: str,
    hour_data_df: pd.DataFrame,
) -> None:
    """Errors on missing Open Time column."""
    hour_data_df = hour_data_df[: len(data)].copy()
    print(pd.Series(data))
    hour_data_df["Open Time"] = pd.Series(data)
    CustomDataset._add_period_index(hour_data_df, freq)


def test_init_custom_dataset(hour_data_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initializes correctly."""
    asset = "BTC"
    currency = "USD"
    freq = "H"

    mocker.patch("coin_test.data.CustomDataset._add_period_index")
    mocker.patch("coin_test.data.CustomDataset._validate_df")

    CustomDataset._add_period_index.return_value = hour_data_df
    CustomDataset._validate_df.return_value = True

    dataset = CustomDataset(hour_data_df, freq, asset, currency)

    pd.testing.assert_frame_equal(dataset.df, hour_data_df)
    assert dataset.metadata.asset == asset
    assert dataset.metadata.currency == currency
    assert dataset.metadata.freq == freq

    CustomDataset._add_period_index.assert_called_once_with(hour_data_df, freq)
    CustomDataset._validate_df.assert_called_once_with(hour_data_df)


def test_init_custom_dataset_invalid_df(
    simple_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Errors on invalid df."""
    mocker.patch("coin_test.data.CustomDataset._add_period_index")
    mocker.patch("coin_test.data.CustomDataset._validate_df")
    CustomDataset._add_period_index.return_value = simple_df
    CustomDataset._validate_df.return_value = False

    with pytest.raises(ValueError):
        CustomDataset(simple_df, "H", "BTC", "USD")


def test_process(hour_data_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Calls processor properly."""
    asset = "BTC"
    currency = "USD"
    freq = "H"

    mocker.patch("coin_test.data.CustomDataset._add_period_index")
    mocker.patch("coin_test.data.CustomDataset._validate_df")
    processor = Mock()

    CustomDataset._add_period_index.return_value = hour_data_df
    CustomDataset._validate_df.return_value = True
    processor.return_value = hour_data_df

    dataset = CustomDataset(hour_data_df, freq, asset, currency)
    processed_dataset = dataset.process([processor])

    assert dataset == processed_dataset
    pd.testing.assert_frame_equal(processed_dataset.df, hour_data_df)
    processor.assert_called_once_with(hour_data_df)
