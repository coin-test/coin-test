"""Test Processors."""

from unittest.mock import Mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import Dataset


def test_validate_df_correct(hour_data_df: pd.DataFrame) -> None:
    """Validates a correctly formatted df."""
    assert Dataset.validate_df(hour_data_df)


def test_validate_df_missing_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df = hour_data_df.drop(columns=["Open"])
    assert not Dataset.validate_df(hour_data_df)


def test_validate_df_duplicate_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df.insert(0, "Open", hour_data_df["Open"], allow_duplicates=True)
    assert not Dataset.validate_df(hour_data_df)


def test_validate_df_incorrect_type(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df with an incorrect type."""
    hour_data_df["Open"] = hour_data_df["Open"].astype(int)
    assert not Dataset.validate_df(hour_data_df)


def test_infer_interval() -> None:
    """Inferrs the correct interval."""
    timestamps = pd.Series([10, 20, 30], dtype=int)
    interval = Dataset.infer_interval(timestamps)
    assert interval == 10


def test_infer_interval_error() -> None:
    """Errors on inconsistent timestamps."""
    timestamps = pd.Series([10, 20, 40], dtype=int)
    with pytest.raises(ValueError):
        Dataset.infer_interval(timestamps)


def test_clean(simple_df: pd.DataFrame) -> None:
    """Runs the processor."""
    processor = Mock()
    processor.process.return_value = simple_df
    df = Dataset._clean(simple_df, [processor])

    pd.testing.assert_frame_equal(df, simple_df)
    processor.process.assert_called_once_with(simple_df)


def test_init_dataset(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initializes correctly."""
    asset = "BTC"
    currency = "USD"
    interval = 10000
    processors = []

    mocker.patch("coin_test.data.Dataset.validate_df")
    mocker.patch("coin_test.data.Dataset.infer_interval")
    mocker.patch("coin_test.data.Dataset._clean")

    Dataset.validate_df.return_value = True
    Dataset._clean.return_value = simple_df

    dataset = Dataset(asset, currency, processors, simple_df, interval)

    assert dataset.asset == asset
    assert dataset.currency == currency
    assert dataset.interval == interval

    Dataset.validate_df.assert_called_once_with(simple_df)
    Dataset.infer_interval.assert_not_called()
    Dataset._clean.assert_called_once_with(simple_df, processors)


def test_init_dataset_infers(hour_data_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Infers interval when not provided."""
    asset = "BTC"
    currency = "USD"
    interval = 10
    processors = []

    mocker.patch("coin_test.data.Dataset.validate_df")
    mocker.patch("coin_test.data.Dataset.infer_interval")
    mocker.patch("coin_test.data.Dataset._clean")

    Dataset.validate_df.return_value = True
    Dataset.infer_interval.return_value = interval
    Dataset._clean.return_value = hour_data_df

    dataset = Dataset(asset, currency, processors, hour_data_df)

    assert dataset.interval == interval
    Dataset.infer_interval.assert_called_once_with(hour_data_df["Open Time"])


def test_init_dataset_invalid_df(
    simple_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Errors on invalid df."""
    asset = "BTC"
    currency = "USD"
    processors = []

    mocker.patch("coin_test.data.Dataset.validate_df")
    Dataset.validate_df.return_value = False

    with pytest.raises(ValueError):
        Dataset(asset, currency, processors, simple_df)
