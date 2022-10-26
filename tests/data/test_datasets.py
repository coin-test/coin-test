"""Test the Loader classes."""

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


def test_infer_interval() -> None:
    """Inferrs the correct interval."""
    timestamps = pd.Series([10, 20, 30], dtype=int)
    interval = CustomDataset._infer_interval(timestamps)
    assert interval == 10


def test_infer_interval_error() -> None:
    """Errors on inconsistent timestamps."""
    timestamps = pd.Series([10, 20, 40], dtype=int)
    with pytest.raises(ValueError):
        CustomDataset._infer_interval(timestamps)


def test_init_price_dataframe_loader(
    hour_data_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Initializes correctly."""
    asset = "BTC"
    currency = "USD"
    interval = 100

    mocker.patch("coin_test.data.CustomDataset._validate_df")
    mocker.patch("coin_test.data.CustomDataset._infer_interval")

    CustomDataset._validate_df.return_value = True
    CustomDataset._infer_interval.return_value = interval

    dataset = CustomDataset(hour_data_df, asset, currency)

    pd.testing.assert_frame_equal(dataset.df, hour_data_df)
    assert dataset.metadata.asset == asset
    assert dataset.metadata.currency == currency
    assert dataset.metadata.interval == interval

    CustomDataset._validate_df.assert_called_once_with(hour_data_df)
    CustomDataset._infer_interval.assert_called_once_with(hour_data_df["Open Time"])


def test_init_dataset_invalid_df(
    simple_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Errors on invalid df."""
    asset = "BTC"
    currency = "USD"

    mocker.patch("coin_test.data.CustomDataset._validate_df")
    CustomDataset._validate_df.return_value = False

    with pytest.raises(ValueError):
        CustomDataset(simple_df, asset, currency)


def test_process(hour_data_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Calls processor properly."""
    asset = "BTC"
    currency = "USD"
    interval = 100

    mocker.patch("coin_test.data.CustomDataset._validate_df")
    mocker.patch("coin_test.data.CustomDataset._infer_interval")
    processor = Mock()

    CustomDataset._validate_df.return_value = True
    CustomDataset._infer_interval.return_value = interval
    processor.process.return_value = hour_data_df

    dataset = CustomDataset(hour_data_df, asset, currency)
    processed_dataset = dataset.process([processor])

    assert dataset == processed_dataset
    pd.testing.assert_frame_equal(processed_dataset.df, hour_data_df)
    processor.process.assert_called_once_with(hour_data_df)
