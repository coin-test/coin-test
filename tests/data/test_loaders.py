"""Test the Loader classes."""
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.data import DataFrameLoader, PriceDataFrameLoader, PriceDataLoader


def test_data_frame_loader(simple_df: pd.DataFrame) -> None:
    """Loads Dataframe."""
    loader = DataFrameLoader(simple_df)
    pd.testing.assert_frame_equal(simple_df, loader.df)


def test_validate_df_correct(hour_data_df: pd.DataFrame) -> None:
    """Validates a correctly formatted df."""
    assert PriceDataLoader._validate_df(hour_data_df)


def test_validate_df_missing_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df = hour_data_df.drop(columns=["Open"])
    assert not PriceDataLoader._validate_df(hour_data_df)


def test_validate_df_duplicate_col(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df missing a require column."""
    hour_data_df.insert(0, "Open", hour_data_df["Open"], allow_duplicates=True)
    assert not PriceDataLoader._validate_df(hour_data_df)


def test_validate_df_incorrect_type(hour_data_df: pd.DataFrame) -> None:
    """Rejects a df with an incorrect type."""
    hour_data_df["Open"] = hour_data_df["Open"].astype(int)
    assert not PriceDataLoader._validate_df(hour_data_df)


def test_infer_interval() -> None:
    """Inferrs the correct interval."""
    timestamps = pd.Series([10, 20, 30], dtype=int)
    interval = PriceDataFrameLoader._infer_interval(timestamps)
    assert interval == 10


def test_infer_interval_error() -> None:
    """Errors on inconsistent timestamps."""
    timestamps = pd.Series([10, 20, 40], dtype=int)
    with pytest.raises(ValueError):
        PriceDataFrameLoader._infer_interval(timestamps)


def test_init_price_dataframe_loader(
    hour_data_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Initializes correctly."""
    asset = "BTC"
    currency = "USD"
    interval = 100

    mocker.patch("coin_test.data.PriceDataFrameLoader._validate_df")
    mocker.patch("coin_test.data.PriceDataFrameLoader._infer_interval")

    PriceDataFrameLoader._validate_df.return_value = True
    PriceDataFrameLoader._infer_interval.return_value = interval

    loader = PriceDataFrameLoader(hour_data_df, asset, currency)

    pd.testing.assert_frame_equal(loader.df, hour_data_df)
    assert loader.metadata.asset == asset
    assert loader.metadata.currency == currency
    assert loader.metadata.interval == interval

    PriceDataFrameLoader._validate_df.assert_called_once_with(hour_data_df)
    PriceDataFrameLoader._infer_interval.assert_called_once_with(
        hour_data_df["Open Time"]
    )


def test_init_dataset_invalid_df(
    simple_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Errors on invalid df."""
    asset = "BTC"
    currency = "USD"

    mocker.patch("coin_test.data.PriceDataFrameLoader._validate_df")
    PriceDataFrameLoader._validate_df.return_value = False

    with pytest.raises(ValueError):
        PriceDataFrameLoader(simple_df, asset, currency)
