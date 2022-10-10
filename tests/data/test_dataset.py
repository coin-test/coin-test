"""Test Processors."""

from unittest.mock import Mock

import pandas as pd
from pytest_mock import MockerFixture

from coin_test.data import Dataset


def test_load_df(simple_csv: str, simple_df: pd.DataFrame) -> None:
    """Test _load_df can load a simple CSV."""
    df = Dataset._load_df(simple_csv)
    pd.testing.assert_frame_equal(df, simple_df)


def test_clean(simple_df: pd.DataFrame) -> None:
    """Test _clean runs the processor."""
    processor = Mock()
    processor.process.return_value = simple_df
    df = Dataset._clean(simple_df, [processor])

    pd.testing.assert_frame_equal(df, simple_df)
    processor.process.assert_called_once_with(simple_df)


def test_init_dataset(
    simple_csv: str, simple_df: pd.DataFrame, mocker: MockerFixture
) -> None:
    """Test initialization succeeds."""
    asset = "BTC"
    currency = "USD"
    processors = []

    mocker.patch("coin_test.data.Dataset._load_df")
    mocker.patch("coin_test.data.Dataset._clean")
    Dataset._load_df.return_value = simple_df

    dataset = Dataset(simple_csv, asset, currency, processors)

    assert dataset.asset == asset
    assert dataset.currency == currency
    assert dataset.interval is None  # Currently un-implemented

    dataset._load_df.assert_called_once_with(simple_csv)
    dataset._clean.assert_called_once_with(simple_df, processors)
