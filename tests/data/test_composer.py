"""Test the Dataset class."""

from unittest.mock import Mock, PropertyMock

import pandas as pd
from pytest_mock import MockerFixture

from coin_test.data import Composer, MetaData


def test_clean(simple_df: pd.DataFrame) -> None:
    """Runs the processor."""
    processor = Mock()
    processor.process.return_value = simple_df
    df = Composer._clean(simple_df, [processor])

    pd.testing.assert_frame_equal(df, simple_df)
    processor.process.assert_called_once_with(simple_df)


def test_init_composer(simple_df: pd.DataFrame, mocker: MockerFixture) -> None:
    """Initializes correctly."""
    processors = []
    metadata = MetaData("BTC", "USD", 100)
    loader = Mock()
    df_mock = PropertyMock(return_value=simple_df)
    metadata_mock = PropertyMock(return_value=metadata)
    mocker.patch("coin_test.data.Composer._clean")

    type(loader).df = df_mock
    type(loader).metadata = metadata_mock
    Composer._clean.return_value = simple_df

    composer = Composer(loader, processors)

    pd.testing.assert_frame_equal(composer.df, simple_df)
    assert composer.metadata == metadata

    df_mock.assert_called_once_with()
    metadata_mock.assert_called_once_with()
    Composer._clean.assert_called_once_with(simple_df, processors)
