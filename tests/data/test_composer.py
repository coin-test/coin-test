"""Test the Dataset class."""

from unittest.mock import Mock, PropertyMock

import pandas as pd

from coin_test.data import Composer, MetaData


def test_init_composer(simple_df: pd.DataFrame) -> None:
    """Initializes correctly."""
    metadata = MetaData("BTC", "USD", "H")
    start_time = pd.Timestamp("2020")
    end_time = pd.Timestamp("2021")

    dataset = Mock()
    df_mock = PropertyMock(return_value=simple_df)
    metadata_mock = PropertyMock(return_value=metadata)

    type(dataset).df = df_mock
    type(dataset).metadata = metadata_mock

    composer = Composer([dataset], start_time, end_time)

    pd.testing.assert_frame_equal(composer.datasets[metadata].df, simple_df)

    df_mock.assert_called_once_with()
    metadata_mock.assert_called_once_with()
