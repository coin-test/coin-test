"""Test the Processors class."""

import pandas as pd

from coin_test.data import FillProcessor, IdentityProcessor


def test_identity_processor(simple_df: pd.DataFrame) -> None:
    """Do not transform dataframe."""
    processor = IdentityProcessor()
    new_df = processor(simple_df.copy())
    pd.testing.assert_frame_equal(simple_df, new_df)


def test_fill_processor(hour_data_indexed_df: pd.DataFrame) -> None:
    """Fills missing periods."""
    processor = FillProcessor("H")
    hour_data_indexed_df.drop(hour_data_indexed_df.index[3], inplace=True)

    old_len = len(hour_data_indexed_df)
    new_df = processor(hour_data_indexed_df)
    assert len(new_df) == old_len + 1
