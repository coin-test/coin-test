"""Test the Loader classes."""

import pandas as pd

from coin_test.data import DataFrameLoader


def test_identity_processor(simple_df: pd.DataFrame) -> None:
    """Loads Dataframe."""
    loader = DataFrameLoader(simple_df)
    pd.testing.assert_frame_equal(simple_df, loader.df)
