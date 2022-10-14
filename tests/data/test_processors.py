"""Test the Processors class."""

import pandas as pd

from coin_test.data import IdentityProcessor


def test_identity_processor(simple_df: pd.DataFrame) -> None:
    """Does not transform dataframe."""
    processor = IdentityProcessor()
    new_df = processor.process(simple_df.copy())
    pd.testing.assert_frame_equal(simple_df, new_df)
