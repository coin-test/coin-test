"""Test Processors."""

import pandas as pd

from coin_test.data import IdentityProcessor


def test_identity_processor() -> None:
    """Test IdentityProcessor does nothing."""
    column_name = "test_column"
    data = [1, 10, 100]

    df = pd.DataFrame(data=data, columns=[column_name])
    processor = IdentityProcessor()
    new_df = processor.process(df)

    pd.testing.assert_frame_equal(df, new_df)
