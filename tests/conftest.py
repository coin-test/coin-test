"""Pytest config."""

import pandas as pd
import pytest


@pytest.fixture
def simple_csv() -> str:
    """Simple CSV filepath."""
    return "tests/assets/simple.csv"


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Simple CSV contents."""
    column_names = ["Time", "Value"]
    data = [[0, 10], [1, 100], [2, 1000]]
    return pd.DataFrame(data=data, columns=column_names)


@pytest.fixture
def hour_data() -> str:
    """Hourly data CSV filepath Start: 2022-09-28 00:00 End: 2022-09-28 23:00."""
    return "tests/assets/eth_usdc_1h_9_28.csv"


@pytest.fixture
def hour_data_df(hour_data: str) -> pd.DataFrame:
    """Hourly data contents."""
    dtypes = {
        "Open Time": int,
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
    }
    df = pd.read_csv(hour_data, dtype=dtypes)
    return df


@pytest.fixture
def hour_data_indexed_df(hour_data: str) -> pd.DataFrame:
    """Hourly data contents with period index."""
    dtypes = {
        "Open Time": int,
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": float,
    }
    df = pd.read_csv(hour_data, dtype=dtypes)  # type: ignore
    index = pd.PeriodIndex(
        data=pd.to_datetime(df["Open Time"], unit="s", utc=True),
        freq="H",  # type: ignore
    )
    df.set_index(index, inplace=True)
    df.drop(columns=["Open Time"], inplace=True)
    return df
