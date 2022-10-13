"""Pytest config."""

import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest config."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


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
    """Hourly data CSV filepath."""
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
        "Close Time": int,
    }
    df = pd.read_csv(hour_data, dtype=dtypes)
    return df
