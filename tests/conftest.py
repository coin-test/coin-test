"""Pytest config."""


import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest config."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


@pytest.fixture
def simple_csv() -> str:
    """Simple CSV filepath."""
    return "tests/test_data/simple.csv"


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Simple CSV contents."""
    column_names = ["Time", "Value"]
    data = [[0, 10], [1, 100], [2, 1000]]
    return pd.DataFrame(data=data, columns=column_names)
