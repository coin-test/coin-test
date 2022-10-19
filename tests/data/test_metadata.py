"""Test the MetaData class."""

from coin_test.data import MetaData


def test_metadata() -> None:
    """Initializes properly."""
    asset = "btc"
    currency = "usc"
    interval = 100
    data = MetaData(asset, currency, interval)
    assert data.asset == asset
    assert data.currency == currency
    assert data.interval == interval
