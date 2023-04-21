"""Test the Binance data downloader."""

import datetime as dt
import logging
from typing import Generator

from freezegun import freeze_time
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pytest_mock import MockerFixture

from coin_test.data import BinanceDataset
from coin_test.data.metadata import MetaData
from coin_test.util import AssetPair, Ticker


def test_create_binance_dataset(mocker: MockerFixture) -> None:
    """Create a Binance dataset without errors."""
    name = "BTC-USDT Dataset 1"
    asset_pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "d"
    start = dt.datetime(year=2023, month=1, day=1)
    end = dt.datetime(year=2023, month=2, day=1)

    correct_df = pd.read_csv(
        "tests/data/assets/binance.csv",
        index_col="Open Time",
        parse_dates=["Open Time"],
    )
    index = pd.PeriodIndex(
        data=correct_df.index,
        freq="d",  # type: ignore
    )
    correct_df.set_index(index, inplace=True)

    def local_csvs() -> Generator[str, None, None]:
        for path in (
            "tests/data/assets/BTCUSDT-1d-2023-02.zip",
            "tests/data/assets/BTCUSDT-1d-2023-01.zip",
        ):  # data in reverse order on purpose
            yield path

    mocker.patch("coin_test.data.BinanceDataset._get_download_urls")
    BinanceDataset._get_download_urls.return_value = local_csvs()

    bd = BinanceDataset(name, asset_pair, freq, start, end)

    assert bd.name == name
    assert bd.metadata == MetaData(asset_pair, freq)
    assert_frame_equal(bd.df, correct_df, check_like=True)

    # with open("tests/data/assets/binance.csv", "w") as outfile:
    #     bd.df.to_csv(outfile)


def test_bad_url_binance_dataset(
    caplog: pytest.LogCaptureFixture, mocker: MockerFixture
) -> None:
    """Error on creating a Binance dataset with bad urls."""
    caplog.set_level(logging.DEBUG)

    name = "BTC-USDT Dataset 1"
    asset_pair = AssetPair(Ticker("BTC"), Ticker("USDT"))

    def bad_urls() -> Generator[str, None, None]:
        for bad_url in (
            "https://example.com/bad_url.zip",
            "https://example.com/badder_url.zip",
            "https://example.com/baddest_url.zip",
        ):
            yield bad_url

    mocker.patch("coin_test.data.BinanceDataset._get_download_urls")
    BinanceDataset._get_download_urls.return_value = bad_urls()

    with pytest.raises(ValueError):
        BinanceDataset(name, asset_pair)

    assert (
        "coin_test.data.binance_data_download",
        10,
        "Three endpoint requests failed",
    ) in caplog.record_tuples

    # Three debug logs for failed requests
    assert (
        len(
            tuple(
                filter(lambda msg: ("Failed request" in msg[2]), caplog.record_tuples)
            )
        )
        == 3
    )


def test_get_download_urls(mocker: MockerFixture) -> None:
    """Correctly get url endpoints."""
    asset_pair = AssetPair(Ticker("BTC"), Ticker("USDT"))
    freq = "h"
    alt_freq = "1h"

    def fake_dates() -> Generator[str, None, None]:
        for date in ("1900-01", "1900-02"):
            yield date

    mocker.patch("coin_test.data.BinanceDataset._get_date_ranges")

    BinanceDataset._get_date_ranges.return_value = fake_dates()
    urls = tuple(BinanceDataset._get_download_urls(asset_pair, freq))

    BinanceDataset._get_date_ranges.return_value = fake_dates()
    alt_urls = tuple(BinanceDataset._get_download_urls(asset_pair, alt_freq))

    assert urls == (
        "https://data.binance.vision/data/spot/monthly/"
        "klines/BTCUSDT/1h/BTCUSDT-1h-1900-01.zip",
        "https://data.binance.vision/data/spot/monthly/"
        "klines/BTCUSDT/1h/BTCUSDT-1h-1900-02.zip",
    )

    assert alt_urls == (
        "https://data.binance.vision/data/spot/monthly/"
        "klines/BTCUSDT/1h/BTCUSDT-1h-1900-01.zip",
        "https://data.binance.vision/data/spot/monthly/"
        "klines/BTCUSDT/1h/BTCUSDT-1h-1900-02.zip",
    )


def test_get_date_ranges() -> None:
    """Correctly get date ranges for querying the endpoint."""
    start_date = dt.datetime(year=2021, month=11, day=1)
    end_date = dt.datetime(year=2023, month=1, day=1)

    with freeze_time("2023-03-01"):
        start_never_end_now = BinanceDataset._get_date_ranges()
        start_never_end_set = BinanceDataset._get_date_ranges(end=end_date)
        start_set_end_now = tuple(BinanceDataset._get_date_ranges(start=start_date))
        start_set_end_set = tuple(
            BinanceDataset._get_date_ranges(start=start_date, end=end_date)
        )

        assert next(start_never_end_now) == "2023-03"
        assert next(start_never_end_now) == "2023-02"

        assert next(start_never_end_set) == "2023-01"
        assert next(start_never_end_set) == "2022-12"

        for _ in range(500):  # check it doesn't stop iterating
            assert next(start_never_end_now)
            assert next(start_never_end_set)

        assert len(start_set_end_now) == 17
        assert len(start_set_end_set) == 15

        assert start_set_end_now[-1] == "2021-11"
        assert start_set_end_set[-1] == "2021-11"


def test_asset_pair_to_name() -> None:
    """Correctly create an asset pair name for querying the endpoint."""
    assert (
        BinanceDataset._asset_pair_to_name(AssetPair(Ticker("BTC"), Ticker("USDT")))
        == "BTCUSDT"
    )
