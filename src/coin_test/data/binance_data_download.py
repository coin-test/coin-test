"""Class for downloading data from Binance."""

import datetime as dt
import itertools
import logging
from typing import Generator
from urllib.error import HTTPError

import pandas as pd

from .datasets import PriceDataset
from ..util import AssetPair

logger = logging.getLogger(__name__)


class BinanceDataset(PriceDataset):
    """Create datasets from downloaded Binance data."""

    INTERVALS = (
        "1s",
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1mo",
    )
    DAILY_INTERVALS = (
        "1s",
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    )
    MONTHS = tuple(range(1, 13))
    PERIOD_START_DATE = "2020-01-01"
    BASE_URL = "https://data.binance.vision/data/"
    START_DATE = dt.date(year=2017, month=1, day=1)
    END_DATE = dt.datetime.now()

    def __init__(
        self, asset_pair: AssetPair, start: dt.datetime, end: dt.datetime, freq: str
    ) -> None:
        """Download a Binance dataset."""
        self.df = None

        num_failed_requests = 0
        num_successful_requests = 0

        for url in BinanceDataset._get_download_urls(asset_pair, start, end, freq):
            try:
                temp_csv = pd.read_csv(url, compression="zip", header=None, names=[])
                if self.df is not None:
                    self.df = pd.concat([temp_csv, self.df])
                else:
                    self.df = temp_csv
                num_successful_requests += 1
            except HTTPError:
                num_failed_requests += 1
                logger.debug(f"Failed request to {url}")
            except Exception as e:
                raise ValueError(f"Error downloading data from {url}") from e

    @staticmethod
    def _get_download_urls(
        asset_pair: AssetPair,
        start: dt.datetime,
        end: dt.datetime,
        freq: str,
    ) -> list[str]:
        """Get the download urls."""
        exchange_name = BinanceDataset._asset_pair_to_name(asset_pair)

        if not any(char.isdigit() for char in freq):
            freq = "1" + freq

        urls = [
            (
                f"https://data.binance.vision/data/spot/monthly/"
                f"/klines/{exchange_name}/{freq}/"
                f"{exchange_name}-{freq}-{month}.zip"
            )
            for month in BinanceDataset._get_date_ranges(start, end)
        ]

        return urls

    @staticmethod
    def _get_date_ranges(
        start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> Generator[str, None, None]:
        """Get month ranges for downloading data."""
        if end is None:
            end = dt.datetime.now()

        end_year = end.year
        end_month = end.month
        end_month_num = end_year * 12 + end_month - 1

        if start is None:
            # count backwards forever
            iter_range = itertools.count(end_month_num, -1)
        else:
            # count backwards until start
            start_year = start.year
            start_month = start.month
            start_month_num = start_year * 12 + start_month - 1
            iter_range = range(end_month_num, start_month_num - 1, -1)

        for month in iter_range:
            yield f"{month//12}-{month%12+1:02d}"

    @staticmethod
    def _asset_pair_to_name(asset_pair: AssetPair) -> str:
        """Convert an asset pair to a string for trade lookup."""
        asset_symbol = str(asset_pair.asset.symbol).upper()
        currency_symbol = str(asset_pair.currency.symbol).upper()
        return asset_symbol + currency_symbol
