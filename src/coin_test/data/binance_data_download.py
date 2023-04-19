"""Binance Data Download Class."""

import datetime as dt
import itertools
import logging
from typing import Generator
from urllib.error import HTTPError

import pandas as pd
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .datasets import PriceDataset
from .metadata import MetaData
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
    BASE_URL = "https://data.binance.vision/data/"

    def __init__(
        self,
        name: str,
        asset_pair: AssetPair,
        freq: str = "d",
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> None:
        """Download historical cryptocurrency datasets from Binance.

        Args:
            name: The name of the dataset
            asset_pair: The asset pair to trade on
            freq: The frequency of the data, default daily
                    's' for second data
                    'm' for minute data
                    'h' for hourly data
                    'd' for daily data
                    'w' for week data
                    'mo' for month data
            start: The start date of the dataset. If no start date is provided,
                    download all historical data
            end: The end date of the dataset. If no end date is provided, download
                    until the present day

        Raises:
            ValueError: If the generated URL does not successfully access the
                    Binance dataset
        """
        build_df = pd.DataFrame()
        self._metadata = MetaData(pair=asset_pair, freq=freq)
        self.name = name

        num_failed_requests = 0
        num_successful_requests = 0

        logger.info("Downloading data from Binance")
        with logging_redirect_tqdm():
            for url in tqdm.tqdm(
                BinanceDataset._get_download_urls(asset_pair, freq, start, end)
            ):
                logger.debug(f"Downloading {url}")

                try:
                    temp_csv = pd.read_csv(
                        url,
                        compression="zip",
                        header=None,
                        names=["Open Time", "Open", "High", "Low", "Close", "Volume"],
                        usecols=[0, 1, 2, 3, 4, 5],
                    )
                    build_df = pd.concat([temp_csv, build_df])
                    num_successful_requests += 1

                except HTTPError:
                    num_failed_requests += 1
                    logger.debug(f"Failed request to {url}")

                if num_failed_requests >= 3:
                    logger.debug("Three endpoint requests failed")
                    break

        if build_df.empty:
            raise ValueError(
                "Failed to download Binance data with the given arguments."
            )

        build_df["Open Time"] //= 1000

        self.df = self._add_period_index(build_df, freq)
        logger.info(f"Successfully downloaded {num_successful_requests} datasets")
        logger.debug(f"Downloaded data:\n{self.df.head()}\n...\n{self.df.tail()}")

    @property
    def metadata(self) -> MetaData:
        """Get metadata."""
        return self._metadata

    @staticmethod
    def _get_download_urls(
        asset_pair: AssetPair,
        freq: str,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> Generator[str, None, None]:
        """Get the download urls."""
        exchange_name = BinanceDataset._asset_pair_to_name(asset_pair)

        if not any(char.isdigit() for char in freq):
            freq = "1" + freq
        freq = freq.lower()

        for month in BinanceDataset._get_date_ranges(start, end):
            yield (
                f"https://data.binance.vision/data/spot/monthly/"
                f"klines/{exchange_name}/{freq}/"
                f"{exchange_name}-{freq}-{month}.zip"
            )

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
