"""Define the Composer class."""

from typing import cast, Iterable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .datasets import PriceDataset
from ..util import AssetPair, Ticker


class Composer:
    """Manages datasets for simulation."""

    def __init__(
        self,
        datasets: list[PriceDataset],
        length: pd.Timedelta | pd.DateOffset,
    ) -> None:
        """Intialize a dataset.

        Args:
            datasets: List of PriceDatasets containing all price information for
                use in a simulation run.
            length: Length of time for simulation run.
        """
        start_time, end_time, shared_currency = self._validate_params(datasets, length)
        self.start_time = start_time
        self.end_time = end_time
        self.currency = shared_currency
        self.freq = self._get_min_freq(datasets)
        self.datasets = {ds.metadata.pair: ds for ds in datasets}

    @staticmethod
    def _validate_params(
        datasets: list[PriceDataset],
        length: pd.Timedelta | pd.DateOffset,
    ) -> tuple[pd.Timestamp, pd.Timestamp, Ticker]:
        """Validates input parameters.

        Args:
            datasets: List of PriceDatasets containing all price information for
                use in a simulation run.
            length: Length of time for simulation run.

        Returns:
            pd.Timestamp: Validated start time of simulation
            pd.Timestamp: Validted end time of simulation
            Ticker: Currency shared by all datasets

        Raises:
            ValueError: If datasets are not specified, if length does not fit
                datasets overlap, or if datasets don't share a currency.
        """
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be defined.")

        for ds in datasets:
            if not Composer._validate_missing_data(ds):
                raise ValueError(f"Dataset {ds} does not have data for every period.")

        start_end = Composer._get_start_end(datasets, length)
        if start_end is None:
            raise ValueError("Overlapping time for datasets less than requested length")
        start_time, end_time = start_end

        shared_currency = Composer._get_shared_currency(datasets)
        if shared_currency is None:
            raise ValueError("Not all datasets share a single currency.")

        return start_time, end_time, shared_currency

    @staticmethod
    def _validate_start_end(start: pd.Timestamp, end: pd.Timestamp) -> None:
        if start > end:
            raise ValueError("Start time must be earlier than end time.")

    @staticmethod
    def _validate_missing_data(dataset: PriceDataset) -> bool:
        index = dataset.df.index
        full_index = pd.period_range(
            index[0], index[-1], freq=dataset.metadata.freq  # type: ignore
        )
        missing_index = full_index.difference(dataset.df.index)
        return len(missing_index) == 0

    @staticmethod
    def _get_start_end(
        datasets: list[PriceDataset], length: pd.Timedelta | pd.DateOffset
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Get start and time for datasets."""
        start_times = [
            dataset.df.index[0].start_time for dataset in datasets  # type: ignore
        ]
        end_times = [
            dataset.df.index[-1].start_time for dataset in datasets  # type: ignore
        ]

        start_time = max(start_times)
        end_time = min(end_times)
        if start_time > end_time - length:
            return None

        return end_time - length, end_time

    @staticmethod
    def _get_shared_currency(datasets: list[PriceDataset]) -> Ticker | None:
        """Get shared currency among datasets."""
        base_currency = datasets[0].metadata.pair.currency
        for dataset in datasets[1:]:
            if dataset.metadata.pair.currency != base_currency:
                return None
        return base_currency

    @staticmethod
    def _get_min_freq(datasets: list[PriceDataset]) -> pd.DateOffset:
        """Get minimium frequency among datasets."""
        offsets = [
            cast(pd.DateOffset, to_offset(dataset.metadata.freq))
            for dataset in datasets
        ]
        min_offset = offsets[0]
        timestamp = pd.Timestamp("2001")
        for offset in offsets[1:]:
            # Direct comparison of offsets is disallowed
            if timestamp + offset < timestamp + min_offset:
                min_offset = offset
        return min_offset

    def get_range(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        keys: Iterable[AssetPair] | None = None,
        mask: bool = True,
    ) -> dict[AssetPair, pd.DataFrame]:
        """Get range of data.

        Args:
            start_time: Starting timestamp of range.
            end_time: Ending timestamp of range.
            keys: Optional. AssetPairs to filter results on.
            mask: Optional. Whether to mask non-Open data to NaN.

        Returns:
            dict: Dictionary mapping asset pairs to retrieved rows of data per
                asset pair. Dictionary default to all datasets, but is filtered
                based on `keys` parameter.
        """
        self._validate_start_end(start_time, end_time)
        if keys is None:
            keys = list(self.datasets.keys())

        data = {}
        for key in keys:
            if key not in self.datasets:
                continue
            ds_data = self.datasets[key].df[start_time:end_time]
            if mask:
                ds_data = ds_data.copy()
                ds_data.loc[ds_data.index[-1], ds_data.columns != "Open"] = np.nan
            data[key] = ds_data
        return data

    def get_timestep(
        self,
        timestamp: pd.Timestamp,
        keys: Iterable[AssetPair] | None = None,
        mask: bool = True,
    ) -> dict[AssetPair, pd.DataFrame]:
        """Get single timestep of data.

        Args:
            timestamp: Timestamp to get data for.
            keys: Optional. AssetPairs to filter results on.
            mask: Optional. Whether to mask non-Open data to NaN.

        Returns:
            dict: Dictionary mapping asset pairs to timestamp data per asset
                pair. Dictionary default to all datasets, but is filtered based
                on `keys` parameter.
        """
        return self.get_range(timestamp, timestamp, keys=keys, mask=mask)

    def get_lookback(
        self,
        timestamp: pd.Timestamp,
        lookback: int,
        keys: Iterable[AssetPair] | None = None,
        mask: bool = True,
    ) -> dict[AssetPair, pd.DataFrame]:
        """Get lookback of data.

        Wrapper for `get_range`. Convert integer number of lookback timesteps
        into a time range based of Composer's `freq` attribute.

        Args:
            timestamp: Starting timestamp of lookback.
            lookback: Number of timesteps to lookback.
            keys: Optional. AssetPairs to filter results on.
            mask: Optional. Whether to mask non-Open data to NaN.

        Returns:
            dict: Dictionary mapping asset pairs to retrieved rows of data per
                asset pair. Dictionary default to all datasets, but is filtered
                based on `keys` parameter.
        """
        end_time = timestamp
        start_time = pd.Timestamp(timestamp - self.freq * lookback)
        return self.get_range(start_time, end_time, keys=keys, mask=mask)
