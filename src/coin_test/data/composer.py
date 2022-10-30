"""Define the Composer class."""

from typing import cast, Iterable

import numpy as np
import pandas as pd

from .datasets import PriceDataset
from ..util import AssetPair, Ticker


class Composer:
    """Manages datasets for simulation."""

    def __init__(
        self,
        datasets: list[PriceDataset],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> None:
        """Intialize a dataset.

        Args:
            datasets: List of PriceDatasets containing all price information for
                use in a simulation run.
            start_time: Expected start time to validate datasets against
            end_time: Expected end time to validate datasets against

        Raises:
            ValueError: If start and end time are invalid, if datasets do not
                align to start and end times or if datasets do not share a base
                currency.
        """
        self._validate_start_end(start_time, end_time)
        self.start_time = start_time
        self.end_time = end_time

        if len(datasets) == 0:
            raise ValueError("At least one dataset must be defined.")

        if not all(
            [self._is_within_range(ds, start_time, end_time) for ds in datasets]
        ):
            raise ValueError("Not all datasets cover requested time range")

        shared_currency = self._get_shared_currency(datasets)
        if shared_currency is None:
            raise ValueError("Not all datasets share a single currency.")
        self.currency = shared_currency

        self.freq = self._get_min_freq(datasets)
        self.datasets = {ds.metadata.pair: ds for ds in datasets}

    @staticmethod
    def _validate_start_end(start: pd.Timestamp, end: pd.Timestamp) -> None:
        if start >= end:
            raise ValueError("Start time must be earlier than end time.")

    @staticmethod
    def _is_within_range(
        dataset: PriceDataset, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> bool:
        """Check whether dataset has data for start and end time."""
        return not dataset.df[:start_time].empty and not dataset.df[end_time:].empty

    @staticmethod
    def _get_shared_currency(datasets: list[PriceDataset]) -> Ticker | None:
        """Get shared currency among datasets."""
        base_currency = datasets[0].metadata.pair.currency
        for dataset in datasets[1:]:
            if dataset.metadata.pair.currency != base_currency:
                return None
        return base_currency

    @staticmethod
    def _get_min_freq(datasets: list[PriceDataset]) -> pd.Timedelta:
        """Get minimium frequency among datasets."""

        def _to_timedelta(freq: str) -> pd.Timedelta:
            """Convert frequency string to timedelta object."""
            period_range = pd.period_range(start="2000", periods=2, freq=freq)
            start = cast(pd.Period, period_range[0])
            end = cast(pd.Period, period_range[1])
            return end.start_time - start.start_time

        return min([_to_timedelta(dataset.metadata.freq) for dataset in datasets])

    def get_timestep(
        self,
        timestamp: pd.Timestamp,
        keys: Iterable[AssetPair] | None = None,
        mask: bool = True,
    ) -> dict[AssetPair, pd.Series]:
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
        if keys is None:
            keys = list(self.datasets.keys())

        data = {}
        for key in keys:
            if key not in self.datasets:
                continue
            ds_data = self.datasets[key].df[timestamp]
            if mask:
                ds_data.loc[:, ds_data.columns != "Open"] = np.nan
            data[key] = ds_data
        return data

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
                ds_data.loc[-1, ds_data.columns != "Open"] = np.nan
            data[key] = ds_data
        return data

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
        start_time = timestamp - self.freq * lookback
        return self.get_range(start_time, end_time, keys=keys, mask=mask)
