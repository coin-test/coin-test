"""Define the Dataset class."""

import pandas as pd

from .datasets import PriceDataset


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
        if start_time >= end_time:
            raise ValueError("Start time must be earlier than end time.")
        print(start_time, end_time)
        self.start_time = start_time
        self.end_time = end_time

        if not all(
            [self._is_within_range(ds, start_time, end_time) for ds in datasets]
        ):
            raise ValueError("Not all datasets cover requested time range")

        self.datasets = {ds.metadata: ds for ds in datasets}

    @staticmethod
    def _is_within_range(
        dataset: PriceDataset, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> bool:
        return not dataset.df[:start_time].empty and not dataset.df[end_time:].empty
