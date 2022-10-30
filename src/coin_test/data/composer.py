"""Define the Dataset class."""

import pandas as pd

from .datasets import PriceDataset


class Composer:
    """Manages a dataset."""

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
        """
        self.datasets = {ds.metadata: ds for ds in datasets}
        self.start_time = start_time
        self.end_time = end_time
