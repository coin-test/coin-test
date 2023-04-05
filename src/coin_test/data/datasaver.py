"""Define the Datasaver class."""
import logging
import os
import pickle

from .datasets import Dataset


logger = logging.getLogger(__name__)


class Datasaver:
    """Hold collections of datasets to save."""

    def __init__(
        self, name: str, datasets: list[Dataset] | list[list[Dataset]]
    ) -> None:
        """Initialize a Datasaver.

        Args:
            name (str): string name for datasaver
            datasets (list[Dataset] | list[list[Dataset]]): collection of
                datasets to save
        """
        self.name = "".join(x for x in name if x.isalnum() or x in ["_"])

        if isinstance(datasets[0], Dataset):
            self.datasets_lists = [[d] for d in datasets]
        else:
            self.datasets_lists = datasets

    def save(self, directory: str) -> str:
        """Pickle a Datasaver to disk.

        Args:
            directory (str): Filepath to save to

        Returns:
            str: _description_
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, self.name + ".pkl")
        with open(path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved pickled Datasaver to: {path}")
        return path

    @staticmethod
    def load(fp: str) -> "Datasaver":
        """Load Datasaver from disk.

        Args:
            fp (str): filepath to load from

        Returns:
            Datasaver: Datasaver stored at the location

        Raises:
            ValueError: raises ValueError if the specified file path is not a file
        """
        if not os.path.isfile(fp):
            raise ValueError(f"'{fp}' is not a file.")

        with open(fp, "rb") as f:
            obj = pickle.load(f)
            logger.debug(f"Loaded Datasaver from {fp}")
            return obj
