"""Define the Datasaver class."""
import os
import pickle

from .datasets import Dataset


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

    @staticmethod
    def save(datasaver: "Datasaver", directory: str) -> str:
        """Pickle a Datasaver to disk.

        Args:
            datasaver (DataSaver): DataSaver object to save
            directory (str): Filepath to save to

        Returns:
            str: _description_
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, datasaver.name + ".pkl")
        with open(path, "wb") as outp:
            pickle.dump(datasaver, outp, pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(fp: str) -> "Datasaver":
        """Load Datasaver from disk.

        Args:
            fp (str): filepath to load from

        Returns:
            Datasaver: Datasaver stored at the location
        """
        with open(fp, "rb") as f:
            return pickle.load(f)
