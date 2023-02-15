"""Test the DataSaver class."""

import os
import pickle
from unittest.mock import Mock, mock_open, patch

from pytest_mock import MockerFixture

from coin_test.data import Datasaver, Dataset


def test_datasaver_valid_construction_list() -> None:
    """Construct Datasaver using a single dpeth list."""
    name = "test_datasaver"
    datasets = [Mock(spec=Dataset)]
    dsaver = Datasaver(name, datasets)  # pyright: ignore

    assert name == dsaver.name
    assert [datasets] == dsaver.datasets_lists


def test_datasaver_valid_construction_list_list() -> None:
    """Construct Datasaver using a single dpeth list."""
    name = "test_datasaver12"
    datasets = [[Mock(spec=Dataset), Mock(spec=Dataset)], [Mock(spec=Dataset)]]
    dsaver = Datasaver(name, datasets)  # pyright: ignore

    assert name == dsaver.name
    assert datasets == dsaver.datasets_lists


def test_datasaver_save(mocker: MockerFixture) -> None:
    """Pickle Datasaver object properly."""
    name = "test_datasaver12"
    datasets = [[Mock(spec=Dataset), Mock(spec=Dataset)], [Mock(spec=Dataset)]]
    dsaver = Datasaver(name, datasets)  # pyright: ignore
    valid_local_path = "a_valid_path"

    mocker.patch("pickle.dump")
    pickle.dump.return_value = None

    mocker.patch("os.path.exists")
    os.path.exists.return_value = True

    with patch("builtins.open", mock_open(read_data="data")):
        final_path = dsaver.save(valid_local_path)
        pickle.dump.assert_called()

    assert final_path == valid_local_path + "/" + dsaver.name + ".pkl"


def test_datasaver_save_new_dir(mocker: MockerFixture) -> None:
    """Pickle Datasaver object properly."""
    name = "test_datasaver12"
    datasets = [[Mock(spec=Dataset), Mock(spec=Dataset)], [Mock(spec=Dataset)]]
    dsaver = Datasaver(name, datasets)  # pyright: ignore

    valid_local_path = "a_valid_path"

    mocker.patch("pickle.dump")
    pickle.dump.return_value = None

    mocker.patch("os.path.exists")
    os.path.exists.return_value = False

    mocker.patch("os.makedirs")

    with patch("builtins.open", mock_open(read_data="data")):
        final_path = dsaver.save(valid_local_path)
        pickle.dump.assert_called()

    assert final_path == valid_local_path + "/" + dsaver.name + ".pkl"


def test_datasaver_load(mocker: MockerFixture) -> None:
    """Load Pickled Datasaver object properly."""
    datasaver = Mock()
    valid_local_path = "a_valid_path/test.pkl"

    mocker.patch("pickle.load")
    pickle.load.return_value = datasaver

    with patch("builtins.open", mock_open(read_data="data")):
        test = Datasaver.load(valid_local_path)
        pickle.load.assert_called()
        assert test == datasaver
