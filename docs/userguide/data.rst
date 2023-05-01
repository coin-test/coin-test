Data
====

Data is the backbone of Coin-Test.

Before running any kind of backtest, we must first prepare good data. And, when running a distributional analysis, we must be able to prepare many diverse datasets. Coin-Test enables this by allowing for easy downloading of historic cryptocurrency OHLCV data, as well as providing several statistical techniques for generating synthetic data.

Several convenience functions are also provided for cleaning, splitting and saving data. Data refers to a specific format of time series price data that a backtest uses to run. Internally this is a Pandas DataFrame with consistency checks and additional tooling. Inherently, Datasets are tied to the Strategies that are executed because a strategy is designed to run on a certain frequency or type of data.

.. contents:: Table of Contents
    :backlinks: none
    :local:


Historic Data
-------------

Automatic Downloader
^^^^^^^^^^^^^^^^^^^^

To automatically load historic data, we use the ``BinanceDataset``.
The ``BinanceDataset`` takes an asset pair, a frequency and a time range, then queries the Binance API for the desired data. By default, it will download all available data at the day frequency.

.. code-block:: python

    btc, usdt = btc_usdt = AssetPair.from_str("BTC", "USDT")
    dataset = BinanceDataset("BTC/USDT Daily Data", btc_usdt)

To specify a frequency or range of time, you may also pass the optional `freq`, `start` or `end` arguments:

.. code-block:: python

    dataset = BinanceDataset(
        "BTC/USDT Daily Data",
        btc_usdt,
        freq="mo",
        start=pd.Datetime("2021"),
        end=pd.Datetime("2022"),
    )


The ``freq`` parameter is a frequency string, any one of:

* "s" for second data
* "m" for minute data
* "h" for hour data
* "d" for daily data
* "w" for week data
* "mo" for month data

Manual Data Loading
^^^^^^^^^^^^^^^^^^^

To manually load historic data, we use the ``CustomDataset``.
The ``CustomDataset`` takes a pandas dataframe and a frequency, then processes it to fit the expected coin-test format. The frequency string is the `pandas frequency string format <https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects>`_.

The dataframe must meet the following conditions:

* Must have an "Open" column containing the timestamp or each row or must have a pandas ``PeriodIndex``. If an "Open" column is included, the timestamps must be in seconds, in a string format, or in any datetime type.
* Must have "High", "Low", "Close" and "Volume" columns, where each column has a dtype of ``float``.

Synthetic Data
--------------

Standard Usage
^^^^^^^^^^^^^^

To generate synthetic data, first initialize the generator with the desired seed dataset, then call ``generator.generate()``:

.. code-block:: python

    generator = GarchDatasetGenerator(train_dataset)
    datasets = generator.generate(timedelta=pd.Timedelta(days=25), n=50)

The ``generate()`` method takes three arguments regardless of the type of generator:

* ``timedelta``: The length each synthetic dataset should be.
* ``n``: Optional. The number of synthetic datasets to generate. Defaults to 1
* ``seed``:  Optional. The random seed to use as basis for the generation.

Available Generators
^^^^^^^^^^^^^^^^^^^^

Currently, several different generators are implemented:

* ``GarchDatasetGenerator`` utilizes a GARCH model to produce new synthetic data.
* ``WindowStepDatasetGenerator`` chunks the historic data into smaller pieces instead of generating new synthetic data.
* ``StitchedChunkDatasetGenerator`` randomly samples chunks of the historic data, and stiches them together into new synthetic data.
* ``ReturnsDatasetGenerator`` randomly samples returns from the historic data, and stiches them together into new synthetic data.

.. note::
    ``ReturnsDatasetGenerator`` is an extremely naïve method, and other generators should be favored.

Custom DatasetGenerator
^^^^^^^^^^^^^^^^^^^^^^^

New generators can be implemented as children of the ``DatasetGenerator`` class.
Children must implement the ``generate`` method with the ``timedelta``, ``seed`` and ``n`` arguments. Additionally, they must return a list of ``CustomDataset``. See the historic data section for more details on creating ``CustomDataset``.

Additional Features
-------------------

Cleaning
^^^^^^^^

To clean datasets, we can use ``Processor`` objects. Processors are passed in a list to the ``dataset.process()`` method:

.. code-block:: python

    processor = FillProcessor(freq)
    dataset.process([processor])

Currently coin-test only implements the ``FillProcessor``, which cleans NaN values out of datasets. Custom processors can be implemented by extending the ``Processor`` class. All children must implement a ``__call__`` method that is passed the dataframe to process.

Splitting
^^^^^^^^^

It is critical to create splits when optimizing a strategy, as failing to do so will cause you to overfit to your data and skew your evaluation results. To split datasets, we can use the ``dataset.split()`` method:

.. code-block:: python

    train, test = dataset.split(percent=0.75)

The ``split()`` method can split the dataset with different methods, selected by specifying one of the optional arguments:

* ``timestamp``: Split the dataset at the timestamp.
* ``length``: Split such that the train set has the specified length.
* ``percent``: Split such that the train set is the specified percent of the full dataset.

Saving / Loading
^^^^^^^^^^^^^^^^

Saving or loading data is important when sharing data, or avoiding costly operations involved in generating the data. For example, saving the results of a `BinanceDataset` download or the output of a large synthetic data generation set might save a large amount of time.

Arbitrary datasets can be saved with the ``Datasaver`` class:

.. code-block:: python

    datasaver = Datasaver([d1, d2, d3])
    datasaver.save("datasets")

Or loaded:

.. code-block:: python

    datasaver = Datasaver.load("datasets.pkl")
    datasets = datasaver.dataset_lists
