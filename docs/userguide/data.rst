Data
====

Data is the backbone of Coin-Test.

Data refers to a specific format of time series price data that a backtest uses to run. Internally this is a Pandas DataFrame with consistency checks and additional tooling. Inherently, Datasets are tied to the :doc:`strategies` that are executed because a strategy is designed to run on a certain frequency or type of data.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Data loading
------------

Automatic Downloader
^^^^^^^^^^^^^^^^^^^^

Manual Data Loading
^^^^^^^^^^^^^^^^^^^

Data Generation
---------------

Generating Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Splitting Data
^^^^^^^^^^^^^^

Chunking Data
^^^^^^^^^^^^^

Writing Your Own DatasetGenerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
