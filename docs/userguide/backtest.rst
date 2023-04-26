Backtest
========

.. contents:: Table of Contents
    :backlinks: none
    :local:
    :depth: 1

Once the data and strategy have both been defined, the backtests can be run. Coin-test allows for running many backtests in parallel with a variety of parameters to modify the simulation.

Running Backtest
----------------

Backtests can be run using the ``coin_test.run`` function. This function takes in all of the following arguments for backtesting:

* ``all_datasets``: A list of list of datasets, where each list of datasets is all price data for one backtest. For example, one list can include a ``Dataset`` for BTC/USDT data, and another can have ETH/USDT data. Each list of datasets is used to create a different backtest for the user.
* ``all_strategies``: A list of list of strategies, where each list of strategies is a group of strategies that runs together, and multiple lists of strategies can be run alongside each other to compare different strategies in the same backtesting conditions.
* ``starting_portfolio``: A ``Portfolio`` with the starting monetary value of all assets.
* ``backtest_length``: A pandas ``Timedelta`` object which represents how long a backtest should take.
* ``n_parallel``: The number of parallel backtests to run. When ``1``, backtests are run in sequence.
* ``output_folder``: Where the report and saved backtest results should be saved. The report will be at ``output_folder/report.html`` and the results will be at ``output_folder/backtest_results``
* ``slippage_calculator``: A ``SlippageCalculator`` to compute slippage.
* ``tx_calculator``: A ``TransactionFeeCalculator`` to compute transaction fees.
* ``build_from_save_results``: A path to load the backtest results from. If specified, backtests are not run, and analysis is built from the specified save data.

To use default values, consider the following example:

.. code-block:: python

    from coin_test import run
    datasets = Datasaver.load("datasets.pkl")
    strategies = [
        [strategy1, strategy2, strategy3],  # first test these three strategies working together
        [strategy4],  # then test this strategy all by itself
    ]

    btc, usdt = btc_usdt = AssetPair.from_str("BTC", "USDT")
    starting_portfolio = Portfolio(base_currency=usdt, assets={usdt: Money(100000, usdt)})
    backtest_length = pd.Timedelta(days=90)  # 90 day backtests
    run(datasets, strategies, starting_portfolio, backtest_length)

To add slippage or transaction fees, consider the following example, where custom functions can add this flexibility. Currently, ``ConstantTransactionFeeCalculator``, ``ConstantSlippage``, and ``GaussianSlippage`` are implemented.

.. code-block:: python

    from coin_test.backtest import ConstantTransactionFeeCalculator, ConstantSlippage
    transaction_fee = ConstantTransactionFeeCalculator(basis_points=100)
    slippage = ConstantSlippage(basis_points=25)
    run(datasets, strategies, starting_portfolio, backtest_length,
        slippage_calculator=slippage,
        tx_calculator=transaction_fee
    )


To allocate more cores to process simultaneous backtests and run the process faster, consider using the ``n_parallel`` argument.

.. code-block:: python

    run(datasets, strategies, starting_portfolio, backtest_length, n_parallel=8)


Saving and Loading
-------------------

It is also possible to save and load backtest results. When an ``output_folder`` is specified, the ``output_folder/backtest_results`` folder will be generated. Analysis can be re-generated from this folder using the ``build_from_save_results`` argument:

.. code-block:: python

    run(build_from_save_results="out_folder/backtest_results")


Doing so will generate the analysis from the saved results and will not run any backtests. Any other arguments passed to ``run`` will be ignored.
