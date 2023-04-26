Strategies
==========

The strategy is perhaps the component of the backtest that we care the most about. It is, after all, why we must run any backtest in the first place. Coin-test allows for the implementation of arbitrary strategies that can query and handle data of any frequency.

.. contents:: Table of Contents
    :backlinks: none
    :local:
    :depth: 1

Implementing a strategy
-----------------------

Writing a strategy requires two things. Setting parameters of what the strategy is and what behavior it should have when it executes. These two pieces are used by the backtesting engine during runtime to generate buy and sell signals

The parameter setting is fairly straightforward. The block below shows a demo strategy named “Demo” that executes every day at 9 am and has a lookback of 3 days.

.. code-block:: python

class Demo(Strategy):

    def __init__(self, asset_pair) -> None:

        """Initialize a Demo strategy."""

        super().__init__(name="Demo", asset_pairs=[asset_pair], schedule="0 9 * * *", lookback=dt.timedelta(days=3),)

The ``def __call__(self, time, portfolio, lookback_data):`` function must be implemented on all strategies to determine strategy behavior. This is the core of the strategy and where the complicated logic goes. The strategy should make use of the `lookback` data. This lookback is a Pandas Dataframe containing OHLC data.  Accessing the first(oldest) date in the lookback for Close data is shown below.

.. code-block:: python

lookback_data[asset_pair]["Close"].iloc[0]

The output of the ``__call__`` function should always be a list of ``TradeRequest``. An empty list indicates no trades. The creation of a ``TradeRequest`` is accomplished as shown below. This is a Buy request to buy the asset with 90% of the available base currency.

.. code-block:: python

MarketTradeRequest(asset_pair, Side.BUY, notional=portfolio.available_assets(base_ticker).qty * .9)

Stitching all of these pieces togethers creates a strategy. A simple strategy that Buys/Sells based on if the data has trended up or down over the lookback is shown below.

.. code-block:: python

class Demo(Strategy):

    def __init__(self, asset_pair) -> None:

        """Initialize a Demo strategy."""

        super().__init__(name="Demo", asset_pairs=[asset_pair], schedule="0 9 * * *", lookback=dt.timedelta(days=3),)

        self.perc = 0.5



    def __call__(self, time, portfolio, lookback_data):

        """Execute test strategy."""

        asset_ticker, base_ticker = asset_pair = self.asset_pairs[0]

        lookback = lookback_data[asset_pair]["Close"]



        # If the asset has gone up in price BUY

        if (lookback.iloc[-2] - lookback.iloc[0]) > 0:

            return [MarketTradeRequest(asset_pair, Side.BUY, notional=portfolio.available_assets(base_ticker).qty * self.perc,)]

        else: # the asset has gone down SELL

            return [MarketTradeRequest(asset_pair, Side.SELL, qty=portfolio.available_assets(asset_ticker).qty * self.perc,)]

Gotchas
-------

We strive to make coin-test as easy to use as possible. However, there are still some common pitfalls for strategy development.

Limit and Short Orders
^^^^^^^^^^^^^^^^^^^^^^

Notably, coin-test does not currently support limit orders or short orders. This is a feature that is in development.

Un-Fulfilled Orders
^^^^^^^^^^^^^^^^^^^

If an order exceeds the amount of currency available to the strategy, it will fail to execute. Currently, the strategy gives no direct feedback of this occurrence. However, a strategy can check whether an order was fulfilled by checking its current portfolio value against the previous portfolio values.

Transaction Fees
^^^^^^^^^^^^^^^^

When filing a buy order by notional, slippage will automatically be accounted for. That is, the total price of assets plus the slippage will equal the notional. However, the transaction fees are not accounted for in the notional. As a result, filing an order to sell all the assets of a single type will fail.

Length of lookback data
^^^^^^^^^^^^^^^^^^^^^^^

The length of lookback data the strategy receives is variable and dependent on the dataset. A lookback of five days may receive 24 * 5 datapoints if the dataset is in an hour frequency. Additionally, if the timestep is near the beginning of the dataset, the entire time range may not be fulfilled. For example, if the backtest starts on the first day of the dataset, the first call of the strategy will contain very little data as the dataset does not contain the data within the range. This issue can be avoided by running backtest shorter than the dataset length.

NaNs in close data
^^^^^^^^^^^^^^^^^^

The lookback data the strategy receives also contains open price data for the current day. However, the high, low and close data for the current day are NaN, as the strategy should not have access to this data. If your strategy is aggregating, for example, across the close column of the lookback data, then remember to trim the last day to avoid NaNs.
