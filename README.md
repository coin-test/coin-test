# coin-test

[![Tests](https://github.com/coin-test/coin-test/workflows/Tests/badge.svg)](https://github.com/coin-test/coin-test/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/coin-test/coin-test/branch/main/graph/badge.svg)](https://codecov.io/gh/coin-test/coin-test)
[![License](https://img.shields.io/pypi/l/coin-test)](https://pypi.org/project/coin-test/)
[![Version](https://img.shields.io/pypi/v/coin-test)](https://pypi.org/project/coin-test/)
[![Python Versions](https://img.shields.io/pypi/pyversions/coin-test)](https://pypi.org/project/coin-test/)

Coin-test is a backtesting library designed for cryptocurrency trading. It supports trading strategies across multiple currencies and advanced configurations of tests, including cron-based scheduled execution of strategies, synthetic data generation, slippage modeling, and trading fees.

## Quick Start

Coin-test runs on Python 3.10 or higher. Install the package via pip:

```sh
pip3 install coin-test
```

To run a backtest, import the coin-test library. Then define your data source, strategy, and test settings to run the analysis.

```python
import datetime as dt
import os
import pandas as pd

from coin_test.backtest import Portfolio, Strategy, MarketTradeRequest
from coin_test.data import BinanceDataset
from coin_test.util import AssetPair, Money, Side
```
Then, import data from Binance or a CSV to load
into the backtest.
```python
# define dataset metadata
btc, usdt = btc_usdt = AssetPair.from_str("BTC", "USDT")

dataset = BinanceDataset("BTC/USDT Daily Data", btc_usdt)  # default daily data over all time
```
Strategies are stored in classes as shown below. Each strategy
should have a schedule, which is a cron string representing
when this strategy is run, a lookback, which is how much
data is accessed in the strategy, and a `__call__` method
which returns a list of TradeRequest objects, which represent
trades the strategy wants to make.

```python
class MACD(Strategy):
    def __init__(self, asset_pair) -> None:
        """Initialize a MACD object."""
        super().__init__(
            name="MACD",
            asset_pairs=[asset_pair],
            schedule="0 9 * * *",
            lookback=dt.timedelta(days=26),
        )
        self.perc = 0.2

    def __call__(self, time, portfolio, lookback_data):
        """Execute test strategy."""
        asset_pair = self.asset_pairs[0]
        exp1 = lookback_data[asset_pair]["Close"].ewm(span=12 * 24, adjust=False).mean()
        exp2 = lookback_data[asset_pair]["Close"].ewm(span=26 * 24, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9 * 24, adjust=False).mean()

        if macd.iloc[-1] > exp3.iloc[-1]:
            return [MarketTradeRequest(
                asset_pair,
                Side.BUY,
                notional=portfolio.available_assets(usdt).qty * self.perc,
            )]
        elif macd.iloc[-1] < exp3.iloc[-1]:
            return [MarketTradeRequest(
                asset_pair,
                Side.SELL,
                qty=portfolio.available_assets(btc).qty * self.perc,
            )]
        return []
```
This package supports multiple strategies, train-test splits
for historical data, synthetic data, and further customization.
To run the backtest, create a portfolio with starting values of
assets and call the `run` method.

```python
portfolio = Portfolio(base_currency=usdt, assets={usdt: Money(100000, usdt)})
datasets = [dataset]
strategies = [MACD]

results = coin_test.run(datasets, strategies, portfolio,
                        backtest_length=pd.Timedelta(days=90),
                        n_parallel=8)
```
