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
from coin_test.data import CustomDataset
from coin_test.util import AssetPair, Ticker, Money, Side
```
Then, import data from a CSV or another source to load
into the backtest.
```python
dataset_file = "data/ETHUSDT-1h-monthly/BTCUSDT-1h-2017-08.csv"
header = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
    "Quote asset volume", "Number of trades", "Taker buy base asset volume",
    "Taker buy quote asset volume", "Ignore"
]
df = pd.read_csv(dataset_file, names=header)
df = df.drop(columns=["Close Time", "Quote asset volume", "Number of trades",
                      "Taker buy base asset volume",
                      "Taker buy quote asset volume", "Ignore"])
df["Open Time"] //= 1000  # To seconds
df = df.sort_values(by=["Open Time"])

# define dataset metadata
btc, usdt = asset_pair = AssetPair.from_str("BTC USDT")
freq = "H"

dataset = CustomDataset(df, freq, asset_pair)
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
To run the backtest, define the datasets

```python
from coin_test.backtest import ConstantSlippage, ConstantTransactionFeeCalculator
sc = ConstantSlippage(50)
tc = ConstantTransactionFeeCalculator(50)

datasets = [dataset]
strategies = [MACD]

results = coin_test.run(datasets, strategies, sc, tc,
                        portfolio, pd.Timedelta(days=90),
                        n_parallel=8)
```
