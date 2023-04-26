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
import coin_test
import pandas as pd
from coin_test.backtest import Portfolio, Strategy, MarketTradeRequest
from coin_test.data import BinanceDataset, FillProcessor, GarchDatasetGenerator
from coin_test.util import AssetPair, Money, Side
```

Then define the assets to trade and your starting portfolio.
```python
# Define assets traded and an initial portfolio
eth, usdt = eth_usdt = AssetPair.from_str("ETH", "USDT")
portfolio = Portfolio(base_currency=usdt, assets={eth: Money(eth, 0), usdt: Money(usdt, 10000)})
```
Then, import daily historical data from Binance for the backtest and fill gaps.
```python
# Download the last 150 days of data
freq ='d'
dataset = BinanceDataset("ETH/USDT Daily Data", eth_usdt, freq=freq, start=dt.datetime.today()-dt.timedelta(days=150))
dataset.process([FillProcessor(freq)])
```

Next we wish to generate synthetic data to allow backtesting on a variety of future market conditions.
The existing data is split into a train/test split and then fed to a GARCH statistical model to generate new data.
```python
# Split the data into train test split
train, test = dataset.split(percent=0.75)

# Generate 30 synthetic datasets 90 days long and package them for backtesting
datasets = GarchDatasetGenerator(train).generate(timedelta=pd.Timedelta(days=90), n=30)
datasets = [[d] for d in datasets] # Package the datasets for backtesting
```

Strategies are stored in classes as shown below. Each strategy
should have a schedule, which is a cron string representing
when this strategy is run, a lookback, which is how much
data is accessed in the strategy, and a `__call__` method
which returns a list of TradeRequest objects, which represent
trades the strategy wants to make.

```python
class MACD_discrete_days(Strategy):
	 def __init__(self, asset_pair) -> None:
        """Initialize a MACD object.
           This strategy uses a 26, 12, 9 standard EMACD calculation to generate buy sell signals.
           Made to be used with Hour data"""
        super().__init__(
            name="MACD_Discrete_day",
            asset_pairs=[asset_pair],
            schedule="0 9 * * *",
            lookback=dt.timedelta(days=26),
        )
        self.perc = .98
        self.invested = False

    def __call__(self, time, portfolio, lookback_data):
        """Execute strategy."""
        asset_ticker, base_ticker = asset_pair = self.asset_pairs[0]
        data = lookback_data[asset_pair]["Close"]
        
        macd, signal, fast_ma, slow_ma = macd_indicator(data, 12, 26, 9)

        if signal < macd and not self.invested:
            self.invested =True
            return [MarketTradeRequest(
                asset_pair,
                Side.BUY,
                notional=portfolio.available_assets(base_ticker).qty * self.perc,
            )]
        elif signal > macd and self.invested:
            self.invested = False
            return [MarketTradeRequest(
                asset_pair,
                Side.SELL,
                qty=portfolio.available_assets(asset_ticker).qty * self.perc,
            )]
        else:
            return []
```
To run the backtest, create a portfolio with starting values of
assets and call the `run` method. This package supports multiple strategies,
and further customization, see our user guide and docs for more advanced features. 

```python
# Package the strategies before backtesting
strategies = [[MACD_discrete_days(eth_usdt)]]

# Run the backtest and generate the report
results = coin_test.run(datasets, strategies, portfolio, backtest_length=pd.Timedelta(days=90), n_parallel=8)
```
