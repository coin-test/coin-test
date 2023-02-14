"""Tear sheet class implementation."""

import pandas as pd


from ..backtest import BacktestResults, Portfolio
from ..data import Composer
from ..util import AssetPair


class TearSheet:
    """A tear sheet."""

    def __init__(self, backtest_results: BacktestResults) -> None:
        """Initialize a TearSheet."""
        pass

    @staticmethod
    def create_date_price_df(backtest_results: BacktestResults) -> pd.DataFrame:
        """Create a TimeSeries dataframe for portfolio value over time."""
        return pd.DataFrame()

    @staticmethod
    def value_from_portfolio(p: Portfolio, t: pd.Timestamp, c: Composer) -> float:
        """Get the monetary value of a portfolio."""
        base_currency = p.base_currency
        all_prices = c.get_timestep(t)
        total = p.assets[base_currency].qty

        for ticker, money in p.assets.items():
            asset_pair = AssetPair(base_currency, ticker)
            conversion = all_prices[asset_pair]["Open"].iloc[0]
            total += conversion * money.qty

        return total
