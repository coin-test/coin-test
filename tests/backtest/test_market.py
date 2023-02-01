"""Test the Market classes."""

from statistics import mean

import numpy as np
import pandas as pd

from coin_test.backtest import ConstantSlippage, GaussianSlippage
from coin_test.util import AssetPair, Side


def test_constant_slippage(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
) -> None:
    """Price increases correctly on buy."""
    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )

    BASIS_POINT_ADJ = 10
    constant_slippage = ConstantSlippage(BASIS_POINT_ADJ)
    expected_slippage_buy = average_price * BASIS_POINT_ADJ / 10000

    assert expected_slippage_buy == constant_slippage.calculate(
        asset_pair, Side.BUY, timestamp_asset_price
    )

    expected_slippage_sell = average_price * -BASIS_POINT_ADJ / 10000

    assert expected_slippage_sell == constant_slippage.calculate(
        asset_pair, Side.SELL, timestamp_asset_price
    )


def test_gaussian_slippage_buy(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Price increases correctly on buy."""
    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )
    bp_mean = 10
    bp_std = 2
    rng = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)

    gaussian_slippage = GaussianSlippage(rng2, bp_mean, bp_std)

    expected_slippage_buy = average_price * rng.normal(bp_mean, bp_std) / 10000

    assert expected_slippage_buy == gaussian_slippage.calculate(
        asset_pair, Side.BUY, timestamp_asset_price
    )


def test_gaussian_slippage_sell(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
) -> None:
    """Price increases correctly on Sell."""
    curr_price = timestamp_asset_price[asset_pair]
    average_price = mean(
        (
            curr_price["Open"].iloc[0],
            curr_price["High"].iloc[0],
            curr_price["Low"].iloc[0],
            curr_price["Close"].iloc[0],
        )
    )
    bp_mean = 10
    bp_std = 2
    rng = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)

    gaussian_slippage = GaussianSlippage(rng2, bp_mean, bp_std)

    expected_slippage_buy = average_price * rng.normal(bp_mean, bp_std) / 10000

    assert expected_slippage_buy == gaussian_slippage.calculate(
        asset_pair, Side.SELL, timestamp_asset_price
    )
