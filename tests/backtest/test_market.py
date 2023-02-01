"""Test the Market classes."""

from statistics import mean

import pandas as pd

from coin_test.backtest import ConstantSlippage
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
