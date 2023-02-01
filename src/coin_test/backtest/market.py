"""Define market conditions like slippage and transaction fees."""

from abc import ABC, abstractmethod
from statistics import mean

import pandas as pd

from ..util import AssetPair, Side


class SlippageCalculator(ABC):
    """Calculate the slippage of an asset."""

    @staticmethod
    @abstractmethod
    def calculate(
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> float:
        """Calculate slippage for current a current asset.

        Args:
            asset_pair: The asset pair for the trade
            side: The side for the trade
            current_asset_price: Current price data from composer

        Returns:
            float: The slippage for the transaction.
        """


class ConstantSlippage(SlippageCalculator):
    """A Constant slippage Calculator."""

    def __init__(self, basis_points: float) -> None:
        """Initialize a Constant SlippageCalculator.

        Args:
            basis_points: how many basis points of constant slippage to use
        """
        self.basis_points = basis_points

    def calculate(
        self,
        asset_pair: AssetPair,
        side: Side,
        current_asset_price: dict[AssetPair, pd.DataFrame],
    ) -> float:
        """Calculate slippage for current asset.

        Args:
            asset_pair: The asset pair for the trade
            side: The side for the trade
            current_asset_price: Current price data from composer

        Returns:
            float: The slippage for the transaction.
        """
        curr_price = current_asset_price[asset_pair]
        average_price = mean(curr_price[["Open", "High", "Low", "Close"]].iloc[0])

        if side == Side.BUY:
            return average_price * (self.basis_points / 10000)
        else:
            return average_price * (-self.basis_points / 10000)
