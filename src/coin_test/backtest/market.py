"""Define market conditions like slippage and transaction fees."""

from abc import ABC, abstractmethod
from statistics import mean

import numpy as np
import pandas as pd

from ..util import AssetPair, Side


class SlippageCalculator(ABC):
    """Calculate the slippage of an asset."""

    @abstractmethod
    def __call__(
        self,
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

    def __call__(
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


class GaussianSlippage(SlippageCalculator):
    """A Constant slippage Calculator."""

    def __init__(
        self, rng: np.random.Generator, mean_bp: float, std_dev_bp: float
    ) -> None:
        """Initialize a Gaussian SlippageCalculator.

        Args:
            rng: Numpy Random Number Generator ie. default_rng()
            mean_bp: Mean Basis points to slip
            std_dev_bp: Standard Deviation of Basis points of slippage
        """
        self.rng = rng
        self.mean_bp = mean_bp
        self.std_dev_bp = std_dev_bp

    def __call__(
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

        basis_points = float(self.rng.normal(self.mean_bp, self.std_dev_bp, 1))
        if side == Side.BUY:
            return average_price * (basis_points / 10000)
        else:
            return average_price * (basis_points / 10000)


class TransactionFeeCalculator(ABC):
    """Calculate the transactions fees for a trade."""

    @abstractmethod
    def __call__(
        self, asset_pair: AssetPair, amount: float, adjusted_price: float
    ) -> float:
        """Calculate transaction fees for a given trade request.

        Args:
            asset_pair: The asset pair for the trade
            amount: The quantity of the currency being traded
            adjusted_price: Price of the trade

        Returns:
            float: The transaction fee in the base currency.
        """


class ConstantTransactionFeeCalculator(TransactionFeeCalculator):
    """Calculate Constant the transactions fees for a trade."""

    def __init__(self, basis_points: float) -> None:
        """Initialize a Constant TransactionFeeCalculator.

        Args:
            basis_points: Basis Points used to calculate proportional fees
        """
        self.basis_points = basis_points

    def __call__(
        self,
        asset_pair: AssetPair,
        amount: float,
        adjusted_price: float,
    ) -> float:
        """Calculate transaction fees for a given trade request.

        Args:
            asset_pair: The asset pair for the trade
            amount: The quantity of the currency being traded
            adjusted_price: Price of the trade

        Returns:
            float: The transaction fee in the base currency.
        """
        return float(amount * adjusted_price * self.basis_points / 10000)
