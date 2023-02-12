"""Define the orchestration function."""

import os
import pickle
from typing import Iterator

import pandas as pd

from ..backtest import (
    Portfolio,
    Simulator,
    SlippageCalculator,
    Strategy,
    TransactionFeeCalculator,
)
from ..data import Composer, PriceDataset


def _sim_param_generator(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
) -> Iterator[tuple[list[PriceDataset], list[Strategy]]]:
    """Yield all combinations of datasets and strategies."""
    for datasets in all_datasets:
        for strategies in all_strategies:
            yield datasets, strategies


def run(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    output_folder: str | None = None,
) -> None:
    """Run a full set of backtests.

    Run all combinations of passed datasets and strategies in a multiprocessed
    manner, then produce an analysis report.

    Args:
        all_datasets: List of sets of datasets to be used in backtests.
        all_strategies: List of sets of strategies to be used in backtests.
        slippage_calculator: Slippage calculator to use in simulations.
        tx_calculator: Transaction calculator to use in simulations.
        starting_portfolio: Starting portfolio to use in simulations.
        backtest_length: Length of each backtest.
        output_folder: Folder to save backtest results to. If None, results are
            not saved to disk.
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    sim_params = _sim_param_generator(all_datasets, all_strategies)
    for i, (datasets, strategies) in enumerate(sim_params):
        composer = Composer(datasets, backtest_length)
        sim = Simulator(
            composer, starting_portfolio, strategies, slippage_calculator, tx_calculator
        )
        results = sim.run()
        if output_folder is not None:
            fn = f"{i}_backtest_results.pkl"
            with open(fn, "wb") as f:
                pickle.dump(results, f)
