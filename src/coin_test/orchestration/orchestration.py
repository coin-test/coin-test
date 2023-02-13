"""Define the orchestration function."""

import multiprocessing
from multiprocessing import Queue
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
) -> Iterator[tuple[int, list[PriceDataset], list[Strategy]]]:
    """Yield all combinations of datasets and strategies."""
    i = 0
    for datasets in all_datasets:
        for strategies in all_strategies:
            yield i, datasets, strategies
            i += 1


def _run_agent(
    receiver: Queue,
    sender: Queue,
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    output_folder: str | None = None,
) -> None:
    while (msg := receiver.get()) is not None:
        i, datasets, strategies = msg
        composer = Composer(datasets, backtest_length)
        sim = Simulator(
            composer, starting_portfolio, strategies, slippage_calculator, tx_calculator
        )
        results = sim.run()
        if output_folder is not None:
            fn = f"{i}_backtest_results.pkl"
            with open(fn, "wb") as f:
                pickle.dump(results, f)
        sender.put(results)


def run(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int,
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
        n_parallel: Number of parallel simulations to run.
        output_folder: Folder to save backtest results to. If None, results are
            not saved to disk.
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    main_to_worker = Queue()
    worker_to_main = Queue()

    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(
            target=_run_agent,
            args=(
                main_to_worker,
                worker_to_main,
                slippage_calculator,
                tx_calculator,
                starting_portfolio,
                backtest_length,
                output_folder,
            ),
        )
        for _ in range(n_parallel)
    ]

    results = []
    sim_params = _sim_param_generator(all_datasets, all_strategies)
    for process in processes:
        process.start()
        main_to_worker.put(next(sim_params))
    for params in sim_params:
        ret = worker_to_main.get()
        results.append(ret)
        main_to_worker.put(params)

    for _ in processes:
        main_to_worker.put(None)

    for process in processes:
        process.join()

    # ... Call analysis on results here ...
