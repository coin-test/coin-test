"""Define the orchestration function."""

import multiprocessing
from multiprocessing import Queue
import os
import pickle
from typing import cast, Iterator

import pandas as pd

from ..backtest import (
    BacktestResults,
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


def _run_backtest(
    i: int,
    datasets: list[PriceDataset],
    strategies: list[Strategy],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    output_folder: str | None = None,
) -> BacktestResults:
    composer = Composer(datasets, backtest_length)
    sim = Simulator(
        composer,
        starting_portfolio,
        strategies,
        slippage_calculator,
        tx_calculator,
    )
    results = sim.run()
    if output_folder is not None:
        fn = f"{i}_backtest_results.pkl"
        with open(fn, "wb") as f:
            pickle.dump(results, f)
    return results


def _run_agent(
    receiver: Queue,
    sender: Queue,
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    output_folder: str | None = None,
) -> None:
    try:
        while (msg := receiver.get()) is not None:
            i, datasets, strategies = msg
            results = _run_backtest(
                i,
                datasets,
                strategies,
                slippage_calculator,
                tx_calculator,
                starting_portfolio,
                backtest_length,
                output_folder,
            )
            sender.put(results)
    except Exception as e:
        sender.put(e)


def _run_multiprocessed(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int,
    output_folder: str | None = None,
) -> list[BacktestResults]:
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
    for process in processes:
        process.start()

    sim_params = cast(
        list[tuple[int, list[PriceDataset], list[Strategy]] | None],
        _sim_param_generator(all_datasets, all_strategies),
    )
    messages = list(sim_params) + [None for _ in processes]
    for msg in messages:
        child_ret = worker_to_main.get()
        if isinstance(child_ret, Exception):
            raise child_ret
        results.append(child_ret)
        main_to_worker.put(msg)

    for process in processes:
        process.join()

    return results


def _gen_result(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int,
    output_folder: str | None = None,
) -> list[BacktestResults]:
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    if n_parallel <= 1:
        result = _run_backtest(
            0,
            all_datasets[0],
            all_strategies[0],
            slippage_calculator,
            tx_calculator,
            starting_portfolio,
            backtest_length,
            output_folder,
        )
        results = [result]
    else:
        results = _run_multiprocessed(
            all_datasets,
            all_strategies,
            slippage_calculator,
            tx_calculator,
            starting_portfolio,
            backtest_length,
            n_parallel,
            output_folder,
        )
    return results


def run(
    all_datasets: list[list[PriceDataset]],
    all_strategies: list[list[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int = 1,
    output_folder: str | None = None,
) -> list[BacktestResults]:
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

    Returns:
        List: All results of backtests
    """
    results = _gen_result(
        all_datasets,
        all_strategies,
        slippage_calculator,
        tx_calculator,
        starting_portfolio,
        backtest_length,
        n_parallel,
        output_folder,
    )

    # ... Call analysis on results here ...

    return results
