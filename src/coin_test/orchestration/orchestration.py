"""Define the orchestration function."""
import multiprocessing
from multiprocessing import Queue
import os
from typing import cast, Iterator, Sequence

import pandas as pd
from tqdm import tqdm

from ..backtest import (
    BacktestResults,
    Portfolio,
    Simulator,
    SlippageCalculator,
    Strategy,
    TransactionFeeCalculator,
)
from ..data import Composer, PriceDataset


def _save(result: BacktestResults, output_folder: str | None, i: int) -> None:
    if output_folder is not None:
        fn = os.path.join(output_folder, f"{i}_backtest_results.pkl")
        result.save(fn)


def _run_backtest(
    datasets: Sequence[PriceDataset],
    strategies: Sequence[Strategy],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
) -> BacktestResults:
    """Run a single backtest."""
    composer = Composer(datasets, backtest_length)
    sim = Simulator(
        composer,
        starting_portfolio,
        strategies,
        slippage_calculator,
        tx_calculator,
    )
    return sim.run()


def _sim_param_generator(
    all_datasets: Sequence[Sequence[PriceDataset]],
    all_strategies: Sequence[Sequence[Strategy]],
) -> Iterator[tuple[int, Sequence[PriceDataset], Sequence[Strategy]]]:
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
) -> None:
    """Run method of a single process."""
    try:
        while (msg := receiver.get()) is not None:
            i, datasets, strategies = msg
            results = _run_backtest(
                datasets,
                strategies,
                slippage_calculator,
                tx_calculator,
                starting_portfolio,
                backtest_length,
            )
            sender.put((i, results))
    except Exception as e:
        sender.put(e)


def _gen_multiprocessed(
    all_datasets: Sequence[Sequence[PriceDataset]],
    all_strategies: Sequence[Sequence[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int,
    output_folder: str | None = None,
) -> list[BacktestResults]:
    """Run multiprocessed backtests."""
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
            ),
            daemon=True,
        )
        for _ in range(n_parallel)
    ]
    for process in processes:
        process.start()

    sim_params = cast(
        list[tuple[int, list[PriceDataset], list[Strategy]] | None],
        _sim_param_generator(all_datasets, all_strategies),
    )
    messages = list(sim_params) + [None for _ in processes]

    for msg in messages[: len(processes)]:
        main_to_worker.put(msg)

    print("Starting backtests...")
    results = []
    for msg in tqdm(messages[len(processes) :]):
        child_ret = worker_to_main.get()
        if isinstance(child_ret, Exception):
            raise child_ret
        i, result = child_ret
        _save(result, output_folder, i)
        results.append(result)
        main_to_worker.put(msg)

    for process in processes:
        process.join()

    return results


def _gen_serial(
    all_datasets: Sequence[Sequence[PriceDataset]],
    all_strategies: Sequence[Sequence[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    output_folder: str | None = None,
) -> list[BacktestResults]:
    """Run serial backtests."""
    print("Starting backtests...")
    results = []
    params = _sim_param_generator(all_datasets, all_strategies)
    for i, datasets, strategies in tqdm(params):
        result = _run_backtest(
            datasets,
            strategies,
            slippage_calculator,
            tx_calculator,
            starting_portfolio,
            backtest_length,
        )
        _save(result, output_folder, i)
        results.append(result)
    return results


def _gen_results(
    all_datasets: Sequence[Sequence[PriceDataset]],
    all_strategies: Sequence[Sequence[Strategy]],
    slippage_calculator: SlippageCalculator,
    tx_calculator: TransactionFeeCalculator,
    starting_portfolio: Portfolio,
    backtest_length: pd.Timedelta | pd.DateOffset,
    n_parallel: int,
    output_folder: str | None = None,
) -> list[BacktestResults]:
    """Run backtests."""
    if n_parallel > 1:
        return _gen_multiprocessed(
            all_datasets,
            all_strategies,
            slippage_calculator,
            tx_calculator,
            starting_portfolio,
            backtest_length,
            n_parallel,
            output_folder,
        )
    else:
        return _gen_serial(
            all_datasets,
            all_strategies,
            slippage_calculator,
            tx_calculator,
            starting_portfolio,
            backtest_length,
            output_folder,
        )


def run(
    all_datasets: Sequence[Sequence[PriceDataset]],
    all_strategies: Sequence[Sequence[Strategy]],
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
    results = _gen_results(
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
