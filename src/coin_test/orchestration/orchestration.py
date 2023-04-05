"""Define the orchestration function."""
import logging
import multiprocessing
from multiprocessing import Queue
import os
import traceback
from typing import cast, Iterator, List, Sequence

import pandas as pd
from tqdm import tqdm

from ..analysis import build_datapane
from ..backtest import (
    BacktestResults,
    ConstantSlippage,
    ConstantTransactionFeeCalculator,
    Portfolio,
    Simulator,
    SlippageCalculator,
    Strategy,
    TransactionFeeCalculator,
)
from ..data import Composer, PriceDataset


logger = logging.getLogger(__name__)


def _save(result: BacktestResults, output_folder: str | None, i: int) -> None:
    if output_folder is not None:
        fn = os.path.join(output_folder, f"{i}_backtest_results.pkl")
        result.save(fn)


def _load(
    input_dir: str,
) -> List[BacktestResults]:
    """Wrapper to load list of backtest of results."""
    results: List[BacktestResults] = []

    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}', is not a directory")

    for file in os.listdir(input_dir):
        if file.endswith(".pkl"):
            results.append(BacktestResults.load(os.path.join(input_dir, file)))

    if len(results) == 0:
        raise ValueError(f"No backtest results found in {input_dir}")

    logger.info(f"Loaded {len(results)}BacktestResults from {input_dir}.")
    return results


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
        print(traceback.format_exc())
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

    ctx = multiprocessing.get_context("fork")
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
    all_datasets: Sequence[Sequence[PriceDataset]] | None = None,
    all_strategies: Sequence[Sequence[Strategy]] | None = None,
    starting_portfolio: Portfolio | None = None,
    backtest_length: pd.Timedelta | None = None,
    n_parallel: int = 1,
    output_folder: str | None = None,
    slippage_calculator: SlippageCalculator | None = None,
    tx_calculator: TransactionFeeCalculator | None = None,
    build_from_saved_results: None | str = None,
) -> list[BacktestResults]:
    """Run a full set of backtests.

    Run all combinations of passed datasets and strategies in a multiprocessed
    manner, then produce an analysis report.

    Args:
        all_datasets: List of sets of datasets to be used in backtests.
        all_strategies: List of sets of strategies to be used in backtests.
        starting_portfolio: Starting portfolio to use in simulations.
        backtest_length: Length of each backtest.
        n_parallel: Number of parallel simulations to run.
        output_folder: Folder to save backtest results to. If None, results are
            not saved to disk.
        slippage_calculator: Slippage calc to use in simulations. Defaults to Constant
        tx_calculator: Transaction calc to use in simulations. Defaults to Constant_tx
        build_from_saved_results: Filepath to load saved results from.

    Returns:
        List: All results of backtests

    Raises:
        ValueError: If build_from_save_results isn't provided,
            args must be provided to run a backtest
    """
    if build_from_saved_results is not None:
        logger.info(
            f"Building analysis from saved results from: {build_from_saved_results}"
        )
        results = _load(build_from_saved_results)

    else:
        if (
            all_datasets is None
            or all_strategies is None
            or starting_portfolio is None
            or backtest_length is None
        ):
            raise ValueError(
                "Must specify 'build_from_save_results' or all of: 'all_datasets',\
                    all_strategies', 'starting_portfolio', and 'backtest_length'"
            )

        if slippage_calculator is None:
            slippage_calculator = ConstantSlippage()
            logger.info("Backtesting with ConstantSlippage slippage.")
        if tx_calculator is None:
            tx_calculator = ConstantTransactionFeeCalculator()
            logger.info("Backtesting with ConstantTransactionFeeCalculator tx fees.")

        if output_folder is None:
            logger.info("BacktestResults are not being saved.")
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

    build_datapane(results)

    return results
