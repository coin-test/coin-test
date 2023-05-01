"""General analysis utilities."""

from typing import Sequence

from coin_test.backtest.backtest_results import BacktestResults


def _flatten_strategies(results: BacktestResults) -> str:
    return "-".join(results.strategy_names)


def get_strategies(results: Sequence[BacktestResults]) -> list[str]:
    """Get all unique strategies in list of backtest results.

    Args:
        results: List of bakcktest results.

    Returns:
        strategies: List of strategy names
    """
    return sorted(list(set(_flatten_strategies(r) for r in results)))


def get_strategy_results(
    results: Sequence[BacktestResults],
) -> dict[str, list[BacktestResults]]:
    """Build mapping of startegy to results.

    Args:
        results: List of bakcktest results.

    Returns:
        strategies: Dictionary mapping strategy name to strategy backtest results.
    """
    strategies = get_strategies(results)
    strategy_results = {}
    for strategy in strategies:
        strategy_results[strategy] = []
    for result in results:
        strategy = _flatten_strategies(result)
        strategy_results[strategy].append(result)
    return strategy_results
