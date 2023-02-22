"""Functions to build Datapane Locally."""
from unittest.mock import Mock
import datapane as dp

from coin_test.backtest.backtest_results import BacktestResults
from coin_test.analysis.data_processing import DataframeGenerator


def flatten_strategies(results: BacktestResults) -> str:
    return "".join(results.strategy_names)


def _get_strategies(results: list[BacktestResults]) -> list[str]:
    return [flatten_strategies(r) for r in results]


def build_strategy_page(results: list[BacktestResults], strategy_name: str) -> dp.Page:
    filtered_results = [r for r in results if flatten_strategies(r) == strategy_name]
    return build_page(filtered_results)


def build_page(results: list[BacktestResults]) -> dp.Page:
    """Build a page based on a list of backtest results."""
    # if len(results) == 1:
    metrics = DataframeGenerator.create(results[0])
    page = dp.Page(
        # title=f"Results{''.join(results[0].strategy_names)} Metrics",
        blocks=[
            "### Metrics",
            dp.Group(metrics, metrics, columns=2),
        ],
    )

    return page
    # else:
    #     return dp.Page(
    #         title="Lol imagine having multi df support",
    #         blocks=[
    #             "### ...",
    #         ],
    #     )
    # Build cumulative Metrics


def build_dataset_page(results: list[BacktestResults]) -> dp.Page:
    """Build page that contains the real test data/split."""
    raise NotImplementedError("Missing support for named datasets")


def build_home_page() -> dp.Page:
    return build_page([Mock()])  # dp.Page(dp.Text("Coin-test Dashboard"))


def build_datapane(results: list[BacktestResults]) -> None:
    """Build Datapane from large set of results."""

    strategies = _get_strategies(results)

    strategy_pages = [build_strategy_page(s) for s in strategies]
    home_page = build_home_page()
    page_list = [home_page] + strategy_pages
    dashboard = dp.App(blocks=page_list)

    dashboard.save(path="report.html")


if __name__ == "__main__":
    build_datapane([])
