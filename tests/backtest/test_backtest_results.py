"""Test the BacktestResults class."""

from unittest.mock import MagicMock, Mock

import pandas as pd
from pytest_mock import MockerFixture

from coin_test.backtest import BacktestResults
from coin_test.backtest import (
    ConstantSlippage,
    ConstantTransactionFeeCalculator,
    Portfolio,
)
from coin_test.util import AssetPair, Ticker


def test_create_results_correctly(mocker: MockerFixture) -> None:
    """Initialize correctly."""
    mock_mdata = Mock()
    mock_df = Mock()
    mock_dset = Mock()
    mock_dset.metadata = mock_mdata
    mock_dset.df = mock_df

    mock_mdata2 = Mock()
    mock_df2 = Mock()
    mock_dset2 = Mock()
    mock_dset2.metadata = mock_mdata2
    mock_dset2.df = mock_df2

    composer = MagicMock()
    composer.datasets = {mock_mdata: mock_dset, mock_mdata2: mock_dset2}

    init_portfolio = Mock()

    strat1 = Mock()
    strat1.name = "strat1"
    strat2 = Mock()
    strat2.name = "strat2"
    strategies = [strat1, strat2]

    index = [pd.Timestamp("2023-01-01T12"), pd.Timestamp("2023-01-01T13")]
    portfolios = [Mock(), Mock()]
    col2 = [[Mock()], [Mock()]]
    col3 = [[Mock()], [Mock()]]
    price = [[Mock()], [Mock()]]
    sim_data = (index, portfolios, col2, col3, price)

    sim_results = pd.DataFrame(
        list(
            zip(
                sim_data[0],
                sim_data[1],
                sim_data[2],
                sim_data[3],
                sim_data[4],
                strict=True,
            )
        ),
        columns=["Timestamp", "Portfolios", "Trades", "Pending Trades", "Price"],
    )
    sim_results.set_index("Timestamp")

    slip = ConstantSlippage
    tx = ConstantTransactionFeeCalculator

    mocker.patch("coin_test.backtest.BacktestResults.create_date_price_df")
    BacktestResults.create_date_price_df.return_value = price

    results = BacktestResults(
        composer, init_portfolio, strategies, sim_data, slip, tx  # pyright: ignore
    )

    assert results._seed is None
    assert results._slippage_type == slip
    assert results._tx_fee_type == tx
    assert results._starting_portfolio == init_portfolio
    assert results._data_dict == {mock_mdata: mock_df, mock_mdata2: mock_df2}
    assert results._strategy_names == [strat1.name, strat2.name]
    pd.testing.assert_frame_equal(results._sim_data, sim_results)


def test_calculate_price_from_portfolio(mock_portfolio: Portfolio) -> None:
    """Correctly calculate the monetary value of a portfolio."""
    composer = MagicMock()

    prices_right_now = {
        AssetPair(Ticker("USDT"), Ticker("BTC")): pd.DataFrame(
            columns=["Open"], data=[20000.0]
        ),
        AssetPair(Ticker("USDT"), Ticker("ETH")): pd.DataFrame(
            columns=["Open"], data=[1000.0]
        ),
    }
    mock_get_timestep = MagicMock()

    mock_get_timestep.__getitem__.side_effect = prices_right_now.__getitem__
    composer.get_timestep.return_value = mock_get_timestep

    index = [pd.Timestamp("2023-01-01T12"), pd.Timestamp("2023-01-01T13")]
    portfolios = [mock_portfolio, mock_portfolio]
    col2 = [[Mock()], [Mock()]]
    col3 = [[Mock()], [Mock()]]
    sim_data = (index, portfolios, col2, col3)

    sim_results = pd.DataFrame(
        list(zip(sim_data[0], sim_data[1], sim_data[2], sim_data[3], strict=True)),
        columns=["Timestamp", "Portfolios", "Trades", "Pending Trades"],
    )
    sim_results.set_index("Timestamp")

    price_series = BacktestResults.create_date_price_df(sim_results, composer)

    pd.testing.assert_series_equal(price_series, pd.Series(data=[3756.0, 3756.0]))
