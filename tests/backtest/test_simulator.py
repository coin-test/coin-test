"""Test the Simulator class."""
from copy import copy
import datetime as dt
import logging
from unittest.mock import Mock, PropertyMock

from croniter import croniter
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from coin_test.backtest import BacktestResults, Simulator, Strategy
from coin_test.util import AssetPair, Side


def test_collect_asset_pairs(
    asset_pair: AssetPair, asset_pair_btc_eth: AssetPair, asset_pair_eth_usdt: AssetPair
) -> None:
    """Collect Asset Pairs from strategies."""
    strategy1 = Mock()
    type(strategy1).asset_pairs = PropertyMock(
        return_value=[asset_pair, asset_pair_btc_eth]
    )
    strategy2 = Mock()
    type(strategy2).asset_pairs = PropertyMock(
        return_value=[asset_pair_btc_eth, asset_pair_eth_usdt]
    )

    asset_pairs = Simulator._collect_asset_pairs([strategy1, strategy2])

    assert asset_pairs == set([asset_pair, asset_pair_btc_eth, asset_pair_eth_usdt])


def test_simulation_validation_match(
    asset_pair: AssetPair,
    asset_pair_btc_eth: AssetPair,
    asset_pair_eth_usdt: AssetPair,
    mocker: MockerFixture,
) -> None:
    """Asset Pairs match."""
    mock_composer = Mock()
    type(mock_composer).datasets = PropertyMock(
        return_value=dict.fromkeys(
            [asset_pair, asset_pair_btc_eth, asset_pair_eth_usdt], None
        )
    )

    mocker.patch("coin_test.backtest.Simulator._collect_asset_pairs")
    Simulator._collect_asset_pairs.return_value = set([asset_pair, asset_pair_btc_eth])
    valid = Simulator._validate(mock_composer, [])
    assert valid is True


def test_simulation_validation_mismatch(
    asset_pair: AssetPair,
    asset_pair_btc_eth: AssetPair,
    asset_pair_eth_usdt: AssetPair,
    mocker: MockerFixture,
) -> None:
    """Invalid Asset Pairs with match."""
    mock_composer = Mock()
    type(mock_composer).datasets = PropertyMock(
        return_value=dict.fromkeys([asset_pair, asset_pair_btc_eth], None)
    )

    mocker.patch("coin_test.backtest.Simulator._collect_asset_pairs")
    Simulator._collect_asset_pairs.return_value = set(
        [asset_pair, asset_pair_btc_eth, asset_pair_eth_usdt]
    )

    valid = Simulator._validate(mock_composer, [])
    assert valid is False


def _make_mock_trade_request(
    asset_pair: AssetPair, side: Side, qty: float, should_execute: bool, price: float
) -> Mock:
    mock_tr = Mock()
    type(mock_tr).asset_pair = PropertyMock(return_value=asset_pair)
    type(mock_tr).side = PropertyMock(return_value=side)
    type(mock_tr).qty = PropertyMock(return_value=qty)
    mock_tr.should_execute.return_value = should_execute
    mock_trade = Mock()
    type(mock_trade).asset_pair = PropertyMock(return_value=asset_pair)
    type(mock_trade).side = PropertyMock(return_value=side)
    type(mock_trade).price = PropertyMock(return_value=price)
    type(mock_trade).amount = PropertyMock(return_value=qty)
    mock_tr.build_trade.return_value = mock_trade
    return mock_tr


def test_split_pending_orders_all_execute(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Execute all orders."""
    mock_trade1 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=2, should_execute=True, price=1
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.SELL, qty=1, should_execute=True, price=1
    )

    pending_orders = [mock_trade1, mock_trade2]
    exec_orders, remaining_orders = Simulator._split_pending_orders(
        pending_orders, timestamp_asset_price
    )

    assert remaining_orders == []
    assert exec_orders == [mock_trade1, mock_trade2]


def test_split_pending_orders_none_execute(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Execute no orders."""
    mock_trade1 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=2, should_execute=False, price=1
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.SELL, qty=1, should_execute=False, price=1
    )

    pending_orders = [mock_trade1, mock_trade2]
    exec_orders, remaining_orders = Simulator._split_pending_orders(
        pending_orders, timestamp_asset_price
    )

    assert remaining_orders == [mock_trade1, mock_trade2]
    assert exec_orders == []


def test_split_pending_orders_split_execute(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Execute no orders."""
    mock_trade1 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=2, should_execute=True, price=1
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.SELL, qty=1, should_execute=False, price=1
    )

    pending_orders = [mock_trade1, mock_trade2]
    exec_orders, remaining_orders = Simulator._split_pending_orders(
        pending_orders, timestamp_asset_price
    )

    assert remaining_orders == [mock_trade2]
    assert exec_orders == [mock_trade1]


def test_execute_orders_success(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Adjust Portfolio according to orders."""
    mock_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=1
    )
    orders = [mock_trade, mock_trade2]

    slippage_calculator = Mock()
    tx_fee_calculator = Mock()
    portfolio = Mock()
    type(portfolio).base_currency = PropertyMock(return_value=asset_pair.currency)
    portfolio_cp = copy(portfolio)
    type(portfolio_cp).base_currency = PropertyMock(return_value=asset_pair.asset)
    portfolio.adjust.return_value = portfolio_cp

    new_portfolio, completed_trades = Simulator._execute_orders(
        portfolio, orders, timestamp_asset_price, slippage_calculator, tx_fee_calculator
    )

    assert new_portfolio.base_currency == asset_pair.asset
    assert completed_trades == [mock_trade.build_trade(), mock_trade2.build_trade()]


def test_execute_orders_failures(
    asset_pair: AssetPair, timestamp_asset_price: dict[AssetPair, pd.DataFrame]
) -> None:
    """Adjust Portfolio according to orders."""
    mock_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )

    orders = [mock_trade]

    slippage_calculator = Mock()
    tx_fee_calculator = Mock()

    portfolio = Mock()
    type(portfolio).base_currency = PropertyMock(return_value=asset_pair.currency)
    portfolio.adjust.return_value = None

    new_portfolio, completed_trades = Simulator._execute_orders(
        portfolio, orders, timestamp_asset_price, slippage_calculator, tx_fee_calculator
    )

    assert new_portfolio == portfolio
    assert completed_trades == []


def test_build_croniter_schedule() -> None:
    """Schedule adjusted appropriately."""
    strat1 = Mock()
    strat1.schedule = "* * * * *"
    strat2 = Mock()
    strat2.schedule = "* */5 * * *"  # Every 5th minute
    start_time = pd.Timestamp(dt.datetime(2022, 12, 28, 22, 51))
    schedule = Simulator._build_croniter_schedule(start_time, [strat1, strat2])

    assert schedule[0][1].get_current(dt.datetime) == start_time
    assert schedule[1][1].get_current(dt.datetime) != start_time


def test_handle_pending_orders(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Propogate Orders as expected."""
    slippage_calculator = Mock()
    transaction_fee_calculator = Mock()

    portfolio = Mock()
    mock_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=1
    )
    orders = [mock_trade, mock_trade2]

    mocker.patch("coin_test.backtest.Simulator._split_pending_orders")
    Simulator._split_pending_orders.return_value = ([mock_trade], [mock_trade2])
    mocker.patch("coin_test.backtest.Simulator._execute_orders")
    Simulator._execute_orders.return_value = (portfolio, [mock_trade.build_trade])

    pending_orders, new_portfolio, exec_trades = Simulator._handle_pending_orders(
        orders,
        timestamp_asset_price,
        portfolio,
        slippage_calculator,
        transaction_fee_calculator,
    )

    assert pending_orders == [mock_trade2]
    assert new_portfolio == portfolio
    assert exec_trades == [mock_trade.build_trade]


def test_run_strategies(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Trades created from running strategies."""
    mock_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )
    mock_trade2 = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )
    strategy1 = Mock()
    strategy1.return_value = [mock_trade]
    type(strategy1).asset_pairs = PropertyMock(return_value=[asset_pair])
    type(strategy1).lookback = PropertyMock(return_value=pd.Timedelta(days=1))

    strategy2 = Mock()
    strategy2.return_value = [mock_trade2]
    type(strategy2).asset_pairs = PropertyMock(return_value=[asset_pair])
    type(strategy2).lookback = PropertyMock(return_value=pd.Timedelta(days=2))

    mock_composer = Mock()
    mock_composer.get_range.return_value = dict.fromkeys([asset_pair], None)
    type(mock_composer).start_time = PropertyMock(return_value=dt.datetime.now())
    type(mock_composer).end_time = PropertyMock(return_value=dt.datetime.now())
    type(mock_composer).freq = PropertyMock(return_value=pd.DateOffset(days=3))

    portfolio = Mock()

    mocker.patch("coin_test.backtest.Simulator._strategies_to_run")
    Simulator._strategies_to_run.return_value = [strategy1, strategy2]
    mocker.patch("coin_test.backtest.Simulator._validate")
    Simulator._validate.return_value = True

    mock_slippage_calculator = Mock()
    mock_transaction_calculator = Mock()
    sim = Simulator(
        mock_composer,
        portfolio,
        [],
        mock_slippage_calculator,
        mock_transaction_calculator,
    )

    time = pd.Timestamp(dt.datetime.now())
    schedule = [(strategy1, croniter("@yearly")), (strategy2, croniter("@yearly"))]
    trade_requests = sim.run_strategies(schedule, time, portfolio)

    strategy1.assert_called_with(time, portfolio, mock_composer.get_range.return_value)
    strategy2.assert_called_with(time, portfolio, mock_composer.get_range.return_value)

    assert trade_requests == [mock_trade, mock_trade2]


def test_warning_on_run_strategies(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Trades created from running strategies."""
    caplog.set_level(logging.WARN)

    # Error on Strategy
    strategy1 = Mock(side_effect=IndexError("Error Message in Strategy"))
    type(strategy1).asset_pairs = PropertyMock(return_value=[asset_pair])
    type(strategy1).lookback = PropertyMock(return_value=pd.Timedelta(days=1))
    type(strategy1).name = "strategy1"

    mock_composer = Mock()
    mock_composer.get_range.return_value = dict.fromkeys([asset_pair], None)
    type(mock_composer).start_time = PropertyMock(return_value=dt.datetime.now())
    type(mock_composer).end_time = PropertyMock(return_value=dt.datetime.now())
    type(mock_composer).freq = PropertyMock(return_value=pd.DateOffset(days=3))

    portfolio = Mock()

    mocker.patch("coin_test.backtest.Simulator._strategies_to_run")
    Simulator._strategies_to_run.return_value = [strategy1]
    mocker.patch("coin_test.backtest.Simulator._validate")
    Simulator._validate.return_value = True

    mock_slippage_calculator = Mock()
    mock_transaction_calculator = Mock()
    sim = Simulator(
        mock_composer,
        portfolio,
        [],
        mock_slippage_calculator,
        mock_transaction_calculator,
    )

    time = pd.Timestamp(dt.datetime.now())
    schedule = [(strategy1, croniter("@yearly"))]
    trade_requests = sim.run_strategies(schedule, time, portfolio)

    strategy1.assert_called_with(time, portfolio, mock_composer.get_range.return_value)

    assert caplog.record_tuples == [
        (
            "coin_test.backtest.simulator",
            logging.WARN,
            "Strategy strategy1 caught an exception.",
        ),
        ("coin_test.backtest.simulator", logging.WARN, "Error Message in Strategy"),
    ]

    assert trade_requests == []

    sim_err = Simulator(
        mock_composer,
        portfolio,
        [],
        mock_slippage_calculator,
        mock_transaction_calculator,
        warn_on_error=False,
    )

    with pytest.raises(ValueError):
        sim_err.run_strategies(schedule, time, portfolio)


def test_simulation_runs_correct_strategies(
    schedule: list[tuple[Strategy, croniter]], timestamp: dt.datetime
) -> None:
    """Simulation runs correct strategies at correct times."""
    simulation_dt = pd.tseries.offsets.DateOffset(hours=1)
    strategies_to_run = Simulator._strategies_to_run(schedule, timestamp, simulation_dt)

    assert schedule[0][0] in strategies_to_run  # minute < hour
    assert schedule[1][0] in strategies_to_run  # hour = hour
    assert schedule[2][0] not in strategies_to_run  # day > hour
    assert len(strategies_to_run) == 2  # 2/3

    # now try again with a day offset, all strategies should run
    new_simulation_dt = pd.tseries.offsets.DateOffset(days=1)
    new_strategies_to_run = Simulator._strategies_to_run(
        schedule, timestamp + simulation_dt, new_simulation_dt
    )

    assert schedule[0][0] in new_strategies_to_run  # minute < day
    assert schedule[1][0] in new_strategies_to_run  # hour < day
    assert schedule[2][0] in new_strategies_to_run  # day = day
    assert len(new_strategies_to_run) == 3  # 3/3


def test_run(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Run Simulator correctly."""
    mock_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=2
    )
    mock_executed_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=3
    )

    mock_ran_trade = _make_mock_trade_request(
        asset_pair, Side.BUY, qty=1, should_execute=True, price=4
    )

    mock_composer = Mock()
    mock_composer.get_range.return_value = dict.fromkeys([asset_pair], None)
    time = pd.Timestamp.now()
    type(mock_composer).start_time = PropertyMock(return_value=time)
    type(mock_composer).end_time = PropertyMock(
        return_value=time + pd.DateOffset(days=2)
    )
    type(mock_composer).freq = PropertyMock(return_value=pd.DateOffset(days=1))
    mock_composer.get_timestep.return_value = timestamp_asset_price
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
    mock_composer.datasets = {mock_mdata: mock_dset, mock_mdata2: mock_dset2}

    portfolio = Mock()
    portfolio_handled = Mock()

    strategy1 = Mock()
    strategy1.return_value = [mock_trade]
    type(strategy1).asset_pairs = PropertyMock(return_value=[asset_pair])
    type(strategy1).lookback = PropertyMock(return_value=pd.Timedelta(days=1))
    type(strategy1).schedule = PropertyMock(return_value="* * * * *")

    strategy2 = Mock()
    strategy2.return_value = [mock_trade]
    type(strategy2).asset_pairs = PropertyMock(return_value=[asset_pair])
    type(strategy2).lookback = PropertyMock(return_value=pd.Timedelta(days=1))
    type(strategy2).schedule = PropertyMock(
        return_value=f"* * {(time + pd.DateOffset(days=1)).day} * *"
    )

    mocker.patch("coin_test.backtest.Simulator._build_croniter_schedule")
    Simulator._build_croniter_schedule.return_value = None

    mocker.patch("coin_test.backtest.Simulator._handle_pending_orders")
    Simulator._handle_pending_orders.return_value = (
        [mock_trade],
        portfolio_handled,
        [mock_executed_trade.build_trade()],
    )

    mocker.patch("coin_test.backtest.Simulator.run_strategies")
    Simulator.run_strategies.return_value = [mock_ran_trade]

    mocker.patch("coin_test.backtest.Simulator._validate")
    Simulator._validate.return_value = True

    mock_slippage_calculator = Mock()
    mock_transaction_calculator = Mock()
    # TODO: This should be refactored to be a mocked object
    sim = Simulator(
        mock_composer,
        portfolio,
        [strategy1, strategy2],
        mock_slippage_calculator,
        mock_transaction_calculator,
    )

    mocker.patch("coin_test.backtest.BacktestResults.create_date_price_df")
    price = [Mock(), Mock(), Mock()]
    BacktestResults.create_date_price_df.return_value = price

    backtest_results = sim.run()
    ex_hist_times = [
        mock_composer.start_time - mock_composer.freq,
        mock_composer.start_time,
        mock_composer.start_time + mock_composer.freq,
    ]
    ex_hist_port = [portfolio, portfolio_handled, portfolio_handled]
    ex_hist_trades = [
        [],
        [
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
        ],
        [
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
            mock_executed_trade.build_trade(),
        ],
    ]
    ex_pending = [
        [],
        [mock_trade, mock_trade],
        [mock_trade, mock_trade, mock_trade, mock_trade],
    ]
    df = pd.DataFrame(
        list(
            zip(
                ex_hist_times,
                ex_hist_port,
                ex_hist_trades,
                ex_pending,
                price,
                strict=True,
            )
        ),
        columns=["Timestamp", "Portfolios", "Trades", "Pending Trades", "Price"],
    )
    df = df.set_index("Timestamp", drop=True)

    # TODO: BacktestResults should be mocked and just called with correct items
    pd.testing.assert_frame_equal(backtest_results.sim_data, df)
    assert portfolio == backtest_results.starting_portfolio


def test_construct_simulator(asset_pair: AssetPair, mocker: MockerFixture) -> None:
    """Simulator initilaizes correctly."""
    slippage_calculator = Mock()
    mock_transaction_calculator = Mock()

    mock_composer = Mock()
    mock_composer.get_range.return_value = dict.fromkeys([asset_pair], None)
    time = dt.datetime.now()
    type(mock_composer).start_time = PropertyMock(return_value=time)
    type(mock_composer).end_time = PropertyMock(
        return_value=time + pd.Timedelta(days=2)
    )
    type(mock_composer).freq = PropertyMock(return_value=pd.DateOffset(days=1))

    portfolio = Mock()
    strategy1 = Mock()

    mocker.patch("coin_test.backtest.Simulator._validate")
    Simulator._validate.return_value = True

    sim = Simulator(
        mock_composer,
        portfolio,
        [strategy1],
        slippage_calculator,
        mock_transaction_calculator,
    )

    assert sim._portfolio is portfolio
    assert sim._composer is mock_composer
    assert sim._strategies == [strategy1]
    assert sim._start_time == mock_composer.start_time
    assert sim._end_time == mock_composer.end_time
    assert sim._simulation_dt == mock_composer.freq
    assert sim._slippage_calculator == slippage_calculator


def test_construct_invalid_simulator(
    asset_pair: AssetPair, mocker: MockerFixture
) -> None:
    """Invalid Simulator raises ValueError."""
    mock_composer = Mock()
    mock_transaction_calculator = Mock()

    mock_composer.get_range.return_value = dict.fromkeys([asset_pair], None)
    time = dt.datetime.now()
    type(mock_composer).start_time = PropertyMock(return_value=time)
    type(mock_composer).end_time = PropertyMock(
        return_value=time + pd.Timedelta(days=2)
    )
    type(mock_composer).freq = PropertyMock(return_value=pd.DateOffset(days=1))

    portfolio = Mock()
    strategy1 = Mock()

    mocker.patch("coin_test.backtest.Simulator._validate")
    Simulator._validate.return_value = False

    with pytest.raises(ValueError):
        _ = Simulator(
            mock_composer, portfolio, [strategy1], Mock(), mock_transaction_calculator
        )
