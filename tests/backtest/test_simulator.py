"""Test the Simulator class."""
from copy import copy
from unittest.mock import Mock, PropertyMock

import pandas as pd
from pytest_mock import MockerFixture

from coin_test.backtest import Simulator
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

    portfolio = Mock()
    type(portfolio).base_currency = PropertyMock(return_value=asset_pair.currency)
    portfolio_cp = copy(portfolio)
    type(portfolio_cp).base_currency = PropertyMock(return_value=asset_pair.asset)
    portfolio.adjust.return_value = portfolio_cp

    new_portfolio, completed_trades = Simulator._execute_orders(
        portfolio, orders, timestamp_asset_price
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

    portfolio = Mock()
    type(portfolio).base_currency = PropertyMock(return_value=asset_pair.currency)
    portfolio.adjust.return_value = None

    new_portfolio, completed_trades = Simulator._execute_orders(
        portfolio, orders, timestamp_asset_price
    )

    assert new_portfolio == portfolio
    assert completed_trades == []


def test_handle_pending_orders(
    asset_pair: AssetPair,
    timestamp_asset_price: dict[AssetPair, pd.DataFrame],
    mocker: MockerFixture,
) -> None:
    """Propogate Orders as expected."""
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
        orders, timestamp_asset_price, portfolio
    )

    assert pending_orders == [mock_trade2]
    assert new_portfolio == portfolio
    assert exec_trades == [mock_trade.build_trade]
