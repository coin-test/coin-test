"""Test the orchestration function."""

import os
from typing import cast
from unittest.mock import call, MagicMock, Mock

import pytest
from pytest_mock import MockerFixture

import coin_test.orchestration.orchestration as orc


def _build_backtest_arg_mocks() -> tuple[
    list[list[Mock]], list[list[Mock]], Mock, Mock, Mock, Mock
]:
    """Build empty mocks for all backtest args."""
    strategies = [[Mock(), Mock()], [Mock()], [Mock(), Mock(), Mock()]]
    datasets = [[Mock(), Mock(), Mock()], [Mock(), Mock()]]
    slippage_calculator = Mock()
    transaction_calculator = Mock()
    starting_portfolio = Mock()
    length = Mock()
    return (
        strategies,
        datasets,
        slippage_calculator,
        transaction_calculator,
        starting_portfolio,
        length,
    )


def test_save() -> None:
    """Result save method called."""
    result = Mock()
    output_folder = "test_folder/"
    i = 0
    orc._save(result, output_folder, i)
    fp = os.path.join(output_folder, f"{i}_backtest_result.pkl")
    result.save.called_once_with(fp)


def test_dont_save() -> None:
    """Result save method not called."""
    result = Mock()
    output_folder = None
    i = 0
    orc._save(result, output_folder, i)
    result.save.not_called()


def test_run_backtest(mocker: MockerFixture) -> None:
    """Build composer and simulation, then return simulation results."""
    mocker.patch("coin_test.orchestration.orchestration.Composer")
    mock_composer = Mock()
    cast(Mock, orc.Composer).return_value = mock_composer

    mocker.patch("coin_test.orchestration.orchestration.Simulator")
    mock_return = Mock()
    mock_simulator = Mock()
    mock_simulator.run.return_value = mock_return
    cast(Mock, orc.Simulator).return_value = mock_simulator

    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()
    result = orc._run_backtest(datasets[0], strategies[0], sc, tc, portfolio, length)

    cast(Mock, orc.Composer).assert_called_once_with(datasets[0], length)
    cast(Mock, orc.Simulator).assert_called_once_with(
        mock_composer,
        portfolio,
        strategies[0],
        sc,
        tc,
    )
    mock_simulator.run.assert_called_once_with()
    assert result == mock_return


def test_sim_param_generator() -> None:
    """Return all combinations."""
    strategies, datasets, _, _, _, _ = _build_backtest_arg_mocks()

    correct_combinations = [
        (0, datasets[0], strategies[0]),
        (1, datasets[0], strategies[1]),
        (2, datasets[0], strategies[2]),
        (3, datasets[1], strategies[0]),
        (4, datasets[1], strategies[1]),
        (5, datasets[1], strategies[2]),
    ]
    combinations = list(orc._sim_param_generator(datasets, strategies))
    assert combinations == correct_combinations


def test_run_agent(mocker: MockerFixture) -> None:
    """Run backtest and receive / send messages."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()

    receiver = MagicMock()
    sender = MagicMock()

    messages = [(0, datasets[0], strategies[0]), (0, datasets[1], strategies[1]), None]
    receiver.get.side_effect = messages

    mock_results = [Mock(), Mock()]
    mocker.patch("coin_test.orchestration.orchestration._run_backtest")
    orc._run_backtest.side_effect = mock_results

    orc._run_agent(receiver, sender, sc, tc, portfolio, length)

    assert receiver.get.call_args_list is not None
    assert len(receiver.get.call_args_list) == 3

    calls = [call(msg[1], msg[2], sc, tc, portfolio, length) for msg in messages[:-1]]
    orc._run_backtest.assert_has_calls(calls)
    assert len(orc._run_backtest.call_args_list) == 2

    calls = [
        call((msg[0], result))
        for msg, result in zip(messages[:-1], mock_results, strict=True)
    ]
    sender.put.assert_has_calls(calls)
    assert len(sender.put.call_args_list) == 2


def test_run_agent_error(mocker: MockerFixture) -> None:
    """Send error on exception raised."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()

    receiver = MagicMock()
    sender = MagicMock()

    receiver.get.return_value = (0, datasets[0], strategies[0])

    test_exception = Exception("Test exception")
    mocker.patch("coin_test.orchestration.orchestration._run_backtest")
    orc._run_backtest.side_effect = test_exception

    orc._run_agent(receiver, sender, sc, tc, portfolio, length)

    sender.put.assert_called_once_with(test_exception)


def test_gen_multiprocessed(mocker: MockerFixture) -> None:
    """Spawn processes and handle messages."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()

    n_parallel = 4
    backtests = 6
    output_folder = "test_folder/"

    mock_main_to_worker = Mock()
    mock_worker_to_main = Mock()
    mocker.patch("coin_test.orchestration.orchestration.Queue")
    cast(Mock, orc.Queue).side_effect = [mock_main_to_worker, mock_worker_to_main]

    mock_ctx = Mock()
    mocker.patch("coin_test.orchestration.orchestration.multiprocessing.get_context")
    orc.multiprocessing.get_context.return_value = mock_ctx

    processes = [Mock() for _ in range(n_parallel)]
    mock_ctx.Process.side_effect = processes

    mock_sim_params = [Mock() for _ in range(backtests)]
    mocker.patch("coin_test.orchestration.orchestration._sim_param_generator")
    orc._sim_param_generator.return_value = mock_sim_params

    worker_to_main_msgs = [(i, Mock()) for i in range(backtests)]
    mock_worker_to_main.get.side_effect = worker_to_main_msgs

    mocker.patch("coin_test.orchestration.orchestration._save")

    results = orc._gen_multiprocessed(
        datasets, strategies, sc, tc, portfolio, length, n_parallel, output_folder
    )

    orc.multiprocessing.get_context.assert_called_once_with("fork")
    assert len(mock_ctx.Process.call_args_list) == n_parallel
    process_calls = [
        call(
            target=orc._run_agent,
            args=(
                mock_main_to_worker,
                mock_worker_to_main,
                sc,
                tc,
                portfolio,
                length,
            ),
            daemon=True,
        )
        for _ in range(n_parallel)
    ]
    mock_ctx.Process.assert_has_calls(process_calls)

    for process in processes:
        process.start.assert_called_once()
        process.join.assert_called_once()

    orc._sim_param_generator.assert_called_once_with(datasets, strategies)

    assert len(mock_main_to_worker.put.call_args_list) == n_parallel + backtests
    put_calls = [call(p) for p in mock_sim_params] + [
        call(None) for _ in range(n_parallel)
    ]
    mock_main_to_worker.put.has_calls(put_calls)
    assert len(mock_worker_to_main.get.call_args_list) == backtests

    assert len(orc._save.call_args_list) == backtests
    save_calls = [call(msg[1], output_folder, msg[0]) for msg in worker_to_main_msgs]
    assert orc._save.has_calls(save_calls)

    assert results == [msg[1] for msg in worker_to_main_msgs]


def test_gen_multiprocessed_error(mocker: MockerFixture) -> None:
    """Raise exception when child processes raises exception."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()

    n_parallel = 4
    backtests = 6
    output_folder = "test_folder/"

    mock_main_to_worker = Mock()
    mock_worker_to_main = Mock()
    mocker.patch("coin_test.orchestration.orchestration.Queue")
    cast(Mock, orc.Queue).side_effect = [mock_main_to_worker, mock_worker_to_main]

    mocker.patch("coin_test.orchestration.orchestration.multiprocessing.get_context")

    mock_sim_params = [Mock() for _ in range(backtests)]
    mocker.patch("coin_test.orchestration.orchestration._sim_param_generator")
    orc._sim_param_generator.return_value = mock_sim_params

    test_exception = Exception("Test error")
    mock_worker_to_main.get.return_value = test_exception

    with pytest.raises(Exception) as e:
        orc._gen_multiprocessed(
            datasets, strategies, sc, tc, portfolio, length, n_parallel, output_folder
        )
    assert e.value == test_exception


def test_gen_serial(mocker: MockerFixture) -> None:
    """Run backtests in sequence."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()
    output_folder = "test_folder/"

    mocker.patch("coin_test.orchestration.orchestration._sim_param_generator")
    sim_params = [
        (0, datasets[0], strategies[0]),
        (1, datasets[0], strategies[1]),
        (2, datasets[0], strategies[2]),
        (3, datasets[1], strategies[0]),
        (4, datasets[1], strategies[1]),
        (5, datasets[1], strategies[2]),
    ]
    orc._sim_param_generator.return_value = sim_params

    mocker.patch("coin_test.orchestration.orchestration._run_backtest")
    mock_results = [Mock() for _ in sim_params]
    orc._run_backtest.side_effect = mock_results

    mocker.patch("coin_test.orchestration.orchestration._save")

    results = orc._gen_serial(
        datasets, strategies, sc, tc, portfolio, length, output_folder
    )

    orc._sim_param_generator.called_once_with(datasets, strategies)

    assert len(orc._run_backtest.call_args_list) == len(sim_params)
    backtest_args = [
        call(params[1], params[2], sc, tc, portfolio, length) for params in sim_params
    ]
    orc._run_backtest.assert_has_calls(backtest_args)

    assert len(orc._save.call_args_list) == len(sim_params)
    save_args = [
        call(result, output_folder, params[0])
        for result, params in zip(mock_results, sim_params, strict=True)
    ]
    orc._save.assert_has_calls(save_args)

    assert results == mock_results


def test_gen_results_serial(mocker: MockerFixture) -> None:
    """Run backtests in sequence."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()
    n_parallel = 1
    output_folder = "test_folder/"

    mock_results = [Mock()]
    mocker.patch("coin_test.orchestration.orchestration._gen_serial")
    orc._gen_serial.return_value = mock_results

    results = orc._gen_results(
        datasets, strategies, sc, tc, portfolio, length, n_parallel, output_folder
    )

    orc._gen_serial.assert_called_once_with(
        datasets, strategies, sc, tc, portfolio, length, output_folder
    )

    assert results == mock_results


def test_gen_results_multiprocessed(mocker: MockerFixture) -> None:
    """Run backtests in parallel."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()
    n_parallel = 4
    output_folder = "test_folder/"

    mock_results = [Mock()]
    mocker.patch("coin_test.orchestration.orchestration._gen_multiprocessed")
    orc._gen_multiprocessed.return_value = mock_results

    results = orc._gen_results(
        datasets, strategies, sc, tc, portfolio, length, n_parallel, output_folder
    )

    orc._gen_multiprocessed.assert_called_once_with(
        datasets, strategies, sc, tc, portfolio, length, n_parallel, output_folder
    )

    assert results == mock_results


def test_run(mocker: MockerFixture) -> None:
    """Run backtests in parallel."""
    strategies, datasets, sc, tc, portfolio, length = _build_backtest_arg_mocks()
    n_parallel = 4
    output_folder = "test_folder/"

    mock_results = [Mock()]
    mocker.patch("coin_test.orchestration.orchestration._gen_results")
    orc._gen_results.return_value = mock_results
    mocker.patch("coin_test.orchestration.orchestration.build_datapane")

    results = orc.run(
        datasets,
        strategies,
        portfolio,
        length,
        n_parallel,
        output_folder,
        sc,
        tc,
    )

    orc._gen_results.assert_called_once_with(
        datasets,
        strategies,
        sc,
        tc,
        portfolio,
        length,
        n_parallel,
        output_folder,
    )

    assert results == mock_results


def test_run_defaults(mocker: MockerFixture) -> None:
    """Run backtests with defulat slippage and tx calculators."""
    strategies, datasets, _, _, portfolio, length = _build_backtest_arg_mocks()
    n_parallel = 4
    output_folder = "test_folder/"

    mock_results = [Mock()]
    mocker.patch("coin_test.orchestration.orchestration._gen_results")
    orc._gen_results.return_value = mock_results

    mocker.patch("coin_test.orchestration.orchestration.ConstantSlippage")
    sc = Mock()
    cast(Mock, orc.ConstantSlippage).return_value = sc

    mocker.patch(
        "coin_test.orchestration.orchestration.ConstantTransactionFeeCalculator"
    )
    tc = Mock()
    cast(Mock, orc.ConstantTransactionFeeCalculator).return_value = tc

    results = orc.run(
        datasets,
        strategies,
        portfolio,
        length,
        n_parallel,
        output_folder,
    )

    orc._gen_results.assert_called_once_with(
        datasets,
        strategies,
        sc,
        tc,
        portfolio,
        length,
        n_parallel,
        output_folder,
    )

    assert results == mock_results
    orc.build_datapane.assert_called_once_with(mock_results)
