import importlib

import numpy as np
import pytest


@pytest.mark.parametrize(
    "module", ["ibm.ibm_default", "ibm.ibm_qaoa_cpu_noiseless", "ibm.ibm_pce_cpu_noiseless", "dwave.dwave_sa"]
)
def test_cpu_simulator_returns_valid_energy(module, make_qubo, assert_solution):
    solve = importlib.import_module(f"qsplit.adapters.{module}").solve
    qubo = make_qubo([[-1, 2], [0, -2]], ids=[7, 11], offset=3)
    assert_solution(qubo, solve(qubo))


@pytest.mark.parametrize(
    ("module", "limit"),
    [("ibm.ibm_qaoa_cpu_noiseless", 2), ("ibm.ibm_pce_cpu_noiseless", 3)],
)
def test_local_training_limit_preserves_full_solution(module, limit, make_qubo, assert_solution, tmp_path):
    solve = importlib.import_module(f"qsplit.adapters.{module}").solve
    qubo = make_qubo(np.diag([-1, -2, -3, -4, -5]), ids=[7, 11, 15, 19, 23], offset=3)
    config = tmp_path / "training.yaml"
    config.write_text(f"LOCAL_TRAINING_QUBITS: {limit}\n")
    assert_solution(qubo, solve(qubo, config=config))


@pytest.mark.parametrize("matrix", [[[1, -3], [0, 1]], [[2, -4, 1], [0, -1, 3], [0, 0, -2]]])
def test_pce_trains_a_nonconstant_loss_and_solves_small_qubos(
    matrix, make_qubo, exact_solver, assert_solution, monkeypatch
):
    from qiskit_aer import AerSimulator

    from qsplit.adapters.ibm import __ibm_pce as pce

    optimize = pce.minimize
    training = []

    def record(fun, start, **kwargs):
        initial_loss = fun(start)
        result = optimize(fun, start, **kwargs)
        training.append((initial_loss, result.fun))
        return result

    monkeypatch.setattr(pce, "minimize", record)
    qubo = make_qubo(matrix, ids=[7, 11, 19][: len(matrix)], offset=5)
    result = pce.ibm_solve(qubo, AerSimulator(seed_simulator=42))
    assert_solution(qubo, result)
    assert result.energy.min() == pytest.approx(exact_solver(qubo).energy.min())
    assert len(training) == 2
    assert min(loss for _, loss in training) < min(initial for initial, _ in training) - 1e-3


def test_pce_simulator_handles_padding_and_distinct_row_column_ids(assert_solution, exact_solver):
    from qiskit_aer import AerSimulator

    from qsplit.adapters.ibm import __ibm_pce as pce
    from qsplit.qubo import QUBO

    qubo = QUBO(
        np.array([[2, -3, 4], [0, -2, -5], [0, 0, 1.0]]),
        np.array([7, -1, 19]),
        np.array([-2, 7, 5]),
        offset=5,
    )
    result = pce.ibm_solve(qubo, AerSimulator(seed_simulator=42))
    assert_solution(qubo, result)
    assert result.energy.min() == pytest.approx(exact_solver(qubo).energy.min())


@pytest.mark.quantinuum
def test_quantinuum_cpu_simulator(make_qubo, assert_solution):
    pytest.importorskip("pytket.extensions.qiskit")
    from qsplit.adapters.quantinuum.tket_qaoa_cpu_noiseless import solve

    qubo = make_qubo([[-1, 2], [0, -2]], ids=[7, 11], offset=3)
    assert_solution(qubo, solve(qubo))


@pytest.mark.cudaq
def test_cudaq_cpu_simulator(make_qubo, assert_solution, monkeypatch):
    pytest.importorskip("cudaq")
    from qsplit.adapters.nvidia import cudaq_qaoa

    monkeypatch.setattr(cudaq_qaoa, "TARGET_SEQUENCE", [("qpp-cpu", "")])
    solve = cudaq_qaoa.solve
    qubo = make_qubo([[-1, 2], [0, -2]], ids=[7, 11], offset=3)
    assert_solution(qubo, solve(qubo))


def test_dwave_bqm_matches_original_energy_for_all_assignments(make_qubo):
    from itertools import product

    from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm

    qubo = make_qubo([[2, -3, 4], [0, 1, -2], [0, 0, -1]], offset=5)
    bqm = from_qubo_matrix_to_bqm(qubo)
    for bits in product((0, 1), repeat=3):
        x = np.array(bits)
        assert bqm.energy(dict(enumerate(bits))) == pytest.approx(x @ qubo.mat @ x + 5)


def test_ibm_cpu_solve(make_qubo):
    from qsplit.adapters.ibm.ibm_qaoa_cpu_noiseless import solve

    qubo = make_qubo([[0, 1], [0, 0]], ids=[1, 2])
    solutions = solve(qubo)
    assert solutions["energy"].min() == 0
    expected_solutions = {(1, 0), (0, 1), (0, 0)}
    actual_solutions = set(solutions[[1, 2]].itertuples(index=False, name=None))
    assert actual_solutions <= expected_solutions


def test_solve_simulated_annealing(make_qubo):
    import pandas as pd

    from qsplit.adapters.dwave.dwave_sa import solve

    qubo = make_qubo([[1, -2], [0, 1]], offset=5)
    solutions = solve(qubo)
    assert isinstance(solutions, pd.DataFrame)
    assert not solutions.empty
    assert {0, 1, "energy"} <= set(solutions.columns)
