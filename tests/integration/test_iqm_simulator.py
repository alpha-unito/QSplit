"""IQM circuit/transpilation/optimization on a local noisy fake-device simulator."""

import pytest

pytestmark = pytest.mark.iqm


def test_iqm_fake_device_end_to_end(make_qubo, assert_solution):
    pytest.importorskip("iqm.qiskit_iqm")
    from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

    from qsplit.adapters.iqm import util

    qubo = make_qubo([[-1, 2], [0, -2]], ids=[7, 11], offset=3)
    backend = IQMFakeAdonis()
    circuit, mapping, ids = util.get_qaoa_circuit_optimized(backend, qubo)
    assert circuit.num_parameters == 0
    counts = util.run_quantum_optimizer(backend, circuit)
    assert sum(counts.values()) == 500
    solutions = util.to_dataframe(counts, qubo, mapping, ids)
    assert (solutions["counts"] > 0).all()
    assert_solution(qubo, solutions.drop(columns="counts"))
    assert util.__compute_expectation({}, qubo, mapping) == 0
    assert util.__compute_expectation({"00": 1, "10": 3}, qubo, mapping) == pytest.approx(1.5)
