"""Result parsing and transpilation recovery using real circuits and fake jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_aer import AerSimulator

from qsplit.adapters.ibm import util, util_qaoa


@pytest.mark.parametrize("register", ["meas", "c", "custom"])
def test_sampler_classical_register_compatibility(register, monkeypatch):
    counts = {0: 20, 1: 480}
    data = SimpleNamespace(**{register: SimpleNamespace(get_int_counts=lambda: counts)}, keys=lambda: [register])
    sampler = Mock()
    sampler.run.return_value.result.return_value = [SimpleNamespace(data=data)]
    monkeypatch.setattr(util_qaoa, "BackendSamplerV2", Mock(return_value=sampler))
    circuit = QuantumCircuit(1)
    assert util_qaoa.run_quantum_optimizer(AerSimulator(), circuit) == counts
    sampler.run.assert_called_once_with([(circuit,)], shots=500)


def test_unreadable_sampler_result_fails_explicitly(monkeypatch):
    sampler = Mock()
    sampler.run.return_value.result.return_value = [SimpleNamespace(data=SimpleNamespace())]
    monkeypatch.setattr(util_qaoa, "BackendSamplerV2", Mock(return_value=sampler))
    with pytest.raises(RuntimeError, match="No readable classical register"):
        util_qaoa.run_quantum_optimizer(AerSimulator(), QuantumCircuit(1))


def test_dataframe_offset_and_little_endian_bit_order(make_qubo):
    qubo = make_qubo([[-1, 0], [0, -4]], ids=[10, 20], offset=7)
    result = util.to_dataframe({1: 20, 2: 20}, qubo, {10: 0, 20: 1}, [10, 20])
    assert result.iloc[0].to_dict() == {10: 0.0, 20: 1.0, "energy": 3.0}


@pytest.mark.parametrize("local_optimization", [False, True])
def test_transpilation_failure_recovery(local_optimization, make_qubo, monkeypatch):
    qubo = make_qubo([[-1, 2], [0, -2]])
    manager = Mock()
    manager.run.side_effect = TranspilerError("instruction not in Target")
    optimizer = Mock(
        side_effect=lambda backend, circuit, hamiltonian, **kw: circuit.assign_parameters(
            np.zeros(circuit.num_parameters)
        )
    )
    monkeypatch.setattr(util_qaoa, "__optimize_circuit", optimizer)
    circuit, mapping, ids = util_qaoa.get_qaoa_circuit_optimized(
        AerSimulator(), manager, qubo, optimize_on_backend=not local_optimization
    )
    assert circuit.num_parameters == 0 and circuit.num_clbits == 2
    assert mapping == {0: 0, 1: 1} and ids == [0, 1]
    optimizer.assert_called_once()


def test_unrelated_transpilation_error_is_not_hidden(make_qubo):
    manager = Mock()
    manager.run.side_effect = TranspilerError("invalid topology")
    with pytest.raises(TranspilerError, match="invalid topology"):
        util_qaoa.get_qaoa_circuit_optimized(AerSimulator(), manager, make_qubo([[-1, 2], [0, -2]]))
