import runpy
import sys
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import generate_preset_pass_manager
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_aer import AerSimulator

from qsplit import configuration
from qsplit.adapters.ibm import __ibm_pce as pce
from qsplit.adapters.ibm import __ibm_qaoa as ibm
from qsplit.adapters.ibm import util_qaoa
from qsplit.adapters.training import local_training_qubits, training_subproblem
from qsplit.qubo import QUBO

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_dense_evolution(monkeypatch):
    monkeypatch.setattr(PauliEvolutionGate, "to_matrix", Mock(side_effect=AssertionError("Dense evolution matrix")))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "bad", float("inf")])
def test_invalid_training_limit(value):
    with pytest.raises(ValueError, match="LOCAL_TRAINING_QUBITS must be a positive integer"):
        configuration.load({"LOCAL_TRAINING_QUBITS": value})


def test_default_and_integral_training_limit():
    assert local_training_qubits() == 20
    with configuration.use({"LOCAL_TRAINING_QUBITS": 25.0}):
        assert local_training_qubits() == 25


def test_induced_subproblem_preserves_energy_and_arbitrary_ids():
    qubo = QUBO(np.array([[1, 8, 3], [0, -2, -9], [0, 0, 4]]), np.array([40, 10, 70]), np.array([70, 20, 30]), 5)
    original = qubo.mat.copy()
    with configuration.use({"LOCAL_TRAINING_QUBITS": 3}):
        sub = training_subproblem(qubo)
        np.testing.assert_array_equal(sub.rows_idx, [10, 30, 40])
        for bits in product((0, 1), repeat=3):
            assignment = dict(zip(sub.rows_idx, bits))
            rows = np.array([assignment.get(i, 0) for i in qubo.rows_idx])
            cols = np.array([assignment.get(i, 0) for i in qubo.cols_idx])
            assert np.array(bits) @ sub.mat @ bits + sub.offset == rows @ qubo.mat @ cols + qubo.offset
    np.testing.assert_array_equal(qubo.mat, original)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 8}):
        assert training_subproblem(qubo) is qubo


def test_pce_limit_counts_encoded_qubits_and_auxiliary_node(make_qubo):
    qubo = make_qubo(np.diag(np.arange(1, 14)))
    with configuration.use({"LOCAL_TRAINING_QUBITS": 4}):
        sub = training_subproblem(qubo, pce_k=3)
        assert sub.problem_size == 11
        assert pce._build_pce_problem(sub)[0].num_qubits == 4
        assert pce._build_pce_problem(qubo)[0].num_qubits == 5
    with configuration.use({"LOCAL_TRAINING_QUBITS": 2}):
        with pytest.raises(ValueError, match="PCE training requires"):
            training_subproblem(qubo, pce_k=3)


def test_ibm_qaoa_trains_locally_and_submits_full_circuit_once(make_qubo, monkeypatch, assert_solution):
    qubo = make_qubo(np.diag([-1, -2, -3, -4]), ids=[10, 20, 30, 40])
    learned = np.array([0.1, 0.2, 0.3, 0.4])
    estimator = StatevectorEstimator()
    training_calls = []

    def local_run(pubs):
        training_calls.extend(pubs)
        assert all(pub[0].num_qubits <= 2 for pub in pubs)
        return estimator.run(pubs)

    monkeypatch.setattr(util_qaoa, "StatevectorEstimator", lambda: SimpleNamespace(run=local_run))

    def optimize(fun, x0):
        assert np.isfinite(fun(x0))
        assert np.isfinite(fun(learned))
        return SimpleNamespace(x=learned)

    monkeypatch.setattr(util_qaoa, "SPSA", lambda: SimpleNamespace(minimize=optimize))
    hardware = Mock(spec=util_qaoa.IBMBackend)
    pm = Mock(side_effect=None)
    pm.run.side_effect = lambda circuit: circuit
    monkeypatch.setattr(ibm, "generate_preset_pass_manager", Mock(return_value=pm))
    sampler = Mock()
    sampler.run.return_value.result.return_value = [
        SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_int_counts=lambda: {15: 500})))
    ]
    runtime_sampler = Mock(return_value=sampler)
    monkeypatch.setattr(util_qaoa, "RuntimeSamplerV2", runtime_sampler)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 2}):
        assert_solution(qubo, ibm.ibm_solve(qubo, hardware))
    assert len(training_calls) == 2
    runtime_sampler.assert_called_once_with(mode=hardware)
    sampler.run.assert_called_once()
    final = sampler.run.call_args.args[0][0][0]
    expected = util_qaoa.__from_qubo_matrix_to_circuit(qubo)[0].assign_parameters(learned)
    expected.measure_all()
    assert final == expected and final.num_qubits == 4
    hardware.run.assert_not_called()


def test_constant_training_subproblem_still_binds_full_qaoa(make_qubo, monkeypatch):
    qubo = make_qubo([[0, 1], [0, 0]])

    optimizer = Mock(side_effect=AssertionError("Constant objective needs no optimizer"))
    monkeypatch.setattr(util_qaoa, "SPSA", optimizer)
    backend = AerSimulator()
    with configuration.use({"LOCAL_TRAINING_QUBITS": 1}):
        circ, _, _ = util_qaoa.get_qaoa_circuit_optimized(backend, generate_preset_pass_manager(backend=backend), qubo)
    assert circ.num_qubits == 2 and circ.num_parameters == 0
    optimizer.assert_not_called()


def test_pce_trains_with_pce_and_evaluates_full_problem_once(make_qubo, monkeypatch, assert_solution):
    qubo = make_qubo(np.diag([-1, -2, -3, -4, -5]))
    learned = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    local = StatevectorEstimator()
    local_calls = []

    def local_run(pubs):
        local_calls.append(pubs)
        for circuit, ops, _ in pubs:
            assert circuit.num_qubits == 3
            assert all(sum(c != "I" for c in op.paulis.to_labels()[0]) == 3 for op in ops)
        return local.run(pubs)

    monkeypatch.setattr(pce, "StatevectorEstimator", lambda: SimpleNamespace(run=local_run))

    def optimize(fun, x0, **kwargs):
        assert np.isfinite(fun(x0))
        loss = fun(learned)
        assert np.isfinite(loss)
        assert kwargs["options"]["maxiter"] >= 100
        return SimpleNamespace(x=learned, fun=loss)

    monkeypatch.setattr(pce, "minimize", optimize)
    hardware = Mock(spec=pce.IBMBackend)
    monkeypatch.setattr(pce, "generate_preset_pass_manager", lambda **kwargs: generate_preset_pass_manager())
    final_calls = []

    def final_run(pubs):
        final_calls.append(pubs)
        assert len(pubs) == 3
        assert sum(len(pub[1]) for pub in pubs) == 6
        assert all(pub[0].num_qubits == 4 and pub[0].num_parameters == 0 for pub in pubs)
        result = local.run([(c.decompose(reps=10), ops, params) for c, ops, params in pubs]).result()
        for pub_result in result:
            pub_result.metadata["target_precision"] = 0.02
        return SimpleNamespace(result=lambda: result)

    runtime = Mock(return_value=SimpleNamespace(run=final_run))
    decode = Mock(wraps=pce._decode_solution)
    monkeypatch.setattr(pce, "_decode_solution", decode)
    monkeypatch.setattr(pce, "EstimatorV2", runtime)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 3}):
        assert_solution(qubo, pce.ibm_solve(qubo, hardware))
    assert len(local_calls) == 4 and len(final_calls) == 1
    assert decode.call_args.kwargs["atol"] == pytest.approx(0.04)
    runtime.assert_called_once_with(mode=hardware)
    expected = pce._build_pce_problem(qubo)[0].assign_parameters(learned)
    assert final_calls[0][0][0] == generate_preset_pass_manager().run(expected)


def test_constant_induced_pce_problem_skips_regularizer_only_training(make_qubo, monkeypatch, assert_solution):
    matrix = np.zeros((8, 8))
    matrix[0, 2:5] = 10
    matrix[1, 5:8] = 9
    qubo = make_qubo(matrix)
    optimizer = Mock(side_effect=AssertionError("Training subproblem has no couplings"))
    monkeypatch.setattr(pce, "minimize", optimizer)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 3}):
        sub = training_subproblem(qubo, pce_k=3)
        assert not sub.mat.any()
        result = pce.ibm_solve(qubo, AerSimulator(seed_simulator=42))
    assert_solution(qubo, result)
    optimizer.assert_not_called()


def test_pce_decode_preserves_bits_above_64():
    Q = np.zeros((70, 70))
    expectations = dict.fromkeys(range(71), 1)
    expectations[69] = -1
    assert pce._decode_solution(Q, expectations) == {1 << 69: 1}


def test_iqm_training_never_runs_on_device(make_qubo, monkeypatch):
    provider = ModuleType("iqm.qiskit_iqm")
    provider.IQMProvider = Mock()
    provider.transpile_to_IQM = Mock(side_effect=lambda circ, backend: circ)
    monkeypatch.setitem(sys.modules, "iqm", ModuleType("iqm"))
    monkeypatch.setitem(sys.modules, "iqm.qiskit_iqm", provider)
    module = runpy.run_path(str(ROOT / "qsplit/adapters/iqm/util.py"))
    build = module["get_qaoa_circuit_optimized"]
    learned = np.array([0.1, 0.2, 0.3, 0.4])
    sampler = StatevectorSampler(seed=0)
    training_calls = []

    def local_run(pubs, **kwargs):
        training_calls.extend(pubs)
        assert all(circuit.num_qubits == 2 for circuit in pubs)
        return sampler.run(pubs, **kwargs)

    def optimize(fun, x0, **kwargs):
        assert np.isfinite(fun(x0))
        assert np.isfinite(fun(learned))
        return SimpleNamespace(x=learned)

    monkeypatch.setitem(build.__globals__, "minimize", optimize)
    monkeypatch.setitem(build.__globals__, "StatevectorSampler", lambda: SimpleNamespace(run=local_run))
    backend = Mock()
    qubo = make_qubo(np.diag([-1, -2, -3, -4]))
    with configuration.use({"LOCAL_TRAINING_QUBITS": 2}):
        circuit, _, _ = build(backend, qubo)
    backend.run.assert_not_called()
    assert len(training_calls) == 2
    expected = module["__from_qubo_matrix_to_circuit"](qubo)[0].assign_parameters(learned)
    assert circuit == expected
    provider.transpile_to_IQM.assert_called_once_with(expected, backend=backend)
    module["run_quantum_optimizer"](backend, circuit)
    backend.run.assert_called_once_with(circuit, shots=500)


@pytest.mark.quantinuum
def test_quantinuum_only_final_execution_uses_nexus(make_qubo, monkeypatch):
    pytest.importorskip("pytket.extensions.qiskit")
    from qsplit.adapters.quantinuum import __tket_qaoa as adapter
    from qsplit.adapters.quantinuum import util

    qubo = make_qubo(np.diag([-1, -2, -3, -4]))
    local_eval = util.get_operator_expectation_value
    calls = []
    learned = np.array([0.1, 0.2, 0.3, 0.4])
    for attr in ["compile", "execute"]:
        monkeypatch.setattr(util.qnx, attr, Mock())
    monkeypatch.setattr(util.qnx.circuits, "upload", Mock())
    monkeypatch.setattr(util.qnx.projects, "get_or_create", Mock())
    util.qnx.execute.return_value = [SimpleNamespace(get_counts=lambda: {(1, 1, 1, 1): 100})]

    def evaluate(circ, operator, backend):
        assert circ.n_qubits == 2
        util.qnx.execute.assert_not_called()
        calls.append(circ)
        return local_eval(circ, operator, backend)

    def optimize(fun, x0, **kwargs):
        assert np.isfinite(fun(x0))
        return SimpleNamespace(x=learned)

    monkeypatch.setattr(util, "get_operator_expectation_value", evaluate)
    monkeypatch.setattr(util, "minimize", optimize)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 2, "QNEXUS_QPU": "H2-1E"}):
        adapter.__tket_solve(qubo, None)
    assert len(calls) == 1
    util.qnx.execute.assert_called_once()
    util.qnx.circuits.upload.assert_called_once()
    final = util.qnx.circuits.upload.call_args.args[0]
    expected = util._qaoa_circuit(qubo, 2)[0]
    expected.symbol_substitution(dict(zip(sorted(expected.free_symbols(), key=str), learned)))
    expected.measure_all()
    assert final == expected and final.n_qubits == 4


@pytest.mark.quantinuum
def test_constant_subproblem_still_binds_full_quantinuum_circuit(make_qubo, monkeypatch):
    pytest.importorskip("pytket.extensions.qiskit")
    from qsplit.adapters.quantinuum import util

    optimizer = Mock(side_effect=AssertionError("Constant objective needs no optimizer"))
    monkeypatch.setattr(util, "minimize", optimizer)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 1}):
        circuit, _, _ = util.get_qaoa_circuit_optimized(make_qubo([[0, 1], [0, 0]]), None)
    assert circuit.n_qubits == 2 and not circuit.free_symbols()
    optimizer.assert_not_called()


def test_cudaq_trains_subproblem_and_samples_full_circuit(make_qubo, monkeypatch):
    cudaq = ModuleType("cudaq")
    cudaq.set_target = Mock()
    util = ModuleType("qsplit.adapters.nvidia.util")
    learned = [0.1, 0.2, 0.3, 0.4]
    util.from_qubo_matrix_to_circuit = Mock(side_effect=lambda q: (q.problem_size, "cost", {}, []))
    util.optimize_circuit = Mock(return_value=learned)
    util.run_quantum_optimizer = Mock(return_value={})
    util.to_dataframe = Mock(return_value="result")
    monkeypatch.setitem(sys.modules, "cudaq", cudaq)
    monkeypatch.setitem(sys.modules, "qsplit.adapters.nvidia.util", util)
    solve = runpy.run_path(str(ROOT / "qsplit/adapters/nvidia/cudaq_qaoa.py"))["solve"]
    qubo = make_qubo(np.diag([-1, -2, -3, -4]))
    assert solve(qubo, config={"LOCAL_TRAINING_QUBITS": 2}) == "result"
    util.optimize_circuit.assert_called_once_with(2, "cost")
    util.run_quantum_optimizer.assert_called_once_with(4, learned)
