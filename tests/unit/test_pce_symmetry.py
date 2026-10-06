from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, Statevector

from qsplit.adapters.ibm import __ibm_pce as pce
from qsplit.qubo import QUBO

TOL = 1e-12


def sample_correlations(circuit, groups, samples=12):
    rng = np.random.default_rng(20261006)
    values = []
    for params in rng.uniform(-np.pi, np.pi, (samples, circuit.num_parameters)):
        state = Statevector.from_instruction(circuit.assign_parameters(params))
        values.append([state.expectation_value(op).real for group in groups for op in group])
    return np.asarray(values)


@pytest.mark.parametrize("n", [1, 2, 5, 11, 29])
def test_every_encoded_variable_can_respond_to_parameters(n, make_qubo):
    circuit, groups, _, _ = pce._build_pce_problem(make_qubo(np.diag(-np.arange(1, n + 1))))
    parity = Pauli("X" * circuit.num_qubits)
    state = Statevector.from_instruction(circuit.assign_parameters(np.arange(6) * 0.17 + 0.13))
    assert abs(state.expectation_value(parity).real - 1) > 1e-3
    values = sample_correlations(circuit, groups)
    variation = np.ptp(values, axis=0)
    print(f"variables={n}, qubits={circuit.num_qubits}, smallest observable variation={variation.min():.3f}")
    assert np.all(variation > 1e-3)
    assert circuit.num_parameters == 6


def test_chain_reflection_does_not_tie_distinct_x_variables(make_qubo):
    circuit, groups, _, _ = pce._build_pce_problem(make_qubo(np.eye(11)))
    labels = [op.paulis.to_labels()[0] for op in groups[0]]
    values = sample_correlations(circuit, groups)
    pairs = [(i, labels.index(label[::-1])) for i, label in enumerate(labels)]
    assert any(i != j for i, j in pairs)
    for i, j in pairs:
        if i != j:
            assert np.max(np.abs(values[:, i] - values[:, j])) > 1e-3


def test_three_qubit_loss_is_not_constant(make_qubo):
    circuit, groups, edges, Q = pce._build_pce_problem(make_qubo([[1, -3], [0, 1]]))
    estimator = StatevectorEstimator()
    rng = np.random.default_rng(42)
    losses = [
        pce.__pce_loss(params, circuit, groups, estimator, edges, len(Q) + 1, circuit.num_qubits)["loss"]
        for params in rng.uniform(-np.pi, np.pi, (6, circuit.num_parameters))
    ]
    assert np.ptp(losses) > 1e-3


def test_diagonal_qubo_weights_affect_training_loss(make_qubo):
    estimator = StatevectorEstimator()
    params = np.arange(6) * 0.17 + 0.13
    losses = []
    for diagonal in [-np.arange(1, 12), np.arange(1, 12) ** 2]:
        circuit, groups, edges, Q = pce._build_pce_problem(make_qubo(np.diag(diagonal)))
        result = pce.__pce_loss(params, circuit, groups, estimator, edges, len(Q) + 1, circuit.num_qubits)
        assert all(v == len(Q) for _, v in edges)
        assert abs(result["exp_map"][len(Q)]) > 1e-3
        losses.append(result["loss"])
    assert abs(losses[0] - losses[1]) > 1e-3


def test_positive_rescaling_preserves_loss(make_qubo):
    estimator = StatevectorEstimator()
    params = np.arange(6) * 0.17 + 0.13
    losses = []
    for scale in [1e-4, 1, 1e4]:
        qubo = make_qubo(scale * np.array([[1, -3, 4], [0, 2, -5], [0, 0, -2]]))
        circuit, groups, edges, Q = pce._build_pce_problem(qubo)
        losses.append(pce.__pce_loss(params, circuit, groups, estimator, edges, len(Q) + 1, circuit.num_qubits)["loss"])
    np.testing.assert_allclose(losses, losses[0], atol=TOL, rtol=0)


def test_uncertain_signs_are_stable_and_try_both_auxiliary_orientations():
    Q = np.array([[1.0, -3.0], [0.0, 1.0]])
    assert pce._decode_solution(Q, {0: 1.0, 1: 0.0, 2: 0.0}) == {3: 1}
    for y, auxiliary in product([-1e-15, 1e-15], repeat=2):
        assert pce._decode_solution(Q, {0: 1.0, 1: y, 2: auxiliary}) == {3: 1}
    assert pce._decode_solution(Q, {0: 1.0, 1: 0.02, 2: -0.02}, atol=0.03) == {3: 1}


def test_polishing_respects_small_coefficients_and_preserves_reliable_signs():
    Q = np.diag([-1e-10, 2e-10])
    assert pce._decode_solution(Q, {0: 1, 1: 1, 2: 1}) == {1: 1}
    zero = np.zeros((2, 2))
    assert pce._decode_solution(zero, {0: -1, 1: 1, 2: 1}) == {1: 1}


def test_polishing_improves_energy_and_ends_at_a_one_bit_local_minimum():
    rng = np.random.default_rng(42)
    Q = np.triu(rng.normal(size=(8, 8)))
    for bits in rng.integers(0, 2, size=(12, 8)):
        before = bits @ Q @ bits
        x = pce._polish_solution(Q, bits.copy())
        energy = x @ Q @ x
        assert energy <= before + TOL
        for i in range(len(x)):
            flipped = x.copy()
            flipped[i] = 1 - flipped[i]
            assert flipped @ Q @ flipped >= energy - TOL


@pytest.mark.parametrize("layout", ["aligned", "disjoint", "padding", "repeated"])
def test_qubo_to_auxiliary_ising_preserves_all_binary_energies(layout, make_qubo):
    if layout == "aligned":
        qubo = make_qubo([[2, -3, 4], [1, -2, -5], [2, 3, 1]], ids=[7, 11, 19], offset=5)
    else:
        rows, cols = {
            "disjoint": ([7, 11, 19], [3, 7, 5]),
            "padding": ([7, -1, 19], [-2, 7, 5]),
            "repeated": ([7, 7, 19], [3, 7, 3]),
        }[layout]
        qubo = QUBO(np.array([[2, -3, 4], [0, -2, -5], [0, 0, 1.0]]), np.array(rows), np.array(cols), 5)
    _, _, edges, Q = pce._build_pce_problem(qubo)
    mapping, variables = pce._variables_mapping(qubo)
    constant = qubo.offset + np.trace(Q) / 2 + np.triu(Q, 1).sum() / 4
    for bits in product((0, 1), repeat=len(variables)):
        x = np.asarray(bits)
        assignment = dict(zip(variables, bits))
        rows = np.array([assignment.get(var, 0) for var in qubo.rows_idx])
        cols = np.array([assignment.get(var, 0) for var in qubo.cols_idx])
        expected = rows @ qubo.mat @ cols + qubo.offset
        assert x @ Q @ x + qubo.offset == pytest.approx(expected)
        for auxiliary in [-1, 1]:
            spins = np.append((1 - 2 * x) * auxiliary, auxiliary)
            encoded = constant + sum(weight * spins[u] * spins[v] for (u, v), weight in edges.items())
            assert encoded == pytest.approx(expected)
        state = sum(int(bit) << mapping[var] for var, bit in assignment.items())
        df = pce.to_dataframe({state: 1}, qubo, mapping, variables)
        assert df.energy.iloc[0] == pytest.approx(expected)
        assert set(df.columns) == set(variables) | {"energy"}


def test_general_ansatz_control_has_nonzero_y_and_z(make_qubo):
    circuit, groups, _, _ = pce._build_pce_problem(make_qubo(np.eye(11)))
    control = efficient_su2(circuit.num_qubits, su2_gates=["ry", "rz"], reps=2)
    values = sample_correlations(control, groups)
    assert np.all(np.ptp(values, axis=0) > 1e-3)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_constant_qubo_needs_no_optimizer_or_backend(n, make_qubo, monkeypatch, assert_solution):
    qubo = make_qubo(np.zeros((n, n)), offset=7)
    monkeypatch.setattr(pce, "minimize", Mock(side_effect=AssertionError("Constant objective")))
    backend = Mock(side_effect=AssertionError("Constant objective"))
    result = pce.ibm_solve(qubo, backend)
    assert_solution(qubo, result)
    assert result.energy.iloc[0] == 7
    backend.run.assert_not_called()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_qubo_is_rejected(value, make_qubo):
    qubo = make_qubo([[1]])
    qubo.mat[0, 0] = value
    with pytest.raises(ValueError, match="finite QUBO"):
        pce._build_pce_problem(qubo)
    with pytest.raises(ValueError, match="finite expectation"):
        pce._decode_solution(np.eye(1), {0: value, 1: 1})
