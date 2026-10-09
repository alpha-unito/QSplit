from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2, IBMBackend
from scipy.optimize import minimize

from qsplit.adapters.ibm.util import get_variables_mapping, to_dataframe
from qsplit.adapters.training import training_subproblem
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend) -> pd.DataFrame:
    circuit, observables, edges, Q = _build_pce_problem(qubo)
    var_to_qubit, all_vars = _variables_mapping(qubo)
    if not edges:
        return to_dataframe({0: 1}, qubo, var_to_qubit, all_vars)

    variable_ids = np.asarray(all_vars, dtype=int)
    canonical_qubo = QUBO(Q, variable_ids, variable_ids, offset=qubo.offset)
    training_qubo = training_subproblem(canonical_qubo, pce_k=3)
    training_circuit, training_ops, training_edges, training_Q = _build_pce_problem(training_qubo)
    training_circuit = training_circuit.decompose(reps=10)
    estimator = StatevectorEstimator()

    def objective(params):
        return __pce_loss(
            params,
            training_circuit,
            training_ops,
            estimator,
            training_edges,
            len(training_Q) + 1,
            training_circuit.num_qubits,
        )["loss"]

    reps = 3
    delta_t = 0.25
    initial_params = [(1 - i / reps) * delta_t for i in range(1, reps + 1)]
    initial_params += [(i / reps) * delta_t for i in range(1, reps + 1)]
    params = np.asarray(initial_params)
    if training_edges:
        starts = [params, np.random.default_rng(42).uniform(-np.pi, np.pi, len(params))]
        results = [
            minimize(objective, start, method="COBYLA", options={"rhobeg": 0.5, "maxiter": 200}, tol=1e-4)
            for start in starts
        ]
        params = min(results, key=lambda result: result.fun).x

    num_qubits = circuit.num_qubits
    bound = circuit.assign_parameters(params)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    bound = pm.run(bound)
    mapped = [[op.apply_layout(bound.layout) for op in group] for group in observables]
    final_estimator = (
        EstimatorV2(mode=backend) if isinstance(backend, IBMBackend) else BackendEstimatorV2(backend=backend)
    )
    result = __pce_loss([], bound, mapped, final_estimator, edges, len(Q) + 1, num_qubits)
    counts = _decode_solution(Q, result["exp_map"], atol=max(1e-8, 2 * result["precision"]))
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)


def __build_pce(pauli: str, node_list: list, n_qubits: int, k: int) -> list[SparsePauliOp]:
    pauli_correlation_encoding = []
    for idx, c in enumerate(combinations(range(n_qubits), k)):
        if idx >= len(node_list):
            break
        paulis = ["I"] * n_qubits
        for qubit_idx in c:
            paulis[qubit_idx] = pauli
        pauli_correlation_encoding.append(("".join(paulis)[::-1], 1.0))

    hamiltonians = []
    for p_str, weight in pauli_correlation_encoding:
        hamiltonians.append(SparsePauliOp.from_list([(p_str, weight)]))
    return hamiltonians


def __pce_loss(
    x: list[float],
    ansatz: QuantumCircuit,
    hamiltonians: list,
    estimator,
    J_prime: dict,
    num_nodes: int,
    num_qubits: int,
) -> dict[str, float | dict]:
    job = estimator.run([(ansatz, group, x) for group in hamiltonians if group])
    result = job.result()

    node_exp_map = {}
    idx = 0
    for r in result:
        for ev in r.data.evs:
            node_exp_map[idx] = ev
            idx += 1

    weight_scale = max((abs(weight) for weight in J_prime.values()), default=1.0)
    loss_val = 0
    alpha = num_qubits

    for (edge0, edge1), weight in J_prime.items():
        loss_val += (weight / weight_scale) * (
            np.tanh(alpha * node_exp_map[edge0]) * np.tanh(alpha * node_exp_map[edge1])
        )

    regulation_term = 0
    for i in range(num_nodes):
        regulation_term += np.tanh(alpha * node_exp_map[i]) ** 2
    regulation_term = (regulation_term / num_nodes) ** 2

    beta = 1 / 2
    v = sum(abs(weight) / weight_scale for weight in J_prime.values()) / 2 + (num_nodes - 1) / 4
    regulation_term = beta * v * regulation_term

    loss_val += regulation_term

    precision = max((r.metadata.get("target_precision", 0.0) for r in result), default=0.0)
    return {"loss": float(loss_val), "exp_map": node_exp_map, "precision": precision}


def _variables_mapping(qubo: QUBO):
    all_vars = [var for var in get_variables_mapping(qubo)[1] if var >= 0]
    return {var: i for i, var in enumerate(all_vars)}, all_vars


def _build_pce_problem(qubo: QUBO, k: int = 3):
    if not np.isfinite(qubo.mat).all() or not np.isfinite(qubo.offset):
        raise ValueError("PCE requires finite QUBO coefficients and offset")
    var_to_qubit, all_vars = _variables_mapping(qubo)
    n = len(all_vars)
    Q = np.zeros((n, n))
    rows = [i for i, var in enumerate(qubo.rows_idx) if var >= 0]
    cols = [i for i, var in enumerate(qubo.cols_idx) if var >= 0]
    row_indices = np.asarray([var_to_qubit[qubo.rows_idx[i]] for i in rows], dtype=int)
    col_indices = np.asarray([var_to_qubit[qubo.cols_idx[i]] for i in cols], dtype=int)
    np.add.at(Q, (row_indices[:, None], col_indices[None, :]), qubo.mat[np.ix_(rows, cols)])
    Q = np.triu(Q, 1) + np.triu(Q.T, 1) + np.diag(np.diag(Q))
    J_prime = {}

    u_idx, v_idx = np.triu_indices(n, k=1)
    for u, v in zip(u_idx, v_idx):
        if Q[u, v] != 0:
            J_prime[(u, v)] = Q[u, v] / 4.0

    diag_Q = np.diag(Q)
    sum_rows_cols = np.sum(Q, axis=1) + np.sum(Q, axis=0) - 2 * diag_Q
    h = -diag_Q / 2.0 - sum_rows_cols / 4.0

    dummy_index = n
    for u, val in enumerate(h):
        if val != 0:
            J_prime[(u, dummy_index)] = val

    num_nodes = n + 1
    q = k
    while 3 * comb(q, k) < num_nodes:
        q += 1
    num_qubits = q

    list_size = num_nodes // 3
    remainder = num_nodes % 3
    nodes = list(range(num_nodes))
    split_1 = list_size + (1 if remainder > 0 else 0)
    split_2 = split_1 + list_size + (1 if remainder > 1 else 0)

    node_x = nodes[:split_1]
    node_y = nodes[split_1:split_2]
    node_z = nodes[split_2:]

    pce_x = __build_pce("X", node_x, num_qubits, k)
    pce_y = __build_pce("Y", node_y, num_qubits, k)
    pce_z = __build_pce("Z", node_z, num_qubits, k)

    cost_ops = []
    for i in range(num_qubits - 1):
        paulis = ["I"] * num_qubits
        paulis[i] = "Z"
        paulis[i + 1] = "Z"
        cost_ops.append(("".join(paulis)[::-1], 1.0))

    for i in range(num_qubits):
        paulis = ["I"] * num_qubits
        paulis[i] = "Z"
        cost_ops.append(("".join(paulis)[::-1], 0.5 + (i + 1) / (num_qubits + 1)))

    base_cost_op = SparsePauliOp.from_list(cost_ops)
    reps = 3
    qc = qaoa_ansatz(cost_operator=base_cost_op, reps=reps)
    return qc, [pce_x, pce_y, pce_z], J_prime, Q


def _decode_solution(Q, exp_map, *, atol=1e-8):
    n = len(Q)
    dummy_index = n
    best_exp_arr = np.array([exp_map[idx] for idx in range(n + 1)])
    if not np.isfinite(best_exp_arr).all():
        raise ValueError("PCE decoding requires finite expectation values")
    x_raw = np.where(best_exp_arr >= 0, 1, -1)
    uncertain = np.abs(best_exp_arr) <= atol
    auxiliary_signs = (-1, 1) if uncertain[dummy_index] else (x_raw[dummy_index],)
    fillings = (-1, 1) if uncertain[:n].any() else (1,)
    best_x = None
    best_energy = np.inf
    for auxiliary in auxiliary_signs:
        for filling in fillings:
            spins = np.where(uncertain[:n], filling, x_raw[:n])
            x = ((1 - spins * auxiliary) // 2).astype(int)
            x = _polish_solution(Q, x)
            energy = x @ Q @ x
            if energy < best_energy:
                best_x, best_energy = x, energy

    state_int = sum(int(bit) << idx for idx, bit in enumerate(best_x))
    return {state_int: 1}


def _polish_solution(Q, x):
    n = len(Q)
    diag_Q = np.diag(Q)
    H = Q @ x + x @ Q - 2 * diag_Q * x
    tolerance = 1e-12 * np.max(np.abs(Q), initial=0.0)

    improved = True
    while improved:
        improved = False
        for u in range(n):
            delta_z = 1 - 2 * x[u]
            delta_E = (Q[u, u] + H[u]) * delta_z
            if delta_E < -tolerance:
                x[u] = 1 - x[u]
                improved = True
                H += (Q[u, :] + Q[:, u]) * delta_z
                H[u] -= 2 * Q[u, u] * delta_z

    return x
