from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.passmanager import BasePassManager
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from scipy.optimize import minimize

from qsplit.adapters.ibm.util import get_variables_mapping, to_dataframe
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend) -> pd.Dataframe:
    var_to_qubit, all_vars = get_variables_mapping(qubo)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    quantum_results = __run_quantum_optimizer(var_to_qubit, all_vars, qubo, backend, pm, k=3)
    return to_dataframe(quantum_results, qubo, var_to_qubit, all_vars)


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
    job = estimator.run([(ansatz, hamiltonians[0], x), (ansatz, hamiltonians[1], x), (ansatz, hamiltonians[2], x)])
    result = job.result()

    node_exp_map = {}
    idx = 0
    for r in result:
        for ev in r.data.evs:
            node_exp_map[idx] = ev
            idx += 1

    loss_val = 0
    alpha = num_qubits

    for (edge0, edge1), weight in J_prime.items():
        loss_val += weight * np.tanh(alpha * node_exp_map[edge0]) * np.tanh(alpha * node_exp_map[edge1])

    regulation_term = 0
    for i in range(num_nodes):
        regulation_term += np.tanh(alpha * node_exp_map[i]) ** 2
    regulation_term = (regulation_term / num_nodes) ** 2

    beta = 1 / 2
    v = len(J_prime) / 2 + (num_nodes - 1) / 4
    regulation_term = beta * v * regulation_term

    loss_val += regulation_term

    return {"loss": loss_val, "exp_map": node_exp_map}


def __run_quantum_optimizer(
    var_to_qubit, all_vars, qubo: QUBO, backend, pm: BasePassManager, k: int = 3
) -> dict[int, int]:
    var_to_qubit, all_vars = get_variables_mapping(qubo)
    n = len(all_vars)

    Q = np.zeros((n, n))
    for i, row_var in enumerate(qubo.rows_idx):
        for j, col_var in enumerate(qubo.cols_idx):
            Q[var_to_qubit[row_var], var_to_qubit[col_var]] = qubo.mat[i, j]

    J_prime = {}
    for u in range(n):
        for v in range(u + 1, n):
            if Q[u, v] != 0:
                J_prime[(u, v)] = Q[u, v] / 4.0

    dummy_index = n
    for u in range(n):
        h_u = -Q[u, u] / 2.0
        for v in range(n):
            if u < v:
                h_u -= Q[u, v] / 4.0
            elif v < u:
                h_u -= Q[v, u] / 4.0
        if h_u != 0:
            J_prime[(u, dummy_index)] = h_u

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

    base_cost_op = SparsePauliOp.from_list(cost_ops)
    reps = 3
    qc = qaoa_ansatz(cost_operator=base_cost_op, reps=reps)
    qc = pm.run(qc)

    pce_mapped = [
        [op.apply_layout(qc.layout) if getattr(qc, "layout", None) else op for op in pce_x],
        [op.apply_layout(qc.layout) if getattr(qc, "layout", None) else op for op in pce_y],
        [op.apply_layout(qc.layout) if getattr(qc, "layout", None) else op for op in pce_z],
    ]

    estimator = EstimatorV2(mode=backend)

    exp_result = []

    def loss_wrapper(x):
        exp = __pce_loss(x, qc, pce_mapped, estimator, J_prime, num_nodes, num_qubits)
        exp_result.append(exp)
        return exp["loss"]

    delta_t = 0.25
    gamma_list = [(i / reps) * delta_t for i in range(1, reps + 1)]
    beta_list = [(1 - (i / reps)) * delta_t for i in range(1, reps + 1)]
    initial_params = beta_list + gamma_list

    minimize(
        loss_wrapper,
        initial_params,
        method="COBYLA",
        options={"rhobeg": 1.0, "maxiter": len(initial_params) + 2},
        tol=1e-4,
    )

    best_exp_map = min(exp_result, key=lambda val: val["loss"])["exp_map"]

    x_raw = {}
    for idx in range(num_nodes):
        x_raw[idx] = 1 if best_exp_map[idx] >= 0 else -1

    x_dummy = x_raw[dummy_index]
    z = {idx: int((1 - (x_raw[idx] * x_dummy)) / 2) for idx in range(n)}

    improved = True
    while improved:
        improved = False
        for u in range(n):
            delta_z = 1 - 2 * z[u]
            delta_E = Q[u, u] * delta_z
            for v in range(n):
                if u < v:
                    delta_E += Q[u, v] * z[v] * delta_z
                elif v < u:
                    delta_E += Q[v, u] * z[v] * delta_z

            if delta_E < -1e-6:
                z[u] = 1 - z[u]
                improved = True

    bin_str = "".join([str(z[i]) for i in range(n)])[::-1]
    state_int = int(bin_str, 2)

    return {state_int: 1}
