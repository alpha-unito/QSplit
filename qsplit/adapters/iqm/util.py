# Copyright (C) 2025  The QSplit Contributors.
# See the 'CONTRIBUTORS' file at the top-level directory of this distribution.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from qsplit.qubo import QUBO


def __get_variables_mapping(qubo: QUBO) -> tuple[dict[int, int], list[int]]:
    all_vars = sorted(list(set(qubo.rows_idx) | set(qubo.cols_idx)))
    var_to_qubit = {var: i for i, var in enumerate(all_vars)}
    return var_to_qubit, all_vars


def __from_qubo_matrix_to_circuit(qubo: QUBO) -> tuple[QuantumCircuit, dict[int, int], list[int]]:
    var_to_qubit, all_vars = __get_variables_mapping(qubo)
    num_qubits = len(all_vars)
    pauli_list = []

    for i, row_var in enumerate(qubo.rows_idx):
        for j, col_var in enumerate(qubo.cols_idx):
            coeff = qubo.mat[i, j]
            if coeff == 0:
                continue
            if row_var == col_var:
                pauli_list.append(("Z", [var_to_qubit[row_var]], coeff))
            else:
                pauli_list.append(("ZZ", [var_to_qubit[row_var], var_to_qubit[col_var]], coeff))

    if qubo.offset != 0:
        pauli_list.append(("I" * num_qubits, list(range(num_qubits)), qubo.offset))

    cost_hamiltonian = SparsePauliOp.from_sparse_list(pauli_list, num_qubits).simplify()

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    circuit.measure_all()

    return circuit, var_to_qubit, all_vars


def __compute_expectation(counts: dict[str, int], qubo: QUBO, var_to_qubit: dict[int, int]) -> float:
    avg_energy = 0
    total_shots = 0

    for bitstring, count in counts.items():
        full_solution = np.array([int(bit) for bit in bitstring])[::-1]
        sol_dict = {var_name: full_solution[q_idx] for var_name, q_idx in var_to_qubit.items()}
        vec_row = np.array([sol_dict[r] for r in qubo.rows_idx])
        vec_col = np.array([sol_dict[c] for c in qubo.cols_idx])
        energy = vec_row @ qubo.mat @ vec_col.T + qubo.offset
        avg_energy += energy * count
        total_shots += count

    return avg_energy / total_shots if total_shots > 0 else 0


def get_qaoa_circuit_optimized(backend, qubo: QUBO) -> tuple[QuantumCircuit, dict[int, int], list[int]]:
    circuit, var_to_qubit, all_vars = __from_qubo_matrix_to_circuit(qubo)
    transpiled_circuit = transpile_to_IQM(circuit, backend=backend)

    def objective_function(theta: list[float]) -> float:
        bound_circ = transpiled_circuit.assign_parameters(theta)
        job = backend.run(bound_circ, shots=500)
        counts = job.result().get_counts()
        return __compute_expectation(counts, qubo, var_to_qubit)

    init_params = [1.0, -1.0, 1.0, -1.0]
    result = minimize(objective_function, init_params, method="COBYLA", options={"maxiter": 100}, tol=1e-2)
    optimized_circ = transpiled_circuit.assign_parameters(result.x)

    return optimized_circ, var_to_qubit, all_vars


def run_quantum_optimizer(backend: IQMProvider, optimized_circuit: QuantumCircuit) -> dict[str, int]:
    job = backend.run(optimized_circuit, shots=500)
    return job.result().get_counts()


def to_dataframe(counts: dict[str, int], qubo: QUBO, var_to_qubit: dict[int, int], all_vars: list[int]) -> pd.DataFrame:
    data = []
    num_qubits = len(all_vars)

    for bitstring, count in counts.items():
        bitstring = bitstring.zfill(num_qubits)
        full_solution = np.array([int(bit) for bit in bitstring])[::-1]
        sol_dict = {var_name: full_solution[q_idx] for var_name, q_idx in var_to_qubit.items()}
        vec_row = np.array([sol_dict[r] for r in qubo.rows_idx])
        vec_col = np.array([sol_dict[c] for c in qubo.cols_idx])
        energy = vec_row @ qubo.mat @ vec_col.T + qubo.offset
        row = sol_dict.copy()
        row["energy"] = energy
        row["counts"] = count
        data.append(row)

    res = pd.DataFrame(data)
    res = res.groupby([c for c in res.columns if c not in ["counts"]], as_index=False).sum()
    res = res.sort_values(by="energy", ascending=True)
    cols = [c for c in res.columns if c not in ["energy", "counts"]]
    cols.sort()
    res = res[cols + ["energy", "counts"]]
    best_energy = res["energy"].min()
    return res[res["energy"] == best_energy]
