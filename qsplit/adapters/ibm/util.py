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

# Acknowledgement:
# Parts of this code are adapted from the official IBM Quantum documentation
# regarding the Quantum Approximate Optimization Algorithm (QAOA).
# Source: https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm
# Modifications have been made to tailor the implementation to local requirements.

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.passmanager import BasePassManager
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import EstimatorV2, IBMBackend
from qiskit_ibm_runtime import SamplerV2

from qsplit.qubo import QUBO


def __get_variables_mapping(qubo: QUBO) -> tuple[dict[int, int], list[int]]:
    all_vars = sorted(list(set(qubo.rows_idx) | set(qubo.cols_idx)))
    var_to_qubit = {var: i for i, var in enumerate(all_vars)}
    return var_to_qubit, all_vars


def __from_qubo_matrix_to_circuit(qubo: QUBO) -> tuple[QuantumCircuit, SparsePauliOp, dict[int, int], list[int]]:
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

    cost_hamiltonian = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)
    cost_hamiltonian = cost_hamiltonian.simplify()

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    circuit.measure_all()

    return circuit, cost_hamiltonian, var_to_qubit, all_vars


__objective_func_vals = []


def __optimize_circuit(backend: IBMBackend | AerSimulator, candidate_circuit: QuantumCircuit,
                       cost_hamiltonian: SparsePauliOp) -> QuantumCircuit:
    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
    estimator = EstimatorV2(backend)
    estimator.options.default_shots = 500
    if isinstance(backend, IBMBackend):
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

    def objective_function(params: list[float]) -> float:
        return __cost_func_estimator(params, candidate_circuit, cost_hamiltonian, estimator)

    optimizer = COBYLA()
    result = optimizer.minimize(fun=objective_function, x0=init_params)
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    return optimized_circuit


def __cost_func_estimator(params: list[float], ansatz: QuantumCircuit, hamiltonian: SparsePauliOp,
                          estimator: EstimatorV2) -> float:
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    __objective_func_vals.append(cost)
    return cost


def get_qaoa_circuit_optimized(backend: IBMBackend | AerSimulator, pm: BasePassManager, qubo: QUBO) -> tuple[
    QuantumCircuit, dict[int, int], list[int]]:
    circuit, cost_hamiltonian, var_to_qubit, all_vars = __from_qubo_matrix_to_circuit(qubo)
    candidate_circuit = pm.run(circuit)
    optimized_circ = __optimize_circuit(backend, candidate_circuit, cost_hamiltonian)
    return optimized_circ, var_to_qubit, all_vars


def run_quantum_optimizer(backend: IBMBackend | AerSimulator, optimized_circuit: QuantumCircuit) -> dict[int, int]:
    sampler = SamplerV2(mode=backend)
    if isinstance(backend, IBMBackend):
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=500)
    counts_int = job.result()[0].data.meas.get_int_counts()
    return counts_int


def to_dataframe(counts_int: dict[int, int], qubo: QUBO, var_to_qubit: dict[int, int],
                 all_vars: list[int]) -> pd.DataFrame:
    data = []
    num_qubits = len(all_vars)

    for state_int, count in counts_int.items():
        bin_str = np.binary_repr(state_int, width=num_qubits)
        full_solution = np.array([int(bit) for bit in bin_str])[::-1]
        sol_dict = {var_name: full_solution[q_idx] for var_name, q_idx in var_to_qubit.items()}
        vec_row = np.array([sol_dict[r] for r in qubo.rows_idx])
        vec_col = np.array([sol_dict[c] for c in qubo.cols_idx])
        energy = vec_row @ qubo.mat @ vec_col.T
        row = sol_dict.copy()
        row['energy'] = energy
        data.append(row)

    res = pd.DataFrame(data)
    res = res.sort_values(by='energy', ascending=True)
    cols = [c for c in res.columns if c not in ['energy']]
    cols.sort()
    res = res[cols + ['energy']]
    best_energy = res['energy'].min()

    return res[res['energy'] == best_energy]
