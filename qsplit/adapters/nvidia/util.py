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

from typing import List

import cudaq
import numpy as np
import pandas as pd
from cudaq import spin
from scipy.optimize import minimize

from qsplit.qubo import QUBO

logger = logging.getLogger(__name__)


@cudaq.kernel
def __generic_qaoa_kernel(
    qubit_count: int,
    layers: int,
    hamiltonian_terms_src: List[int],
    hamiltonian_terms_tgt: List[int],
    hamiltonian_coeffs: List[float],
    linear_terms_qubits: List[int],
    linear_terms_coeffs: List[float],
    thetas: List[float],
):
    q = cudaq.qvector(qubit_count)
    h(q)

    for p in range(layers):
        gamma = thetas[2 * p]
        beta = thetas[2 * p + 1]

        for i in range(len(hamiltonian_terms_src)):
            u = hamiltonian_terms_src[i]
            v = hamiltonian_terms_tgt[i]
            coeff = hamiltonian_coeffs[i]
            x.ctrl(q[u], q[v])
            rz(2.0 * gamma * coeff, q[v])
            x.ctrl(q[u], q[v])

        for i in range(len(linear_terms_qubits)):
            u = linear_terms_qubits[i]
            coeff = linear_terms_coeffs[i]
            rz(2.0 * gamma * coeff, q[u])

        for i in range(qubit_count):
            rx(2.0 * beta, q[i])


def __get_variables_mapping(qubo: QUBO) -> tuple[dict[int, int], list[int]]:
    all_vars = sorted(list(set(qubo.rows_idx) | set(qubo.cols_idx)))
    var_to_qubit = {var: i for i, var in enumerate(all_vars)}
    return var_to_qubit, all_vars


def from_qubo_matrix_to_circuit(qubo: QUBO) -> tuple[tuple, cudaq.SpinOperator, dict[int, int], list[int]]:
    var_to_qubit, all_vars = __get_variables_mapping(qubo)
    num_qubits = len(all_vars)
    quad_src = []
    quad_tgt = []
    quad_coeffs = []
    lin_qubits = []
    lin_coeffs = []
    cost_hamiltonian = 0.0 * spin.i(0)

    for i, row_var in enumerate(qubo.rows_idx):
        for j, col_var in enumerate(qubo.cols_idx):
            coeff = qubo.mat[i, j]
            if coeff == 0:
                continue

            u = var_to_qubit[row_var]
            v = var_to_qubit[col_var]

            if row_var == col_var:
                cost_hamiltonian += coeff * spin.z(u)
                lin_qubits.append(u)
                lin_coeffs.append(coeff)
            else:
                cost_hamiltonian += coeff * spin.z(u) * spin.z(v)
                quad_src.append(u)
                quad_tgt.append(v)
                quad_coeffs.append(coeff)

    layers = 2
    circuit_args = (num_qubits, layers, quad_src, quad_tgt, quad_coeffs, lin_qubits, lin_coeffs)
    circuit_bundle = (__generic_qaoa_kernel, circuit_args)
    return circuit_bundle, cost_hamiltonian, var_to_qubit, all_vars


def optimize_circuit(circuit_bundle: tuple, cost_hamiltonian: cudaq.SpinOperator) -> list[float]:
    kernel, kernel_args = circuit_bundle
    layers = kernel_args[1]
    parameter_count = 2 * layers

    def objective_function(params: list[float] | np.ndarray) -> float:
        value = float(cudaq.observe(kernel, cost_hamiltonian, *kernel_args, list(params)).expectation())
        if not np.isfinite(value):
            return 1e12
        return value

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, parameter_count)
    objective_function(initial_parameters)
    if hasattr(optimizer, "initial_parameters"):
        optimizer.initial_parameters = initial_parameters.tolist()
    try:
        result = optimizer.optimize(
            dimensions=parameter_count,
            function=objective_function,
        )
    except RuntimeError as exc:
        if "nlopt failure" not in str(exc).lower():
            raise
        logger.warning("CUDA-Q COBYLA failed with nlopt failure. Retrying once.")
        if hasattr(optimizer, "initial_parameters"):
            optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, parameter_count).tolist()
        try:
            result = optimizer.optimize(
                dimensions=parameter_count,
                function=objective_function,
            )
        except RuntimeError as exc2:
            if "nlopt failure" not in str(exc2).lower():
                raise
            logger.warning("CUDA-Q COBYLA failed again. Falling back to scipy.optimize (Nelder-Mead).")
            scipy_result = minimize(
                objective_function,
                x0=initial_parameters,
                method="Nelder-Mead",
                options={"maxiter": 120},
            )
            if not scipy_result.success:
                raise RuntimeError(
                    f"CUDA-Q and scipy optimizers failed: {scipy_result.message}"
                ) from exc2
            return scipy_result.x.tolist()
    return result[1]


def run_quantum_optimizer(circuit_bundle: tuple, optimized_params: list[float]) -> cudaq.SampleResult:
    kernel, kernel_args = circuit_bundle
    counts = cudaq.sample(kernel, *kernel_args, optimized_params, shots_count=500)
    return counts


def to_dataframe(
    counts: cudaq.SampleResult, qubo: QUBO, var_to_qubit: dict[int, int], all_vars: list[int]
) -> pd.DataFrame:
    data = []
    for bitstring, count in counts.items():
        bits = [int(x) for x in bitstring[::-1]]
        if len(bits) < len(all_vars):
            bits += [0] * (len(all_vars) - len(bits))

        sol_dict = {var_name: bits[q_idx] for var_name, q_idx in var_to_qubit.items()}
        vec_row = np.array([sol_dict[r] for r in qubo.rows_idx])
        vec_col = np.array([sol_dict[c] for c in qubo.cols_idx])
        energy = vec_row @ qubo.mat @ vec_col.T
        row = sol_dict.copy()
        row["energy"] = energy
        data.append(row)

    res = pd.DataFrame(data)

    if res.empty:
        return pd.DataFrame()

    res = res.sort_values(by="energy", ascending=True)
    cols = [c for c in res.columns if c not in ["energy"]]
    cols.sort()
    res = res[cols + ["energy"]]
    best_energy = res["energy"].min()
    return res[res["energy"] == best_energy]


# Since cudaq are generated dynamically this is a work-around for the linter
if False:

    def h(q): ...
    def x(q): ...
    def rx(a, q): ...
    def rz(a, q): ...
