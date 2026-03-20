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

import math
import os
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd
import qnexus as qnx
import sympy as sp
from pytket import Circuit, Qubit
from pytket.extensions.qiskit import AerBackend
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator, gen_term_sequence_circuit, get_operator_expectation_value
from scipy.optimize import minimize


class TKET_BACKEND(Enum):
    LOCAL = 0
    QNEXUS = 1


def __get_variables_mapping(qubo):
    all_vars = sorted(list(set(qubo.rows_idx) | set(qubo.cols_idx)))
    var_to_qubit = {v: i for i, v in enumerate(all_vars)}
    return var_to_qubit, all_vars


def __from_qubo_matrix_to_circuit(qubo):
    var_to_qubit, all_vars = __get_variables_mapping(qubo)
    op_dict = {}

    for i, r in enumerate(qubo.rows_idx):
        for j, c in enumerate(qubo.cols_idx):
            coeff = float(qubo.mat[i, j])
            if coeff == 0:
                continue

            qr = Qubit(var_to_qubit[r])
            if r == c:
                qp = QubitPauliString([qr], [Pauli.Z])
            else:
                qc = Qubit(var_to_qubit[c])
                qp = QubitPauliString([qr, qc], [Pauli.Z, Pauli.Z])

            op_dict[qp] = op_dict.get(qp, 0) + coeff

    return QubitPauliOperator(op_dict), var_to_qubit, all_vars


def get_qaoa_circuit_optimized(qubo, backend, reps: int = 2):
    operator, var_to_qubit, all_vars = __from_qubo_matrix_to_circuit(qubo)
    n = len(all_vars)
    circ = Circuit(n)
    for q in range(n):
        circ.H(q)

    betas = [sp.Symbol(f"beta_{i}") for i in range(reps)]
    gammas = [sp.Symbol(f"gamma_{i}") for i in range(reps)]

    for r in range(reps):
        scaled = {k: v * 2 * gammas[r] / sp.pi for k, v in operator._dict.items()}
        scaled_op = QubitPauliOperator(scaled)
        term_circ = gen_term_sequence_circuit(scaled_op, Circuit(n))
        circ.append(term_circ)

        for q in range(n):
            circ.Rx(2 * betas[r] / sp.pi, q)
    circ.measure_all()

    optimized = _optimize(circ, operator, backend)
    measured = optimized.copy()
    measured.measure_all()

    return measured, var_to_qubit, all_vars


def __evaluate_pauli_z(bitstring, operator):
    bitstring = bitstring[::-1]
    total = 0.0

    for qps, coeff in operator._dict.items():
        term_val = 1.0
        for qubit, pauli in qps.map.items():
            if pauli != Pauli.Z:
                continue
            idx = qubit.index[0]
            bit = bitstring[idx]
            term_val *= 1 if bit == "0" else -1
        total += coeff * term_val

    return total


def __compute_expectation_from_counts(counts, operator):
    exp = 0
    shots = sum(counts.values())

    for bitstring, freq in counts.items():
        z_val = __evaluate_pauli_z(bitstring, operator)
        exp += z_val * freq / shots

    return exp


def _optimize(circ, operator, backend):
    symbols = sorted(circ.free_symbols(), key=str)

    if not symbols:
        return circ

    init = [math.pi / 2 if "beta" in str(s) else math.pi for s in symbols]

    def objective(params):
        subs = {s: float(v) for s, v in zip(symbols, params)}
        c = circ.copy()
        c.symbol_substitution(subs)
        if isinstance(backend, AerBackend):
            val = get_operator_expectation_value(c, operator, backend, n_shots=100)
            return float(np.real_if_close(val))
        else:
            project = qnx.projects.get_or_create("qsplit-qaoa")
            config = qnx.QuantinuumConfig(device_name=os.getenv("QNEXUS_QPU"))
            name = f"qaoa-opt-{datetime.now(timezone.utc).isoformat()}"
            ref = qnx.circuits.upload(c, name=f"opt-{name}", project=project)
            compiled = qnx.compile([ref], name=name, backend_config=config, project=project)
            result = qnx.execute(
                compiled, n_shots=[100], backend_config=config, name=f"execute-{name}", project=project, timeout=None
            )[0]
            return __compute_expectation_from_counts(result.get_counts(), operator)

    res = minimize(objective, np.array(init), method="COBYLA", options={"maxiter": 5}, tol=1e-4)
    best = res.x
    final = circ.copy()
    final.symbol_substitution({s: float(v) for s, v in zip(symbols, best)})
    return final


def run_quantum_optimizer(optimized_circuit, backend):
    if isinstance(backend, AerBackend):
        backend_circuit = backend.get_compiled_circuit(optimized_circuit)
        result = backend.run_circuit(backend_circuit, n_shots=100)
    else:
        project = qnx.projects.get_or_create("qsplit-qaoa")
        config = qnx.QuantinuumConfig(device_name=os.getenv("QNEXUS_QPU"))
        name = f"qaoa-{datetime.now(timezone.utc).isoformat()}"
        ref = qnx.circuits.upload(optimized_circuit, name=f"compile-{name}", project=project)
        compiled = qnx.compile([ref], name=name, backend_config=config, project=project)
        result = qnx.execute(
            compiled, n_shots=[100], backend_config=config, name=f"execute-{name}", project=project, timeout=None
        )[0]

    return _counts_to_int(result.get_counts())


def _counts_to_int(raw_counts):
    out = {}
    for k, v in raw_counts.items():
        if isinstance(k, tuple):
            bitstr = "".join(str(b) for b in k[::-1])
        else:
            bitstr = k[::-1]
        out[int(bitstr, 2)] = v
    return out


def to_dataframe(counts_int, qubo, var_to_qubit, all_vars):
    data = []
    n = len(all_vars)

    for state_int, count in counts_int.items():
        bin_str = np.binary_repr(state_int, width=n)
        sol = np.array([int(b) for b in bin_str])[::-1]
        sol_dict = {v: sol[i] for v, i in var_to_qubit.items()}

        vec_row = np.array([sol_dict[r] for r in qubo.rows_idx])
        vec_col = np.array([sol_dict[c] for c in qubo.cols_idx])
        energy = vec_row @ qubo.mat @ vec_col.T + qubo.offset

        row = sol_dict.copy()
        row["energy"] = energy
        data.append(row)

    res = pd.DataFrame(data)
    res = res.sort_values("energy")
    best = res["energy"].min()
    return res[res["energy"] == best]
