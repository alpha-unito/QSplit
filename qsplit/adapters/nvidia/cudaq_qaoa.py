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
import logging
import os

import cudaq
import pandas as pd

from qsplit.adapters.nvidia.util import (
    from_qubo_matrix_to_circuit,
    optimize_circuit,
    run_quantum_optimizer,
    to_dataframe,
)
from qsplit.qubo import QUBO


TARGET_SEQUENCE: list[tuple[str, str]] = [
    ("nvidia", "fp64"),
    ("nvidia", "fp32"),
    ("tensornet", "fp32"),
]


def _solve_on_target(qubo: QUBO, target: str, option: str) -> pd.DataFrame:
    cudaq.set_target(target, option=option)
    circuit_bundle, cost_hamiltonian, var_to_qubit, all_vars = from_qubo_matrix_to_circuit(qubo)
    optimized_params = optimize_circuit(circuit_bundle, cost_hamiltonian)
    counts = run_quantum_optimizer(circuit_bundle, optimized_params)
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)


def solve(qubo: QUBO) -> pd.DataFrame:
    last_size_error: RuntimeError | None = None
    for target, option in TARGET_SEQUENCE:
        try:
            return _solve_on_target(qubo, target=target, option=option)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "architecture mismatch" in message or "invalid simulator requested" in message:
                continue
            if (
                "requested size is too big" in message
                or "insufficient workspace" in message
                or "cuda memory allocation" in message
            ):
                last_size_error = exc
                continue
            raise
    if last_size_error is not None:
        raise RuntimeError(
            "CUDA-Q could not fit the circuit with configured GPU targets "
            f"{TARGET_SEQUENCE}. Reduce subproblem size (lower cut_dim)."
        ) from last_size_error
    raise RuntimeError(f"No valid CUDA-Q GPU targets available: {TARGET_SEQUENCE}")
