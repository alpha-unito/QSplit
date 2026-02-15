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

import cudaq
import pandas as pd
from util import from_qubo_matrix_to_circuit, optimize_circuit, run_quantum_optimizer, to_dataframe

from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    cudaq.set_target("nvidia", option="fp64")
    circuit_bundle, cost_hamiltonian, var_to_qubit, all_vars = from_qubo_matrix_to_circuit(qubo)
    optimized_params = optimize_circuit(circuit_bundle, cost_hamiltonian)
    counts = run_quantum_optimizer(circuit_bundle, optimized_params)
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)
