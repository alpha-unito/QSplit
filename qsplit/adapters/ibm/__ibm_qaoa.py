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

import pandas as pd
from qiskit import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime import IBMBackend
except Exception:
    IBMBackend = None

from qsplit.adapters.ibm.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend) -> pd.DataFrame:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(backend, pm, qubo)
    counts_int = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
