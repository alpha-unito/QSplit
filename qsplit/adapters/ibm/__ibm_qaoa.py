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

from qsplit.adapters.ibm.util import to_dataframe
from qsplit.adapters.ibm.util_qaoa import get_qaoa_circuit_optimized, run_quantum_optimizer
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend, backend_optimizer=None, optimize_on_backend: bool = True) -> pd.DataFrame:
    if backend_optimizer is None:
        backend_optimizer = backend
    pm = generate_preset_pass_manager(backend=backend_optimizer, optimization_level=2)
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(
        backend_optimizer,
        pm,
        qubo,
        optimize_on_backend=optimize_on_backend,
    )
    counts_int = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
