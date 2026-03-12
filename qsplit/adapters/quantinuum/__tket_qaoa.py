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

from qsplit.adapters.quantinuum.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


def __tket_solve(qubo: QUBO, backend, backend_optimizer) -> pd.DataFrame:
    measured_circ, var_to_qubit, all_vars = get_qaoa_circuit_optimized(qubo=qubo, backend=backend_optimizer)
    counts_int = run_quantum_optimizer(optimized_circuit=measured_circ, backend=backend)

    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
