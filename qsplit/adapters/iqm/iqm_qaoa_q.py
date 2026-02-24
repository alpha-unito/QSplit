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

import os

import pandas as pd
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

from qsplit.adapters.iqm.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    url = os.getenv("IQM_SERVER_URL")
    _ = os.getenv("IQM_TOKEN")
    qc = os.getenv("IQM_QUANTUM_COMPUTER", "garnet")
    quantum_tune = os.getenv("QUANTUM_TUNE_QAOA") == "True"

    backend = IQMProvider(url=url, quantum_computer=qc).get_backend()
    backend_optimizer = backend if quantum_tune else IQMFakeGarnet()
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(backend=backend_optimizer, qubo=qubo)
    counts = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)
