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
    quantum_tune = os.getenv("QUANTUM_TUNE_IQM") == "True"

    backend = IQMProvider(url=url, quantum_computer=qc).get_backend()
    backend_optimizer = backend if quantum_tune else IQMFakeGarnet()
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(backend=backend_optimizer, qubo=qubo)
    counts = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)
