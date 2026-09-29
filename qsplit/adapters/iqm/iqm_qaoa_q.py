import pandas as pd
from iqm.qiskit_iqm import IQMProvider

from qsplit import configuration
from qsplit.adapters.iqm.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    url = configuration.get("IQM_SERVER_URL", "https://resonance.meetiqm.com/")
    token = configuration.require("IQM_TOKEN")
    qc = configuration.get("IQM_QUANTUM_COMPUTER", "garnet")

    backend = IQMProvider(url=url, token=token, quantum_computer=qc).get_backend()
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(backend=backend, qubo=qubo)
    counts = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts, qubo, var_to_qubit, all_vars)
