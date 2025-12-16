import pandas as pd
from qiskit import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend

from qsplit.adapters.ibm.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend: IBMBackend | AerSimulator) -> pd.DataFrame:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(backend, pm, qubo)
    counts_int = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
