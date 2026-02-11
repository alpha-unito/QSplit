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
