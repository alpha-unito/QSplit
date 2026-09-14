import pandas as pd

from qsplit.adapters.quantinuum.util import get_qaoa_circuit_optimized, run_quantum_optimizer, to_dataframe
from qsplit.qubo import QUBO


def __tket_solve(qubo: QUBO, backend) -> pd.DataFrame:
    measured_circ, var_to_qubit, all_vars = get_qaoa_circuit_optimized(qubo=qubo, backend=backend)
    counts_int = run_quantum_optimizer(optimized_circuit=measured_circ, backend=backend)

    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
