import pandas as pd
from qiskit import generate_preset_pass_manager

from qsplit.adapters.ibm.util import to_dataframe
from qsplit.adapters.ibm.util_qaoa import get_qaoa_circuit_optimized, run_quantum_optimizer
from qsplit.qubo import QUBO


def ibm_solve(qubo: QUBO, backend) -> pd.DataFrame:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    circuit, var_to_qubit, all_vars = get_qaoa_circuit_optimized(
        backend,
        pm,
        qubo,
    )
    counts_int = run_quantum_optimizer(backend, circuit)
    return to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
