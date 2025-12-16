import pandas as pd
from qiskit_aer import AerSimulator

from qsplit.adapters.ibm.__ibm_qaoa import ibm_solve
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_solve(qubo, AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=None))
