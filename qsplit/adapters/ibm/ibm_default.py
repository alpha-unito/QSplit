import pandas as pd
from qiskit_aer import AerSimulator

from qsplit.adapters.ibm.__ibm_pce import ibm_pce
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_pce(qubo, AerSimulator())
