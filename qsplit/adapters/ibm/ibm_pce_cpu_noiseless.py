import pandas as pd
from qiskit_aer import AerSimulator

from qsplit import configuration
from qsplit.adapters.ibm.__ibm_pce import ibm_solve
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_solve(qubo, AerSimulator())
