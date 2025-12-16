import pandas as pd
from qiskit_aer import AerSimulator

from qsplit.adapters.ibm.__ibm_qaoa import ibm_solve
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_solve(qubo, AerSimulator(method='tensor_network', device='gpu', use_cuTensorNet_autotuning=True))
