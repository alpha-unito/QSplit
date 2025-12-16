import os

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from qsplit.adapters.ibm.__ibm_qaoa import ibm_solve
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_solve(qubo, QiskitRuntimeService(channel='ibm_cloud', token=os.environ["TOKEN_IBM"],
                                                instance=os.environ["CRN_IBM"]).least_busy())
