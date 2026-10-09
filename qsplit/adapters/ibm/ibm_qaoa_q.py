import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from qsplit import configuration
from qsplit.adapters.ibm.__ibm_qaoa import ibm_solve
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    return ibm_solve(
        qubo,
        QiskitRuntimeService(
            channel="ibm_cloud", token=configuration.require("TOKEN_IBM"), instance=configuration.require("CRN_IBM")
        ).least_busy(),
    )
