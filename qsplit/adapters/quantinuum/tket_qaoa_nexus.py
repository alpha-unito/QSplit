import pandas as pd
import qnexus as qnx

from qsplit import configuration
from qsplit.adapters.quantinuum.__tket_qaoa import __tket_solve
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    qnx.auth.login_no_interaction(configuration.require("QNEXUS_USER"), configuration.require("QNEXUS_PASSWORD"))
    return __tket_solve(qubo=qubo, backend=None)
