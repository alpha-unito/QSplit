import os

import pandas as pd
import qnexus as qnx

from qsplit.adapters.quantinuum.__tket_qaoa import __tket_solve
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    qnx.auth.login_no_interaction(os.getenv("QNEXUS_USER"), os.getenv("QNEXUS_PASSWORD"))
    return __tket_solve(qubo=qubo, backend=None)
