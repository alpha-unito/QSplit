import pandas as pd
from pytket.extensions.qiskit import AerBackend

from qsplit.adapters.quantinuum.__tket_qaoa import __tket_solve
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    backend = AerBackend()
    backend._qiskit_backend.set_options(method="matrix_product_state")
    return __tket_solve(qubo=qubo, backend=backend)
