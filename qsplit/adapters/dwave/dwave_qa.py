import pandas as pd
from dwave.system import EmbeddingComposite, DWaveSampler

from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    return to_dataframe(EmbeddingComposite(DWaveSampler()).sample(from_qubo_matrix_to_bqm(qubo), num_reads=10))
