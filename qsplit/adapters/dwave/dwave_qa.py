import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite

from qsplit import configuration
from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    return to_dataframe(
        EmbeddingComposite(DWaveSampler(token=configuration.require("DWAVE_API_TOKEN"))).sample(
            from_qubo_matrix_to_bqm(qubo), num_reads=10
        ),
        qubo,
    )
