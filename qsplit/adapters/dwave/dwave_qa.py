import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite

from qsplit import configuration
from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO


@configuration.configured
def solve(qubo: QUBO) -> pd.DataFrame:
    # GHSA-w8v5-vhqr-4h9v: DiskCache has no patched release; avoid its pickle cache.
    return to_dataframe(
        EmbeddingComposite(DWaveSampler(token=configuration.require("DWAVE_API_TOKEN"), cache_home="off")).sample(
            from_qubo_matrix_to_bqm(qubo), num_reads=10
        ),
        qubo,
    )
