import dimod
import numpy as np
import pandas as pd
from dimod import BinaryQuadraticModel, SampleSet

from qsplit.qubo import QUBO


def from_qubo_matrix_to_bqm(qubo: QUBO) -> BinaryQuadraticModel:
    linear = np.diag(qubo.mat).astype(np.float64)
    q_rows, q_cols = np.triu_indices_from(qubo.mat, k=1)
    quad = qubo.mat[q_rows, q_cols].astype(np.float64)
    mask = quad != 0
    quadratic = (q_rows[mask], q_cols[mask], quad[mask])
    return BinaryQuadraticModel.from_numpy_vectors(linear, quadratic, offset=qubo.offset, vartype=dimod.BINARY)


def to_dataframe(sampleset: SampleSet) -> pd.DataFrame:
    res = sampleset.to_pandas_dataframe()
    res = res.drop(columns=['num_occurrences']).drop_duplicates().sort_values(by='energy', ascending=True)
    return res[res['energy'] == min(res['energy'])]
