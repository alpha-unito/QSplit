from math import comb

import numpy as np

from qsplit import configuration
from qsplit.qubo import QUBO


def local_training_qubits() -> int:
    return int(configuration.get("LOCAL_TRAINING_QUBITS", 20))


def training_subproblem(qubo: QUBO, *, pce_k: int | None = None) -> QUBO:
    limit = local_training_qubits()
    if pce_k is not None:
        if limit < pce_k:
            raise ValueError(f"PCE training requires LOCAL_TRAINING_QUBITS >= {pce_k}")
        limit = 3 * comb(limit, pce_k) - 1
    variables = sorted(set(qubo.rows_idx) | set(qubo.cols_idx))
    if len(variables) <= limit:
        return qubo

    weights = dict.fromkeys(variables, 0.0)
    for var, weight in zip(qubo.rows_idx, np.abs(qubo.mat).sum(axis=1)):
        weights[var] += weight
    for var, weight in zip(qubo.cols_idx, np.abs(qubo.mat).sum(axis=0)):
        weights[var] += weight
    selected = sorted(sorted(variables, key=lambda var: (-weights[var], var))[:limit])
    positions = {var: i for i, var in enumerate(selected)}
    rows = [i for i, var in enumerate(qubo.rows_idx) if var in positions]
    cols = [i for i, var in enumerate(qubo.cols_idx) if var in positions]
    matrix = np.zeros((len(selected), len(selected)))
    matrix[np.ix_([positions[qubo.rows_idx[i]] for i in rows], [positions[qubo.cols_idx[i]] for i in cols])] = qubo.mat[
        np.ix_(rows, cols)
    ]
    indices = np.asarray(selected)
    return QUBO(matrix, indices, indices.copy(), offset=qubo.offset)
