import os

import numpy as np

from qsplit.qubo import QUBO


def vars_count(qubo: QUBO) -> int:
    rows_found, cols_found = np.nonzero(qubo.mat)
    variables_in_rows = qubo.rows_idx[rows_found]
    variables_in_cols = qubo.cols_idx[cols_found]
    valid = (variables_in_rows >= 0) & (variables_in_cols >= 0)
    unique_vars = np.unique(np.concatenate([variables_in_rows[valid], variables_in_cols[valid]]))
    return int(len(unique_vars))


def is_empty(qubo: QUBO) -> bool:
    return not np.any(qubo.mat[np.ix_(qubo.rows_idx >= 0, qubo.cols_idx >= 0)])


def is_sparse(qubo: QUBO, cut_dim=None) -> bool:
    if cut_dim:
        return vars_count(qubo) <= int(cut_dim)
    return vars_count(qubo) <= int(os.environ["CUT_DIM"])
