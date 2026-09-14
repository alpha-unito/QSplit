import os

import numpy as np

from qsplit.qubo import QUBO


def vars_count(qubo: QUBO) -> int:
    rows_found, cols_found = np.nonzero(qubo.mat)
    variables_in_rows = qubo.rows_idx[rows_found]
    variables_in_cols = qubo.cols_idx[cols_found]
    unique_vars = np.unique(np.concatenate([variables_in_rows, variables_in_cols]))
    return int(len(unique_vars))


def is_empty(qubo: QUBO) -> bool:
    return np.count_nonzero(qubo.mat) == 0 or qubo.problem_size == 0


def is_sparse(qubo: QUBO, cut_dim=None) -> bool:
    if cut_dim:
        return vars_count(qubo) <= int(cut_dim)
    return vars_count(qubo) <= int(os.environ["CUT_DIM"])
