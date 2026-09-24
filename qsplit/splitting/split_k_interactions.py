import os

import numpy as np

from qsplit.qubo import QUBO


def split_problem(qubo: QUBO) -> list[QUBO]:
    cut_dim = min(int(os.environ["CUT_DIM"]), qubo.problem_size)
    res = []

    mat_abs = np.abs(qubo.mat)
    diag_mask = np.eye(qubo.problem_size, dtype=bool)
    interaction_matrix = mat_abs + mat_abs.T
    interaction_matrix[diag_mask] = 0.0

    for i in range(qubo.problem_size):
        num_neighbors = cut_dim - 1
        best_neighbors = np.argsort(interaction_matrix[i])[qubo.problem_size - num_neighbors :]
        sub_indices = np.append(best_neighbors, i)
        sub_indices = np.unique(sub_indices)

        sub_mat = qubo.mat[np.ix_(sub_indices, sub_indices)]
        sub_cols = qubo.cols_idx[sub_indices]
        sub_rows = qubo.rows_idx[sub_indices]

        sub_qubo = QUBO(mat=sub_mat, rows_idx=sub_rows, cols_idx=sub_cols)
        res.append(sub_qubo)

    return res
