import math

import numpy as np

from qsplit import configuration
from qsplit.qubo import QUBO


@configuration.configured
def split_problem(qubo: QUBO) -> list[QUBO]:
    cut_dim = int(configuration.require("CUT_DIM"))
    stride = int(configuration.get("STRIDE", -1))
    if stride <= 0:
        stride = cut_dim

    n = qubo.problem_size
    if n <= cut_dim:
        padding_needed = cut_dim - n
    else:
        steps = math.ceil((n - cut_dim) / stride)
        padded_size = steps * stride + cut_dim
        padding_needed = padded_size - n

    if padding_needed > 0:
        qubo.mat = np.pad(qubo.mat, ((0, padding_needed), (0, padding_needed)), mode="constant", constant_values=0)
        new_indices = np.arange(-1, -(padding_needed + 1), -1)
        qubo.rows_idx = np.concatenate([qubo.rows_idx, new_indices])
        qubo.cols_idx = np.concatenate([qubo.cols_idx, new_indices])
        qubo.problem_size = qubo.mat.shape[0]

    res = []
    for i in range(0, qubo.problem_size - cut_dim + 1, stride):
        for j in range(i, qubo.problem_size - cut_dim + 1, stride):
            row_end = i + cut_dim
            col_end = j + cut_dim
            sub_mat = qubo.mat[i:row_end, j:col_end]

            if np.count_nonzero(sub_mat) == 0:
                continue

            sub_qubo = QUBO(sub_mat, qubo.rows_idx[i:row_end], qubo.cols_idx[j:col_end])
            res.append(sub_qubo)

    return res
