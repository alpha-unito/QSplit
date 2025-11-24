from typing import Tuple

import numpy as np

from qsplit.qubo import QUBO


def split_problem(qubo: QUBO) -> Tuple[QUBO, QUBO, QUBO]:
    """
        Returns 3 sub-problems in qubo form.
        The 3 sub-problems correspond to the matrices obtained by dividing the qubo matrix of the original problem
        in half both horizontally and vertically.
        The sub-problem for the sub-matrix in the bottom left corner is not given as this is always empty.
        The order of the results is:
        - Upper left sub-matrix,
        - Upper right sub-matrix,
        - Lower right sub-matrix.

        All sub-problems are converted to obtain an upper triangular matrix.
    """
    split_idx = qubo.problem_size // 2

    ul_mat = qubo.mat[:split_idx, :split_idx]
    ur_mat = qubo.mat[:split_idx, split_idx:]
    lr_mat = qubo.mat[split_idx:, split_idx:]
    if not np.all(qubo.mat[split_idx:, :split_idx] == 0):
        raise ValueError("Lower left sub-matrix must be 0")

    res = (QUBO(ul_mat, cols_idx=qubo.cols_idx[:split_idx], rows_idx=qubo.rows_idx[:split_idx]),
           QUBO(ur_mat, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[:split_idx]),
           QUBO(lr_mat, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[split_idx:]))

    return res
